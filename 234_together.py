import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from rdp import rdp
from typing import List, Tuple, Dict, Any

DEBUG = True  # Включить вывод отладочной информации


# ------------------------------------------------------------
# Вспомогательные функции (без изменений)
# ------------------------------------------------------------
def order_boundary_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not points:
        return []
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return ordered


def simplify_boundary(points: List[Tuple[float, float]], epsilon: float = 2.2) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points
    pts = np.array(points)
    simplified = rdp(pts, epsilon=epsilon, return_mask=False)
    return [tuple(p) for p in simplified]


def filter_close_points(points: List[Tuple[float, float]], min_dist: float = 0.5) -> List[Tuple[float, float]]:
    if not points:
        return []
    filtered = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(np.array(p) - np.array(filtered[-1])) > min_dist:
            filtered.append(p)
    return filtered


def edges_from_vertices(vertices: List[Tuple[float, float]]) -> List[Tuple[np.ndarray, float, int]]:
    n = len(vertices)
    edges = []
    for i in range(n):
        p1 = np.array(vertices[i])
        p2 = np.array(vertices[(i + 1) % n])
        vec = p2 - p1
        length = np.linalg.norm(vec)
        edges.append((vec, length, i))
    return edges


def find_most_parallel_edges(vertices: List[Tuple[float, float]]) -> Tuple[int, int]:
    edges = edges_from_vertices(vertices)
    n = len(edges)
    best_pair = None
    best_score = -1.0

    if DEBUG:
        print("\n" + "=" * 60)
        print("ПОИСК НАИБОЛЕЕ ПАРАЛЛЕЛЬНЫХ РЁБЕР")
        print("=" * 60)
        print(f"Всего рёбер: {n}")

    for i in range(n):
        for j in range(i + 1, n):
            vertices_i = {i, (i + 1) % n}
            vertices_j = {j, (j + 1) % n}
            if vertices_i & vertices_j:
                if DEBUG:
                    print(f"Ребро {i} и ребро {j} - СОСЕДНИЕ, пропускаем")
                continue

            v1, l1, _ = edges[i]
            v2, l2, _ = edges[j]
            if l1 * l2 == 0:
                continue
            cos = abs(np.dot(v1, v2)) / (l1 * l2)
            score = (l1 + l2) * cos

            if DEBUG:
                print(f"Ребро {i}: длина={l1:.2f}, вектор={v1}")
                print(f"Ребро {j}: длина={l2:.2f}, вектор={v2}")
                print(f"  cos={cos:.4f}, score={score:.2f}")

            if score > best_score:
                best_score = score
                best_pair = (i, j)
                if DEBUG:
                    print(f"  *** НОВЫЙ ЛУЧШИЙ! ***")

    if DEBUG and best_pair:
        print(f"\nВЫБРАНЫ РЁБРА: {best_pair[0]} и {best_pair[1]}")

    return best_pair


def line_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray):
    A = np.column_stack((d1, -d2))
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        return None, None
    b = p2 - p1
    ts = np.linalg.solve(A, b)
    t, s = ts
    inter_pt = p1 + t * d1
    return inter_pt, s


# ------------------------------------------------------------
# ИСПРАВЛЕННЫЕ ФУНКЦИИ (критические для работы)
# ------------------------------------------------------------
def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Проверяет, находится ли точка внутри полигона (алгоритм с лучом)."""
    x, y = point
    inside = False
    n = len(polygon)

    p0 = np.asarray(polygon[0])
    p_last = np.asarray(polygon[-1])
    is_closed = np.allclose(p0, p_last, atol=1e-6)

    for i in range(n - 1 if not is_closed else n):
        pt1 = np.asarray(polygon[i])
        pt2 = np.asarray(polygon[(i + 1) % n])
        x1, y1 = pt1
        x2, y2 = pt2

        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-10) + x1):
            inside = not inside

    return inside


def build_two_lines(vertices: List[Tuple[float, float]],
                    offset_distance: float = 2.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """НОВАЯ ВЕРСИЯ: строит две ПАРАЛЛЕЛЬНЫЕ прямые + выводит маршрут inner line."""
    if DEBUG:
        print("\n" + "=" * 60)
        print("ПОСТРОЕНИЕ ПРЯМЫХ (НОВАЯ ВЕРСИЯ — ПАРАЛЛЕЛЬНЫЕ ПРЯМЫЕ)")
        print("=" * 60)

    if len(vertices) < 4:
        if DEBUG:
            print("Слишком мало вершин для обработки")
        dummy = np.array([vertices[0], vertices[1]]) if len(vertices) >= 2 else np.zeros((2, 2))
        return dummy, dummy, [vertices[0], vertices[0]]

    e1_idx, e2_idx = find_most_parallel_edges(vertices)
    if e1_idx is None or e2_idx is None:
        if DEBUG:
            print("Не удалось найти параллельные рёбра")
        dummy = np.array([vertices[0], vertices[1]])
        return dummy, dummy, [vertices[0], vertices[0]]

    n = len(vertices)

    def edge_len(start_idx: int) -> float:
        p1 = np.array(vertices[start_idx])
        p2 = np.array(vertices[(start_idx + 1) % n])
        return np.linalg.norm(p2 - p1)

    len_e1 = edge_len(e1_idx)
    len_e2 = edge_len(e2_idx)
    if len_e1 <= len_e2:
        small_idx, large_idx = e1_idx, e2_idx
        small_len = len_e1
    else:
        small_idx, large_idx = e2_idx, e1_idx
        small_len = len_e2

    if DEBUG:
        print(f"Меньшее ребро: индекс {small_idx} (длина {small_len:.2f})")
        print(f"Большое ребро: индекс {large_idx} (длина {len_e2 if small_idx == e1_idx else len_e1:.2f})")

    def get_cw_path_info(from_idx: int, to_idx: int):
        path = []
        i = from_idx
        while True:
            path.append(i)
            if i == to_idx:
                break
            i = (i + 1) % n
        intermediates = max(0, len(path) - 2)
        return path, intermediates

    e1_end = (e1_idx + 1) % n
    e2_end = (e2_idx + 1) % n

    _, num_int1 = get_cw_path_info(e1_end, e2_idx)
    _, num_int2 = get_cw_path_info(e2_end, e1_idx)

    if DEBUG:
        print(f"Дуга 1 (от {e1_end} до {e2_idx}): {num_int1} промежуточных вершин")
        print(f"Дуга 2 (от {e2_end} до {e1_idx}): {num_int2} промежуточных вершин")

    if num_int1 >= num_int2:
        longer_from_idx, longer_to_idx = e1_end, e2_idx
    else:
        longer_from_idx, longer_to_idx = e2_end, e1_idx

    if DEBUG:
        print(f"Выбрана длинная дуга: от вершины {longer_from_idx} до {longer_to_idx}")

    # ----- ВЫВОД МАРШРУТА ДЛЯ INNER LINE (line1) -----
    path_indices, _ = get_cw_path_info(longer_from_idx, longer_to_idx)
    path_vertices = [vertices[i] for i in path_indices]
    if DEBUG:
        print("\nМаршрут inner line (line1):")
        arrow = " -> ".join([f"({v[0]:.1f},{v[1]:.1f})" for v in path_vertices])
        print(f"  {arrow}")
        indices_str = " -> ".join(map(str, path_indices))
        print(f"  Индексы вершин: {indices_str}\n")
    # -------------------------------------------------

    small_endpoints = {small_idx, (small_idx + 1) % n}
    if longer_from_idx in small_endpoints:
        chosen_small_idx = longer_from_idx
        chosen_large_idx = longer_to_idx
    elif longer_to_idx in small_endpoints:
        chosen_small_idx = longer_to_idx
        chosen_large_idx = longer_from_idx
    else:
        chosen_small_idx = small_idx
        chosen_large_idx = large_idx

    line1_start_pt = vertices[chosen_small_idx]
    line1_end_pt = vertices[chosen_large_idx]
    line1 = np.array([line1_start_pt, line1_end_pt])

    d = max(small_len - 1.0, 0.5)

    if DEBUG:
        print(f"Первая прямая (line1): {line1_start_pt} → {line1_end_pt}")
        print(f"Смещение второй прямой: d = {d:.2f} (теоретическое)")

    dir_vec = line1[1] - line1[0]
    line_len = np.linalg.norm(dir_vec)
    if line_len < 1e-6:
        if DEBUG:
            print("Вырожденная первая прямая")
        dummy = line1.copy()
        polygon = [tuple(line1[0]), tuple(line1[1]), tuple(line1[0])]
        return line1, dummy, polygon

    unit_dir = dir_vec / line_len
    perp1 = np.array([-unit_dir[1], unit_dir[0]])
    perp2 = -perp1

    mid = (line1[0] + line1[1]) / 2
    center = np.mean([np.array(p) for p in vertices], axis=0)

    vec_to_center = center - mid
    norm = np.linalg.norm(vec_to_center)
    if norm > 1e-8:
        unit_to_center = vec_to_center / norm
    else:
        unit_to_center = np.array([0., 0.])

    dot1 = np.dot(perp1, unit_to_center)
    dot2 = np.dot(perp2, unit_to_center)
    chosen_perp = perp1 if dot1 > dot2 else perp2

    if DEBUG:
        direction = "ВНУТРЬ (к центру)" if np.dot(chosen_perp, vec_to_center) > 0 else "НАРУЖУ"
        print(f"Выбрано направление смещения: {direction}")

    small_start = np.array(vertices[small_idx])
    small_end = np.array(vertices[(small_idx + 1) % n])
    small_mid = (np.array(vertices[small_idx]) + np.array(vertices[(small_idx + 1)])) / 2.0

    large_p = np.array(vertices[large_idx])
    large_vec = np.array(vertices[(large_idx + 1) % n]) - large_p

    inter_pos, s_pos = line_intersection(small_mid, dir_vec, large_p, large_vec)
    inter_neg, s_neg = line_intersection(small_mid, -dir_vec, large_p, large_vec)

    candidates = []
    if inter_pos is not None and s_pos is not None and 0.0 <= s_pos <= 1.0:
        candidates.append((inter_pos, s_pos, 1.0))
    if inter_neg is not None and s_neg is not None and 0.0 <= s_neg <= 1.0:
        candidates.append((inter_neg, s_neg, -1.0))

    if candidates:
        inter_large, _, _ = candidates[0]
    else:
        if inter_pos is not None:
            inter_large = inter_pos
        else:
            inter_large = small_mid

    line2 = np.array([small_mid, inter_large])

    if DEBUG:
        vec2 = line2[1] - line2[0]
        actual_cos = np.dot(vec2, dir_vec) / (
                np.linalg.norm(vec2) * np.linalg.norm(dir_vec) + 1e-8
        )
        print(f"Вторая прямая (line2): {line2[0]} → {line2[1]}")
        print(f"  Проверка параллельности: cos = {actual_cos:.6f} (должен быть ≈ 1.000000)")
        print(f"  Начинается из середины меньшего ребра: {small_mid}")
        print(f"  Заканчивается на втором ребре (s ≈ {s_pos if 's_pos' in locals() else s_neg:.3f})")

    small_attach1 = line1[0]
    large_attach1 = line1[1]
    small_attach2 = line2[0]
    large_attach2 = line2[1]

    poly_cand1 = [small_attach1, large_attach1, large_attach2, small_attach2]
    poly_cand2 = [small_attach1, small_attach2, large_attach2, large_attach1]

    mid_strip = (small_attach1 + large_attach1 + small_attach2 + large_attach2) / 4.0

    chosen_polygon_list = poly_cand1
    for cand in [poly_cand1, poly_cand2]:
        poly_closed = cand + [cand[0]] if not np.allclose(np.asarray(cand[0]), np.asarray(cand[-1]),
                                                          atol=1e-6) else cand
        if point_in_polygon(tuple(mid_strip), poly_closed):
            chosen_polygon_list = cand
            break

    polygon = [tuple(p) for p in chosen_polygon_list]
    if not np.allclose(np.asarray(polygon[0]), np.asarray(polygon[-1]), atol=1e-6):
        polygon.append(polygon[0])

    if DEBUG:
        print(f"\nМНОГОУГОЛЬНИК (keep region, {len(polygon)} точек):")
        for i, pt in enumerate(polygon):
            print(f"  [{i}]: {pt}")
        print(f"ВНУТРЕННЯЯ ПРЯМАЯ (line1): {line1[0]} → {line1[1]}")
        print(f"ПАРАЛЛЕЛЬНАЯ ПРЯМАЯ (line2): {line2[0]} → {line2[1]}")

    return line1, line2, polygon
# ------------------------------------------------------------
# ВЕРНУТЫЕ ОРИГИНАЛЬНЫЕ ФУНКЦИИ ОТРИСОВКИ (точно как в вашем первом файле)
# ------------------------------------------------------------
def visualize_result_with_lines(original_sp: Dict[str, Any],
                                reassigned: Dict[int, List[Tuple[float, float]]],
                                neighbor_reassigned: Dict[int, List[Tuple[float, float]]] = None,
                                data: Dict[str, Any] = None,
                                offset_distance: float = 2.0):
    """Оригинальная трёхоконная визуализация (Этап 1 → Этап 2 → Этап 3)"""
    # Исходный суперпиксель
    orig_boundary = [(p["x"], p["y"]) for p in original_sp["boundary_points"]]
    orig_ordered = order_boundary_points(orig_boundary)
    orig_vertices = simplify_boundary(orig_ordered, epsilon=2.2)
    orig_vertices = filter_close_points(orig_vertices)

    line1, line2, polygon = build_two_lines(orig_vertices, offset_distance)

    neighbor_ids = original_sp["neighbors"]
    neighbor_boundaries = {}
    neighbor_centers = {}
    for nid in neighbor_ids:
        sp = next((s for s in data["superpixels"] if s["id"] == nid), None)
        if sp:
            bound = [(p["x"], p["y"]) for p in sp["boundary_points"]]
            bound_ord = order_boundary_points(bound)
            bound_simp = simplify_boundary(bound_ord, epsilon=2.2)
            neighbor_boundaries[nid] = filter_close_points(bound_simp)
            neighbor_centers[nid] = (sp["center"]["x"], sp["center"]["y"])

    center = (original_sp["center"]["x"], original_sp["center"]["y"])

    first_neighbor_id = neighbor_ids[0] if neighbor_ids else None

    # Первый сосед
    first_neighbor_sp = None
    first_neighbor_vertices = None
    first_neighbor_line1 = None
    first_neighbor_line2 = None
    first_neighbor_polygon = None
    first_neighbor_center = None
    first_neighbor_points = []

    if first_neighbor_id and data:
        first_neighbor_sp = next((s for s in data["superpixels"] if s["id"] == first_neighbor_id), None)
        if first_neighbor_sp:
            first_neighbor_bound = [(p["x"], p["y"]) for p in first_neighbor_sp["boundary_points"]]
            first_neighbor_ordered = order_boundary_points(first_neighbor_bound)
            first_neighbor_simplified = simplify_boundary(first_neighbor_ordered, epsilon=2.2)
            first_neighbor_vertices = filter_close_points(first_neighbor_simplified)

            first_neighbor_line1, first_neighbor_line2, first_neighbor_polygon = build_two_lines(
                first_neighbor_vertices, offset_distance)
            first_neighbor_center = (first_neighbor_sp["center"]["x"], first_neighbor_sp["center"]["y"])

    # Объединяем результаты
    all_reassigned = {}
    if reassigned:
        for k, v in reassigned.items():
            if k != '_lines':
                all_reassigned.setdefault(k, []).extend(v)
    if neighbor_reassigned:
        for k, v in neighbor_reassigned.items():
            if k != '_lines':
                all_reassigned.setdefault(k, []).extend(v)
                if k == first_neighbor_id:
                    first_neighbor_points.extend(v)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # График 1: Исходное состояние
    ax1 = axes[0]
    poly_orig = np.array(orig_vertices + [orig_vertices[0]])
    ax1.plot(poly_orig[:, 0], poly_orig[:, 1], 'b-', linewidth=2, label=f'SP {original_sp["id"]}')
    ax1.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=2, alpha=0.8, label='Inner line')
    ax1.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=2, alpha=0.8, label='Outer line')
    poly_array = np.array(polygon)
    ax1.fill(poly_array[:, 0], poly_array[:, 1], alpha=0.3, color='yellow', label='Keep polygon')
    ax1.plot(poly_array[:, 0], poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        ax1.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.5,
                 label=f'Neighbor {nid}' if nid == neighbor_ids[0] else "")

    all_pts = np.array([(p["x"], p["y"]) for p in original_sp["all_points"]])
    ax1.scatter(all_pts[:, 0], all_pts[:, 1], c='blue', s=5, alpha=0.5)
    ax1.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=80, marker='*', label='Center', zorder=5)
    ax1.annotate(str(original_sp["id"]), xy=center, xytext=(5, 5), textcoords='offset points',
                 fontsize=10, fontweight='bold', color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))

    for nid, ncenter in neighbor_centers.items():
        ax1.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=50, marker='o', alpha=0.7)
        ax1.annotate(str(nid), xy=ncenter, xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

    ax1.set_title("Этап 1: Исходное состояние\n(Жёлтый многоугольник - сохраняемые точки)")
    ax1.axis('equal')
    ax1.legend(loc='upper right', fontsize=8)

    # График 2: После обработки выбранного суперпикселя
    ax2 = axes[1]
    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        ax2.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.5, linewidth=1)

    ax2.plot(poly_orig[:, 0], poly_orig[:, 1], 'b--', linewidth=1.5, alpha=0.5, label=f'Original boundary SP {original_sp["id"]}')
    ax2.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=2, alpha=0.8, label='Inner line')
    ax2.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=2, alpha=0.8, label='Outer line')
    ax2.fill(poly_array[:, 0], poly_array[:, 1], alpha=0.3, color='yellow')
    ax2.plot(poly_array[:, 0], poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

    ax2.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=80, marker='*', label='Center', zorder=5)
    ax2.annotate(str(original_sp["id"]), xy=center, xytext=(5, 5), textcoords='offset points',
                 fontsize=10, fontweight='bold', color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))

    for nid, ncenter in neighbor_centers.items():
        ax2.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=50, marker='o', alpha=0.7)
        ax2.annotate(str(nid), xy=ncenter, xytext=(5, 5), textcoords='offset points',
                     fontsize=8, color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    unique_ids = [k for k in reassigned.keys() if k != '_lines']
    colors = [cmap(i % cmap.N) for i in range(len(unique_ids))]

    for idx, sp_id in enumerate(unique_ids):
        points = reassigned[sp_id]
        if points:
            pts = np.array(points)
            ax2.scatter(pts[:, 0], pts[:, 1], c=[colors[idx]], s=8, alpha=0.8, label=f'SP {sp_id}', edgecolors='none')

    ax2.set_title(f"Этап 2: После обработки SP {original_sp['id']}")
    ax2.axis('equal')
    ax2.legend(loc='upper right', fontsize=8, markerscale=1.5)

    # График 3: После обработки первого соседа
    ax3 = axes[2]
    all_centers = {s["id"]: (s["center"]["x"], s["center"]["y"]) for s in data["superpixels"]}

    for nid, verts in neighbor_boundaries.items():
        if nid != first_neighbor_id:
            poly = np.array(verts + [verts[0]])
            ax3.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.3, linewidth=1)

    ax3.plot(poly_orig[:, 0], poly_orig[:, 1], 'b--', linewidth=1.5, alpha=0.5, label=f'Boundary SP {original_sp["id"]}')
    ax3.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=1.5, alpha=0.5)
    ax3.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=1.5, alpha=0.5)

    if first_neighbor_vertices is not None:
        poly_neighbor = np.array(first_neighbor_vertices + [first_neighbor_vertices[0]])
        ax3.plot(poly_neighbor[:, 0], poly_neighbor[:, 1], 'orange', linewidth=2, alpha=0.8, label=f'Boundary SP {first_neighbor_id}')

        if first_neighbor_line1 is not None and first_neighbor_line2 is not None:
            ax3.plot(first_neighbor_line1[:, 0], first_neighbor_line1[:, 1], 'orange', linewidth=2, alpha=0.8, linestyle='-', label=f'Inner line SP {first_neighbor_id}')
            ax3.plot(first_neighbor_line2[:, 0], first_neighbor_line2[:, 1], 'orange', linewidth=2, alpha=0.8, linestyle='--', label=f'Outer line SP {first_neighbor_id}')

        if first_neighbor_polygon is not None:
            neighbor_poly_array = np.array(first_neighbor_polygon)
            ax3.fill(neighbor_poly_array[:, 0], neighbor_poly_array[:, 1], alpha=0.2, color='orange')
            ax3.plot(neighbor_poly_array[:, 0], neighbor_poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

        if first_neighbor_center is not None:
            ax3.scatter(first_neighbor_center[0], first_neighbor_center[1], c='orange', edgecolors='black',
                        linewidth=1.5, s=100, marker='*', label=f'Center SP {first_neighbor_id}', zorder=5)
            ax3.annotate(str(first_neighbor_id), xy=first_neighbor_center, xytext=(8, 8), textcoords='offset points',
                         fontsize=11, fontweight='bold', color='orange',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange", alpha=0.8))

    ax3.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1.5, s=100,
                marker='*', label=f'Center SP {original_sp["id"]}', zorder=5)
    ax3.annotate(str(original_sp["id"]), xy=center, xytext=(8, 8), textcoords='offset points',
                 fontsize=11, fontweight='bold', color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))

    if first_neighbor_points:
        pts = np.array(first_neighbor_points)
        if first_neighbor_center:
            center_np = np.array(first_neighbor_center)
            angles = np.arctan2(pts[:, 1] - center_np[1], pts[:, 0] - center_np[0])
            sorted_indices = np.argsort(angles)
            sorted_pts = pts[sorted_indices]
            sorted_pts = np.vstack([sorted_pts, sorted_pts[0]])
            ax3.plot(sorted_pts[:, 0], sorted_pts[:, 1], 'orange', linewidth=2, alpha=0.8,
                     label=f'Points connection SP {first_neighbor_id}')
        ax3.scatter(pts[:, 0], pts[:, 1], c='orange', s=15, alpha=0.8, edgecolors='black', linewidth=0.5,
                    label=f'Points SP {first_neighbor_id}')

    other_ids = set(all_reassigned.keys()) - {original_sp["id"], first_neighbor_id}
    for oid in other_ids:
        if oid in all_centers:
            oc = all_centers[oid]
            ax3.scatter(oc[0], oc[1], c='gray', edgecolors='black', linewidth=0.5, s=40, marker='o', alpha=0.5)
            ax3.annotate(str(oid), xy=oc, xytext=(5, 5), textcoords='offset points',
                         fontsize=7, color='gray',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.5))

    unique_ids_final = list(all_reassigned.keys())
    colors_final = [cmap(i % cmap.N) for i in range(len(unique_ids_final))]

    for idx, sp_id in enumerate(unique_ids_final):
        if sp_id == first_neighbor_id:
            continue
        points = all_reassigned[sp_id]
        if points:
            pts = np.array(points)
            ax3.scatter(pts[:, 0], pts[:, 1], c=[colors_final[idx]], s=8, alpha=0.8,
                        label=f'SP {sp_id}', edgecolors='none')

    ax3.set_title(f"Этап 3: После обработки SP {first_neighbor_id}\n(Оранжевая линия - точки SP {first_neighbor_id})")
    ax3.axis('equal')
    ax3.legend(loc='upper right', fontsize=8, markerscale=1.5)

    plt.tight_layout()
    plt.show()


def visualize_intermediate_with_lines(original_sp: Dict[str, Any],
                                      reassigned: Dict[int, List[Tuple[float, float]]],
                                      data: Dict[str, Any],
                                      offset_distance: float = 2.0):
    """Оригинальная промежуточная визуализация (одно окно)"""
    orig_boundary = [(p["x"], p["y"]) for p in original_sp["boundary_points"]]
    orig_ordered = order_boundary_points(orig_boundary)
    orig_vertices = simplify_boundary(orig_ordered, epsilon=2.2)
    orig_vertices = filter_close_points(orig_vertices)

    line1, line2, polygon = build_two_lines(orig_vertices, offset_distance)

    neighbor_ids = original_sp["neighbors"]
    neighbor_boundaries = {}
    neighbor_centers = {}
    for nid in neighbor_ids:
        sp = next((s for s in data["superpixels"] if s["id"] == nid), None)
        if sp:
            bound = [(p["x"], p["y"]) for p in sp["boundary_points"]]
            bound_ord = order_boundary_points(bound)
            bound_simp = simplify_boundary(bound_ord, epsilon=2.2)
            neighbor_boundaries[nid] = filter_close_points(bound_simp)
            neighbor_centers[nid] = (sp["center"]["x"], sp["center"]["y"])

    center = (original_sp["center"]["x"], original_sp["center"]["y"])

    plt.figure(figsize=(12, 10))

    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        plt.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.5, linewidth=1)

    poly_orig = np.array(orig_vertices + [orig_vertices[0]])
    plt.plot(poly_orig[:, 0], poly_orig[:, 1], 'b-', linewidth=2, alpha=0.7)

    plt.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=2, alpha=0.8, label='Inner line')
    plt.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=2, alpha=0.8, label='Outer line')
    poly_array = np.array(polygon)
    plt.fill(poly_array[:, 0], poly_array[:, 1], alpha=0.3, color='yellow', label='Keep polygon')
    plt.plot(poly_array[:, 0], poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

    plt.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=100,
                marker='*', label=f'Center SP {original_sp["id"]}', zorder=5)
    plt.annotate(str(original_sp["id"]), xy=center, xytext=(8, 8), textcoords='offset points',
                 fontsize=12, fontweight='bold', color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))

    for nid, ncenter in neighbor_centers.items():
        plt.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=60, marker='o', alpha=0.7)
        plt.annotate(str(nid), xy=ncenter, xytext=(5, 5), textcoords='offset points',
                     fontsize=9, color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    unique_ids = [k for k in reassigned.keys() if k != '_lines']
    colors = [cmap(i % cmap.N) for i in range(len(unique_ids))]

    for idx, sp_id in enumerate(unique_ids):
        points = reassigned[sp_id]
        if points:
            pts = np.array(points)
            plt.scatter(pts[:, 0], pts[:, 1], c=[colors[idx]], s=10, alpha=0.8,
                        label=f'SP {sp_id}', edgecolors='none')

    plt.title(f"Промежуточный этап: после обработки SP {original_sp['id']}\n(Жёлтый многоугольник - сохраняемые точки)")
    plt.axis('equal')
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Остальные функции (process_superpixel_with_lines и main) — без изменений
# ------------------------------------------------------------
def process_superpixel_with_lines(superpixel_id: int, data: Dict[str, Any],
                                  processed_ids: set = None,
                                  offset_distance: float = 2.0) -> Dict[int, List[Tuple[float, float]]]:
    sp = next((s for s in data["superpixels"] if s["id"] == superpixel_id), None)
    if sp is None:
        raise ValueError(f"Суперпиксель с id {superpixel_id} не найден.")

    boundary_points = [(p["x"], p["y"]) for p in sp["boundary_points"]]
    ordered_boundary = order_boundary_points(boundary_points)
    simplified_boundary = simplify_boundary(ordered_boundary, epsilon=2.2)
    vertices = filter_close_points(simplified_boundary)

    line1, line2, polygon = build_two_lines(vertices, offset_distance)

    all_points = [(p["x"], p["y"]) for p in sp["all_points"]]
    neighbors = sp["neighbors"]
    sp_dict = {s["id"]: s for s in data["superpixels"]}

    if processed_ids is None:
        processed_ids = set()

    available_neighbors = [nid for nid in neighbors if nid not in processed_ids]
    neighbor_centers = {nid: (sp_dict[nid]["center"]["x"], sp_dict[nid]["center"]["y"])
                        for nid in available_neighbors if nid in sp_dict}

    reassigned = {superpixel_id: []}

    for pt in all_points:
        if point_in_polygon(pt, polygon):
            reassigned.setdefault(superpixel_id, []).append(pt)
        else:
            best_id = None
            best_dist = float('inf')
            for nid, c in neighbor_centers.items():
                dist = np.linalg.norm(np.array(pt) - np.array(c))
                if dist < best_dist:
                    best_dist = dist
                    best_id = nid
            if best_id is not None:
                reassigned.setdefault(best_id, []).append(pt)
            else:
                reassigned.setdefault(superpixel_id, []).append(pt)

    reassigned['_lines'] = (line1, line2, polygon, vertices)
    return reassigned


# ------------------------------------------------------------
# Основная функция (без изменений)
# ------------------------------------------------------------
if __name__ == "__main__":
    with open("superpixels_full_4.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    OFFSET_DISTANCE = 2.0

    try:
        sp_id = int(input("Введите id суперпикселя: "))
    except ValueError:
        print("Некорректный ввод.")
        exit(1)

    original_sp = next((s for s in data["superpixels"] if s["id"] == sp_id), None)
    if original_sp is None:
        print(f"Суперпиксель с id {sp_id} не найден.")
        exit(1)

    neighbors = original_sp["neighbors"]
    if not neighbors:
        print(f"У суперпикселя {sp_id} нет соседей.")
        reassigned = process_superpixel_with_lines(sp_id, data, set(), OFFSET_DISTANCE)
        visualize_intermediate_with_lines(original_sp, reassigned, data, OFFSET_DISTANCE)
    else:
        first_neighbor_id = neighbors[0]
        print(f"Будет обработан суперпиксель {sp_id} и его сосед {first_neighbor_id}")

        print(f"\n--- Этап 1: Обработка суперпикселя {sp_id} ---")
        reassigned_first = process_superpixel_with_lines(sp_id, data, set(), OFFSET_DISTANCE)
        visualize_intermediate_with_lines(original_sp, reassigned_first, data, OFFSET_DISTANCE)

        updated_data = json.loads(json.dumps(data))

        sp_updated = next((s for s in updated_data["superpixels"] if s["id"] == sp_id), None)
        if sp_updated:
            kept_points = reassigned_first.get(sp_id, [])
            sp_updated["all_points"] = [{"x": p[0], "y": p[1]} for p in kept_points]

        for neighbor_id, points in reassigned_first.items():
            if neighbor_id != sp_id and neighbor_id != '_lines':
                neighbor_sp = next((s for s in updated_data["superpixels"] if s["id"] == neighbor_id), None)
                if neighbor_sp:
                    existing_points = [(p["x"], p["y"]) for p in neighbor_sp["all_points"]]
                    new_points = [(p[0], p[1]) for p in points]
                    all_points = existing_points + new_points
                    neighbor_sp["all_points"] = [{"x": p[0], "y": p[1]} for p in all_points]

        print(f"\n--- Этап 2: Обработка суперпикселя {first_neighbor_id} ---")
        processed_ids = {sp_id}
        reassigned_second = process_superpixel_with_lines(first_neighbor_id, updated_data, processed_ids, OFFSET_DISTANCE)

        visualize_result_with_lines(original_sp, reassigned_first, reassigned_second, updated_data, OFFSET_DISTANCE)

        print("\n--- Статистика обработки ---")
        print(f"Суперпиксель {sp_id}:")
        original_points_count = len(original_sp["all_points"])
        kept_points_count = len(reassigned_first.get(sp_id, []))
        print(f"  Исходное количество точек: {original_points_count}")
        print(f"  Осталось в суперпикселе: {kept_points_count}")
        print(f"  Передано соседям: {original_points_count - kept_points_count}")

        print(f"\nСуперпиксель {first_neighbor_id}:")
        first_neighbor_original = next((s for s in data["superpixels"] if s["id"] == first_neighbor_id), None)
        if first_neighbor_original:
            original_neighbor_count = len(first_neighbor_original["all_points"])
            final_neighbor_count = len(reassigned_second.get(first_neighbor_id, []))
            print(f"  Исходное количество точек: {original_neighbor_count}")
            print(f"  Осталось после обработки: {final_neighbor_count}")
            print(f"  Передано дальше: {original_neighbor_count - final_neighbor_count}")