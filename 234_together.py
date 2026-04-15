import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from rdp import rdp
from typing import List, Tuple, Dict, Any
from scipy.interpolate import splprep, splev   # <-- добавлен импорт
from scipy.spatial import ConvexHull

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

def simplify_boundary(points: List[Tuple[float, float]], epsilon: float = 1.5) -> List[Tuple[float, float]]:
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


def project_point_onto_line(point: np.ndarray, line_point: np.ndarray, line_dir: np.ndarray) -> np.ndarray:
    """Проецирует точку на прямую, заданную точкой и направляющим вектором."""
    vec = point - line_point
    t = np.dot(vec, line_dir) / np.dot(line_dir, line_dir)
    return line_point + t * line_dir


def build_two_lines(vertices: List[Tuple[float, float]],
                    offset_distance: float = 2.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """Построение двух линий (прямых или сплайнов) и многоугольника сохранения.
    polygon точно между кривыми. Алгоритм line2 — оригинальный из вашего файла."""
    if DEBUG:
        print("\n" + "=" * 60)
        print("ПОСТРОЕНИЕ ЛИНИЙ (ПРЯМЫЕ ИЛИ СПЛАЙНЫ)")
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

    path_indices, _ = get_cw_path_info(longer_from_idx, longer_to_idx)
    path_vertices = [vertices[i] for i in path_indices]

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

    small_attach1 = np.array(vertices[chosen_small_idx])
    large_attach1 = np.array(vertices[chosen_large_idx])

    d = max(small_len - 1.0, 0.5)

    small_start = np.array(vertices[small_idx])
    small_end = np.array(vertices[(small_idx + 1) % n])
    small_len = np.linalg.norm(small_end - small_start)

    # ----- ИСПРАВЛЕНИЕ: отсчёт от chosen_small_idx к другому концу ребра -----
    if chosen_small_idx == small_idx:
        start_vertex = small_start
        end_vertex = small_end
    else:  # chosen_small_idx == (small_idx + 1) % n
        start_vertex = small_end
        end_vertex = small_start

    small_edge_dir = end_vertex - start_vertex
    small_edge_len = np.linalg.norm(small_edge_dir)

    if small_len > 1.0:
        t_small_attach = (small_len - 1.0) / small_len
        small_attach2 = start_vertex + t_small_attach * small_edge_dir
        small_attach2 = np.round(small_attach2).astype(int)
    else:
        small_mid = (small_start + small_end) / 2.0
        small_attach2 = small_mid
    # -------------------------------------------------------------------------

    large_p = np.array(vertices[large_idx])
    large_vec = np.array(vertices[(large_idx + 1) % n]) - large_p

    dir_vec = large_attach1 - small_attach1
    line_len = np.linalg.norm(dir_vec)
    if line_len < 1e-6:
        dummy = np.array([small_attach1, large_attach1])
        return dummy, dummy, [tuple(small_attach1), tuple(large_attach1), tuple(small_attach2)]

    # Ищем пересечение прямой от small_attach2 с большим ребром (и его продолжением)
    inter_pos, s_pos = line_intersection(small_attach2, dir_vec, large_p, large_vec)
    inter_neg, s_neg = line_intersection(small_attach2, -dir_vec, large_p, large_vec)

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
            inter_large = small_attach2

    large_attach2 = inter_large

    # ---------- ПОСТРОЕНИЕ ПЕРВОЙ КРИВОЙ (line1) ----------
    path_array = np.array(path_vertices)
    use_spline = (len(path_array) >= 3)
    if use_spline:
        k = min(3, len(path_array) - 1)  # 2 для 3 точек, 3 для 4+ точек
        tck, u = splprep(path_array.T, s=0, k=k)
        u_new = np.linspace(0, 1, 200)
        line1_curve = np.column_stack(splev(u_new, tck))
    else:
        line1_curve = np.array([small_attach1, large_attach1])

    # ---------- ОРИГИНАЛЬНЫЙ АЛГОРИТМ ПОСТРОЕНИЯ ВТОРОЙ КРИВОЙ (line2) ----------
    small_edge_dir_norm = small_edge_dir / small_edge_len
    large_edge_dir_norm = large_vec / np.linalg.norm(large_vec)

    vec_small = small_attach2 - small_start
    t_small = np.dot(vec_small, small_edge_dir_norm) / small_edge_len
    t_small = np.clip(t_small, 0.0, 1.0)
    P_start = small_start + t_small * small_edge_dir

    vec_large = large_attach2 - large_p
    large_edge_len = np.linalg.norm(large_vec)
    t_large = np.dot(vec_large, large_edge_dir_norm) / large_edge_len
    t_large = np.clip(t_large, 0.0, 1.0)
    P_end = large_p + t_large * large_vec

    # 2. Строим line2 с помощью гомотетии
    inter_edges, _ = line_intersection(small_start, small_edge_dir, large_p, large_vec)

    if inter_edges is not None:
        def dist_from_center(pt):
            return np.linalg.norm(pt - inter_edges)

        if dist_from_center(small_attach1) > 1e-6 and dist_from_center(large_attach1) > 1e-6:
            scale_start = dist_from_center(P_start) / dist_from_center(small_attach1)
            scale_end = dist_from_center(P_end) / dist_from_center(large_attach1)
        else:
            scale_start = scale_end = 1.0
    else:
        scale_start = scale_end = 1.0

    if use_spline:
        num_samples = 100
        t_samples = np.linspace(0, 1, num_samples)
        points_line1 = np.column_stack(splev(t_samples, tck))

        line2_points = []
        for i, t in enumerate(t_samples):
            p1 = points_line1[i]
            if inter_edges is not None:
                scale = scale_start * (1 - t) + scale_end * t
                p2 = inter_edges + scale * (p1 - inter_edges)
            else:
                delta_start = P_start - small_attach1
                delta_end = P_end - large_attach1
                delta = delta_start * (1 - t) + delta_end * t
                p2 = p1 + delta
            line2_points.append(p2)
        line2_curve = np.array(line2_points)
    else:
        line2_curve = np.array([P_start, P_end])

    # ---------- МНОГОУГОЛЬНИК ТОЧНО МЕЖДУ КРИВЫМИ ----------
    if use_spline:
        keep_curve = np.vstack((line1_curve, line2_curve[::-1]))
        keep_curve = np.vstack((keep_curve, keep_curve[0]))
        polygon = [tuple(p) for p in keep_curve]

        if DEBUG:
            print(f"\nМНОГОУГОЛЬНИК (сплайн, {len(polygon)} точек) — ТОЧНО МЕЖДУ КРИВЫМИ")
    else:
        poly_cand1 = [small_attach1, large_attach1, P_end, P_start]
        poly_cand2 = [small_attach1, P_start, P_end, large_attach1]

        mid_strip = (small_attach1 + large_attach1 + P_start + P_end) / 4.0

        chosen_polygon_list = poly_cand1
        for cand in [poly_cand1, poly_cand2]:
            poly_closed = cand + [cand[0]] if not np.allclose(np.asarray(cand[0]), np.asarray(cand[-1]), atol=1e-6) else cand
            if point_in_polygon(tuple(mid_strip), poly_closed):
                chosen_polygon_list = cand
                break

        polygon = [tuple(p) for p in chosen_polygon_list]
        if not np.allclose(np.asarray(polygon[0]), np.asarray(polygon[-1]), atol=1e-6):
            polygon.append(polygon[0])

    if DEBUG:
        print(f"ВНУТРЕННЯЯ КРИВАЯ (line1) содержит {len(line1_curve)} точек.")
        print(f"ВНЕШНЯЯ КРИВАЯ (line2) содержит {len(line2_curve)} точек.")

    return line1_curve, line2_curve, polygon
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
    orig_vertices = simplify_boundary(orig_ordered)
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
            bound_simp = simplify_boundary(bound_ord)
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
            first_neighbor_simplified = simplify_boundary(first_neighbor_ordered)
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
    orig_vertices = simplify_boundary(orig_ordered)
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
            bound_simp = simplify_boundary(bound_ord)
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
    simplified_boundary = simplify_boundary(ordered_boundary)
    vertices = filter_close_points(simplified_boundary)

    # Получаем кривые (алгоритм построения НЕ ТРОГАЕМ)
    line1_curve, line2_curve, polygon = build_two_lines(vertices, offset_distance)

    # === НОВЫЙ keep_region: только полоса МЕЖДУ двумя кривыми ===
    if len(line1_curve) >= 4 and len(line2_curve) >= 4:
        keep_curve = np.vstack((line1_curve, line2_curve[::-1]))
        keep_curve = np.vstack((keep_curve, keep_curve[0]))
        keep_polygon = [tuple(p) for p in keep_curve]
    else:
        keep_polygon = polygon

    all_points = [(p["x"], p["y"]) for p in sp["all_points"]]
    neighbors = sp["neighbors"]
    sp_dict = {s["id"]: s for s in data["superpixels"]}

    if processed_ids is None:
        processed_ids = set()

    available_neighbors = [nid for nid in neighbors if nid not in processed_ids]

    # --- НОВЫЙ БЛОК: строим kd-деревья для точек каждого соседа ---
    from scipy.spatial import cKDTree
    neighbor_trees = {}
    for nid in available_neighbors:
        if nid not in sp_dict:
            continue
        neighbor_sp = sp_dict[nid]
        pts = [(p["x"], p["y"]) for p in neighbor_sp["all_points"]]
        if pts:
            neighbor_trees[nid] = cKDTree(pts)
        else:
            # если у соседа нет точек, использовать его центр как fallback
            center = (neighbor_sp["center"]["x"], neighbor_sp["center"]["y"])
            neighbor_trees[nid] = cKDTree([center])

    reassigned = {superpixel_id: []}

    for pt in all_points:
        if point_in_polygon(pt, keep_polygon):
            reassigned.setdefault(superpixel_id, []).append(pt)
        else:
            best_id = None
            best_dist = float('inf')
            for nid, tree in neighbor_trees.items():
                dist, _ = tree.query(pt)   # расстояние до ближайшей точки соседа
                if dist < best_dist:
                    best_dist = dist
                    best_id = nid
            if best_id is not None:
                reassigned.setdefault(best_id, []).append(pt)
            else:
                reassigned.setdefault(superpixel_id, []).append(pt)

    reassigned['_lines'] = (line1_curve, line2_curve, polygon, vertices)
    return reassigned

def recompute_boundary_from_points(sp: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Пересчитывает границу суперпикселя по его точкам (выпуклая оболочка + RDP)."""
    points = [(p["x"], p["y"]) for p in sp["all_points"]]
    if len(points) < 3:
        return []  # недостаточно точек для построения границы
    hull = ConvexHull(points)
    hull_points = [points[i] for i in hull.vertices]   # порядок обхода уже задан
    # упрощаем и фильтруем как в оригинале
    simplified = simplify_boundary(hull_points)
    filtered = filter_close_points(simplified)
    return filtered

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
            # Пересчитываем границу для sp_id (хотя он больше не будет обрабатываться, но для полноты)
            new_boundary = recompute_boundary_from_points(sp_updated)
            if new_boundary:
                sp_updated["boundary_points"] = [{"x": p[0], "y": p[1]} for p in new_boundary]

        # Обновляем данные для соседей, получивших точки
        for neighbor_id, points in reassigned_first.items():
            if neighbor_id != sp_id and neighbor_id != '_lines':
                neighbor_sp = next((s for s in updated_data["superpixels"] if s["id"] == neighbor_id), None)
                if neighbor_sp:
                    existing_points = [(p["x"], p["y"]) for p in neighbor_sp["all_points"]]
                    new_points = [(p[0], p[1]) for p in points]
                    all_points = existing_points + new_points
                    neighbor_sp["all_points"] = [{"x": p[0], "y": p[1]} for p in all_points]
                    #пересчитываем границу для соседа
                    new_boundary_neighbor = recompute_boundary_from_points(neighbor_sp)
                    if new_boundary_neighbor:
                        neighbor_sp["boundary_points"] = [{"x": p[0], "y": p[1]} for p in new_boundary_neighbor]

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