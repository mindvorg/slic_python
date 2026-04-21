import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from rdp import rdp
from typing import List, Tuple, Dict, Any
from scipy.interpolate import splprep, splev   # <-- добавлен импорт
from scipy.spatial import ConvexHull, KDTree
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
    """Выбираем пару рёбер, между которыми МАКСИМАЛЬНОЕ расстояние.
    Это даёт самую широкую полосу внутри суперпикселя → минимум пробелов."""
    if len(vertices) < 3:
        return 0, 1 if len(vertices) >= 2 else (0, 0)

    edges = edges_from_vertices(vertices)
    n = len(edges)
    best_pair = None
    best_width = -1.0

    if DEBUG:
        print("\n" + "=" * 60)
        print("ПОИСК ПАРЫ РЁБЕР С МАКСИМАЛЬНОЙ ШИРИНОЙ (чтобы захватить max площади)")
        print("=" * 60)

    for i in range(n):
        for j in range(i + 1, n):
            if {i, (i+1)%n} & {j, (j+1)%n}:   # соседние рёбра пропускаем
                continue

            # === ИСПРАВЛЕНИЕ: явно приводим к float ===
            p1 = np.array(vertices[i], dtype=float)
            d1 = edges[i][0].astype(float)      # вектор первого ребра

            # Нормаль к первому ребру
            normal = np.array([-d1[1], d1[0]], dtype=float)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal /= norm_len

            # Проекции всех вершин
            projections = [np.dot(np.array(p, dtype=float) - p1, normal) for p in vertices]
            width = max(projections) - min(projections)

            if DEBUG:
                print(f"Рёбра {i}↔{j}: ширина полосы = {width:.2f}")

            if width > best_width:
                best_width = width
                best_pair = (i, j)

    if DEBUG and best_pair:
        print(f"\nВЫБРАНЫ РЁБРА С МАКСИМАЛЬНОЙ ШИРИНОЙ: {best_pair[0]} и {best_pair[1]} "
              f"(ширина = {best_width:.2f})")

    return best_pair

def get_principal_direction(vertices: List[Tuple[float, float]]) -> np.ndarray:
    """Вычисляет главную ось суперпикселя (PCA) с помощью numpy.
    Возвращает единичный вектор направления наибольшей дисперсии."""
    if len(vertices) < 3:
        # Если слишком мало точек — берём направление самого длинного ребра
        edges = edges_from_vertices(vertices)
        if edges:
            longest = max(edges, key=lambda e: e[1])
            vec = longest[0]
            return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-8 else np.array([1.0, 0.0])
        return np.array([1.0, 0.0])

    pts = np.array(vertices)
    # Центрируем
    pts_centered = pts - pts.mean(axis=0)
    # SVD (самый стабильный способ PCA без sklearn)
    _, _, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    main_dir = Vt[0]  # первая главная компонента
    return main_dir / np.linalg.norm(main_dir)

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


def build_two_lines(
    vertices: List[Tuple[float, float]],
    offset_distance: float = 2.0,
    cap_style: str = 'round',      # 'round', 'flat', 'sharp'
    roundness: float = 1.0,        # для 'round' – доля полукруга
    curvature: float = 0.5,        # 0 = прямая, 1 = максимальный изгиб
    num_sections: int = 20         # число сечений для построения центральной кривой
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Строит мазок на основе главной оси (PCA) суперпикселя с возможностью изгиба.
    Центральная кривая строится по центрам сечений, перпендикулярных главной оси.
    """

    def intersect_segment_polygon(p1, p2, poly_verts):
        """Возвращает точки пересечения отрезка p1-p2 с полигоном (список)."""
        def segment_intersection(a1, a2, b1, b2):
            r = a2 - a1
            s = b2 - b1
            denom = np.cross(r, s)
            if abs(denom) < 1e-10:
                return None
            t = np.cross(b1 - a1, s) / denom
            u = np.cross(b1 - a1, r) / denom
            if 0 <= t <= 1 and 0 <= u <= 1:
                return a1 + t * r
            return None

        intersections = []
        n = len(poly_verts)
        for i in range(n):
            q1 = np.asarray(poly_verts[i])
            q2 = np.asarray(poly_verts[(i+1)%n])
            inter = segment_intersection(p1, p2, q1, q2)
            if inter is not None:
                intersections.append(inter)
        return intersections

    if len(vertices) < 3:
        # Вырожденный случай
        main_dir = get_principal_direction(vertices)
        center = np.mean(vertices, axis=0)
        perp = np.array([-main_dir[1], main_dir[0]])
        half_width = offset_distance * 1.5
        p1 = center + half_width * perp
        p2 = center - half_width * perp
        line1 = np.array([p1, p2])
        line2 = np.array([p1 + main_dir * 10, p2 + main_dir * 10])
        poly = [tuple(p) for p in [p1, p2, p2 + main_dir * 10, p1 + main_dir * 10, p1]]
        return line1, line2, poly

    pts = np.array(vertices)
    center = np.mean(pts, axis=0)

    # 1. Главная ось (PCA)
    main_dir = get_principal_direction(vertices)
    perp = np.array([-main_dir[1], main_dir[0]])

    # Проекции точек
    proj_main = np.dot(pts - center, main_dir)
    t_min, t_max = np.min(proj_main), np.max(proj_main)

    proj_perp = np.dot(pts - center, perp)
    w_min, w_max = np.min(proj_perp), np.max(proj_perp)
    width = w_max - w_min
    if width < 1.0:
        width = offset_distance * 1.5
    half_width = width / 2.0

    # 2. Построение центральной кривой
    if curvature <= 0.0 or (t_max - t_min) < 1.0:
        center_curve = np.array([center + t_min * main_dir, center + t_max * main_dir])
    else:
        effective_sections = max(5, int(num_sections * curvature))
        t_vals = np.linspace(t_min, t_max, effective_sections)
        center_points = []

        for t in t_vals:
            base_pt = center + t * main_dir
            perp_length = max(width * 2, (t_max - t_min) * 0.5)
            p1 = base_pt - perp_length * perp
            p2 = base_pt + perp_length * perp

            inters = intersect_segment_polygon(p1, p2, vertices)
            if len(inters) >= 2:
                proj = [np.dot(p - base_pt, perp) for p in inters]
                idx_min = np.argmin(proj)
                idx_max = np.argmax(proj)
                center_sec = (inters[idx_min] + inters[idx_max]) / 2.0
                center_points.append(center_sec)
            elif len(inters) == 1:
                center_points.append(inters[0])
            else:
                center_points.append(base_pt)

        center_points = np.array(center_points)

        if len(center_points) >= 3:
            try:
                s_smooth = len(center_points) * (1.0 - curvature) * 2.0 + 0.1
                k = min(3, len(center_points) - 1)
                tck, u = splprep(center_points.T, s=s_smooth, k=k)
                u_new = np.linspace(0, 1, 100)
                center_curve = np.column_stack(splev(u_new, tck))
            except Exception:
                center_curve = center_points
        else:
            center_curve = center_points

    if len(center_curve) < 2:
        center_curve = np.array([center + t_min * main_dir, center + t_max * main_dir])

    # 3. Построение line1 и line2 как эквидистант
    tangents = np.gradient(center_curve, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / (norms + 1e-8)
    normals = np.empty_like(tangents)
    normals[:, 0] = -tangents[:, 1]
    normals[:, 1] = tangents[:, 0]

    line1_curve = center_curve + half_width * normals
    line2_curve = center_curve - half_width * normals

    # 4. Обработка окончаний
    if cap_style == 'flat':
        poly_points = np.vstack([line1_curve, line2_curve[::-1]])
    elif cap_style == 'sharp':
        tip_start = center_curve[0] - half_width * tangents[0]
        tip_end   = center_curve[-1] + half_width * tangents[-1]
        line1_curve = np.vstack([tip_start, line1_curve, tip_end])
        line2_curve = np.vstack([tip_start, line2_curve, tip_end])
        poly_points = np.vstack([tip_start, line1_curve, tip_end, line2_curve[::-1]])
    else:  # 'round'
        arc_radius = half_width * roundness
        num_arc = max(10, int(20 * roundness))

        dir_start = tangents[0]
        dir_end = tangents[-1]
        perp_start = normals[0]
        perp_end = normals[-1]

        theta_start = np.linspace(np.pi/2, 3*np.pi/2, num_arc)
        arc_start = center_curve[0] + arc_radius * (np.cos(theta_start)[:, None] * perp_start +
                                                    np.sin(theta_start)[:, None] * dir_start)
        theta_end = np.linspace(-np.pi/2, np.pi/2, num_arc)
        arc_end = center_curve[-1] + arc_radius * (np.cos(theta_end)[:, None] * perp_end +
                                                   np.sin(theta_end)[:, None] * dir_end)

        if roundness < 1.0:
            trim_len = int(len(line1_curve) * (1.0 - roundness) / 2)
            line1_trimmed = line1_curve[trim_len:-trim_len] if trim_len > 0 else line1_curve
            line2_trimmed = line2_curve[trim_len:-trim_len] if trim_len > 0 else line2_curve
        else:
            line1_trimmed = line1_curve
            line2_trimmed = line2_curve

        line1_curve = np.vstack([arc_start, line1_trimmed, arc_end])
        line2_curve = np.vstack([arc_start[::-1], line2_trimmed, arc_end[::-1]])

        poly_points = np.vstack([arc_start, line2_trimmed, arc_end, line1_trimmed[::-1]])

    polygon = [tuple(p) for p in poly_points]
    polygon.append(polygon[0])

    return line1_curve, line2_curve, polygon

def visualize_final_splines(data: Dict[str, Any], offset_distance: float = 2.0):
    """Итоговый график: все суперпиксели как цветные мазки прямоугольной кисти.
    Цвета берутся из поля color_rgb JSON.
    """
    print("Построение итогового графика со всеми мазками кисти...")

    plt.figure(figsize=(16, 16))
    ax = plt.gca()

    for sp in data["superpixels"]:
        # Получаем цвет суперпикселя из JSON
        rgb = sp.get("color_rgb")
        if rgb is not None and all(k in rgb for k in ("R", "G", "B")):
            # Нормализуем в диапазон [0, 1] для matplotlib
            color = (rgb["R"] / 255.0, rgb["G"] / 255.0, rgb["B"] / 255.0)
        else:
            # Fallback – серый цвет, если поле отсутствует
            color = (0.7, 0.7, 0.7)

        boundary_points = [(p["x"], p["y"]) for p in sp.get("boundary_points", [])]
        if not boundary_points:
            continue

        ordered = order_boundary_points(boundary_points)
        vertices = simplify_boundary(ordered)
        vertices = filter_close_points(vertices)

        if len(vertices) < 3:
            continue

        line1, line2, polygon = build_two_lines(vertices, offset_distance)
        # # Плоские концы
        # line1, line2, poly = build_two_lines(vertices, offset_distance, cap_style='flat')
        #
        # # Острые концы
        # line1, line2, poly = build_two_lines(vertices, offset_distance, cap_style='sharp')
        #
        # # Частичное закругление (сглаженные углы)
        # line1, line2, poly = build_two_lines(vertices, offset_distance, cap_style='round', roundness=0.4)

        # 1. Заполнение мазка (полупрозрачный цвет суперпикселя)
        if len(line1) >= 2 and len(line2) >= 2:
            keep_curve = np.vstack((line1, line2[::-1]))
            keep_curve = np.vstack((keep_curve, keep_curve[0]))
            ax.fill(keep_curve[:, 0], keep_curve[:, 1],
                    color=color, alpha=1, linewidth=0)

        # 2. Тонкая граница исходного суперпикселя
        # poly = np.array(vertices + [vertices[0]])
        # ax.plot(poly[:, 0], poly[:, 1],
        #         color=color, linewidth=1.0, alpha=0.7, linestyle='-')
        #
        # # 3. Inner и Outer кривые (более яркие, но того же цвета)
        # if len(line1) > 1:
        #     ax.plot(line1[:, 0], line1[:, 1],
        #             color=color, linewidth=2.0, alpha=0.9, linestyle='-')
        # if len(line2) > 1:
        #     ax.plot(line2[:, 0], line2[:, 1],
        #             color=color, linewidth=2.0, alpha=0.9, linestyle='--')

    ax.set_title("Итоговая имитация мазков прямоугольной кисти\n(цвета соответствуют исходным суперпикселям)",
                 fontsize=16, pad=20)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='none', alpha=0.4, label='Мазок кисти (заливка)'),
        plt.Line2D([], [], color='gray', linewidth=2.0, linestyle='-', alpha=0.9, label='Inner линия'),
        plt.Line2D([], [], color='gray', linewidth=2.0, linestyle='--', alpha=0.9, label='Outer линия'),
        plt.Line2D([], [], color='gray', linewidth=1.0, alpha=0.7, label='Граница суперпикселя'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plt.show()

    # Сохранение в высоком разрешении
    plt.savefig("final_brush_strokes.png", dpi=400, bbox_inches='tight')
    print("✅ График сохранён как final_brush_strokes.png")


# Остальные функции (process_superpixel_with_lines и main) — без изменений
# ------------------------------------------------------------
def process_superpixel_with_lines(superpixel_id: int, data: Dict[str, Any],
                                  processed_ids: set = None,
                                  offset_distance: float = 12.0) -> Dict[int, List[Tuple[float, float]]]:
    sp = next((s for s in data["superpixels"] if s["id"] == superpixel_id), None)
    if sp is None:
        raise ValueError(f"Суперпиксель {superpixel_id} не найден.")

    all_points = [(p["x"], p["y"]) for p in sp["all_points"]]
    vertices = recompute_boundary_from_points(sp)

    if len(vertices) < 3:
        vertices = order_boundary_points(all_points)
        vertices = simplify_boundary(vertices)

    # Строим широкую полосу (уровень 1)
    line1_curve, line2_curve, polygon = build_two_lines(vertices, offset_distance)

    # === ОСНОВНАЯ ЛОГИКА ===
    reassigned = {superpixel_id: []}

    for pt in all_points:
        if point_in_polygon(pt, polygon):           # точки внутри широкой полосы
            reassigned[superpixel_id].append(pt)
        else:
            # точки снаружи — отдаём соседям (уровень 3)
            reassigned.setdefault(superpixel_id, []).append(pt)  # временно оставляем у себя

    reassigned['_lines'] = (line1_curve, line2_curve, polygon, vertices)
    return reassigned

def recompute_boundary_from_points(sp: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Пересчитывает границу суперпикселя по его точкам (выпуклая оболочка + RDP).
    Robust-версия: работает даже если точки коллинеарны или их меньше 3."""
    points = [(p["x"], p["y"]) for p in sp["all_points"]]

    if len(points) < 3:
        # Для вырожденных случаев просто возвращаем отсортированные точки
        # (фильтруем близкие, чтобы не было дубликатов)
        filtered = filter_close_points(points)
        return filtered

    # Основной случай
    try:
        # QJ — именно то, что рекомендует SciPy при ошибке "Initial simplex is flat"
        hull = ConvexHull(points, qhull_options='QJ')
        hull_points = [points[i] for i in hull.vertices]
        simplified = simplify_boundary(hull_points)
        filtered = filter_close_points(simplified)
        return filtered
    except Exception as e:
        # Если даже QJ не помог (крайне редкий случай)
        print(f"⚠️  Warning: ConvexHull не сработал для SP {sp.get('id', '?')}: {e}")
        # Fallback: сортируем по главной оси
        pts_array = np.array(points)
        if abs(pts_array[:, 0].max() - pts_array[:, 0].min()) >= abs(pts_array[:, 1].max() - pts_array[:, 1].min()):
            sorted_pts = sorted(points, key=lambda p: p[0])
        else:
            sorted_pts = sorted(points, key=lambda p: p[1])
        filtered = filter_close_points(sorted_pts)
        return filtered
# ------------------------------------------------------------
# Основная функция (без изменений)
# ------------------------------------------------------------
def fill_gaps_global(updated_data: Dict[str, Any], original_data: Dict[str, Any]):
    """Финальная заливка всех потерянных точек — чтобы не было пробелов."""
    print("🔧 Финальная заливка пробелов...")

    # Собираем все оригинальные точки (как set кортежей)
    original_points = set()
    for sp in original_data["superpixels"]:
        for p in sp["all_points"]:
            original_points.add((p["x"], p["y"]))

    # Собираем все текущие точки после обработки
    current_points = set()
    sp_dict = {sp["id"]: sp for sp in updated_data["superpixels"]}
    for sp in updated_data["superpixels"]:
        for p in sp.get("all_points", []):
            current_points.add((p["x"], p["y"]))

    # Потерянные точки
    missing = list(original_points - current_points)
    if not missing:
        print("   Пробелов не найдено ✓")
        return

    print(f"   Найдено {len(missing)} потерянных точек — распределяем...")

    # KD-дерево по центрам всех финальных суперпикселей
    centers = []
    ids = []
    for sid, sp in sp_dict.items():
        cx, cy = sp["center"]["x"], sp["center"]["y"]
        centers.append([cx, cy])
        ids.append(sid)
    if not centers:
        return
    tree = KDTree(centers)

    # Распределяем каждую потерянную точку ближайшему центру
    for mx, my in missing:
        dist, idx = tree.query([mx, my])
        nearest_id = ids[idx]
        sp = sp_dict[nearest_id]
        sp["all_points"].append({"x": float(mx), "y": float(my)})

    # Пересчитываем границы только тем, кто получил новые точки
    for sp in updated_data["superpixels"]:
        if len(sp["all_points"]) > 2:  # чтобы ConvexHull не падал
            new_b = recompute_boundary_from_points(sp)
            if new_b:
                sp["boundary_points"] = [{"x": float(p[0]), "y": float(p[1])} for p in new_b]
if __name__ == "__main__":
    with open("superpixels_full_picasso.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    OFFSET_DISTANCE = 4.0   # ← можно попробовать 14.0 или 16.0 для ещё более широких мазков

    updated_data = json.loads(json.dumps(data))   # глубокая копия
    processed_ids = set()

    print("=== ЗАПУСК ПОСЛЕДОВАТЕЛЬНОЙ ОБРАБОТКИ ВСЕХ СУПЕРПИКСЕЛЕЙ ===")
    print(f"Всего суперпикселей: {len(data['superpixels'])}")
    print("Порядок обработки = порядок в списке data['superpixels']\n")

    for original_sp_dict in data["superpixels"]:
        sp_id = original_sp_dict["id"]

        if sp_id in processed_ids:
            continue

        print(f"→ Обработка суперпикселя {sp_id} ...")

        reassigned = process_superpixel_with_lines(
            sp_id, updated_data, processed_ids.copy(), OFFSET_DISTANCE
        )

        # === Обновляем текущий суперпиксель ===
        sp_updated = next((s for s in updated_data["superpixels"] if s["id"] == sp_id), None)
        if sp_updated:
            kept_points = reassigned.get(sp_id, [])
            sp_updated["all_points"] = [{"x": float(p[0]), "y": float(p[1])} for p in kept_points]

            new_boundary = recompute_boundary_from_points(sp_updated)
            if new_boundary:
                sp_updated["boundary_points"] = [{"x": float(p[0]), "y": float(p[1])} for p in new_boundary]

        # === УМНОЕ ПЕРЕРАСПРЕДЕЛЕНИЕ ТОЧЕК СОСЕДЯМ (уровень 3) ===
        for neighbor_id, points in reassigned.items():
            if neighbor_id == sp_id or neighbor_id == '_lines':
                continue

            neighbor_sp = next((s for s in updated_data["superpixels"] if s["id"] == neighbor_id), None)
            if not neighbor_sp:
                continue

            existing_points = [(p["x"], p["y"]) for p in neighbor_sp["all_points"]]
            new_points = [(p[0], p[1]) for p in points]

            all_points = existing_points + new_points

            # Убираем дубликаты
            seen = set()
            unique_all = []
            for pt in all_points:
                tpt = tuple(pt)
                if tpt not in seen:
                    seen.add(tpt)
                    unique_all.append(pt)

            neighbor_sp["all_points"] = [{"x": float(p[0]), "y": float(p[1])} for p in unique_all]

            new_boundary_neighbor = recompute_boundary_from_points(neighbor_sp)
            if new_boundary_neighbor:
                neighbor_sp["boundary_points"] = [{"x": float(p[0]), "y": float(p[1])} for p in new_boundary_neighbor]

        processed_ids.add(sp_id)

    # === ФИНАЛЬНАЯ ЗАЛИВКА ПРОБЕЛОВ ===
    print("\n✅ Обработка ВСЕХ суперпикселей завершена!")
    fill_gaps_global(updated_data, data)          # ← глобальная страховка
    visualize_final_splines(updated_data, OFFSET_DISTANCE)

    # Краткая статистика
    print("\n--- Итоговая статистика ---")
    total_orig = sum(len(sp["all_points"]) for sp in data["superpixels"])
    total_final = sum(len(sp["all_points"]) for sp in updated_data["superpixels"])
    print(f"Всего точек исходно : {total_orig}")
    print(f"Всего точек финально: {total_final}")
    print(f"Изменение: {total_final - total_orig:+} точек")
