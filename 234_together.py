import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from rdp import rdp
from typing import List, Tuple, Dict, Any
DEBUG = True  # Включить вывод отладочной информации


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def order_boundary_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Упорядочивает неупорядоченный набор точек границы в замкнутый контур.
    Используется сортировка по углу относительно центра масс.
    """
    if not points:
        return []
    # Центр масс
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    # Сортировка по углу
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return ordered


def simplify_boundary(points: List[Tuple[float, float]], epsilon: float = 2.2) -> List[Tuple[float, float]]:
    """
    Упрощает границу с помощью алгоритма Рамера-Дугласа-Пекера.
    Возвращает упрощённый список точек (незамкнутый).
    """
    if len(points) < 3:
        return points
    # RDP ожидает массив точек
    pts = np.array(points)
    simplified = rdp(pts, epsilon=epsilon, return_mask=False)
    return [tuple(p) for p in simplified]


def filter_close_points(points: List[Tuple[float, float]], min_dist: float = 0.5) -> List[Tuple[float, float]]:
    """
    Удаляет точки, которые слишком близки к предыдущим.
    """
    if not points:
        return []
    filtered = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(np.array(p) - np.array(filtered[-1])) > min_dist:
            filtered.append(p)
    return filtered


def edges_from_vertices(vertices: List[Tuple[float, float]]) -> List[Tuple[np.ndarray, float, int]]:
    """
    Возвращает список рёбер многоугольника: (вектор, длина, индекс начала)
    """
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
    """
    Возвращает индексы двух наиболее параллельных рёбер (индексы начала).
    Рёбра НЕ должны быть соседними (не иметь общих вершин).
    """
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


def find_path_between_edges(vertices: List[Tuple[float, float]],
                            edge1: Tuple[int, int],
                            edge2: Tuple[int, int]) -> List[int]:
    """
    Находит путь между двумя рёбрами (список индексов вершин).
    Возвращает путь, проходящий ПО ВСЕМ промежуточным вершинам.
    """
    a1, a2 = edge1
    b1, b2 = edge2
    n = len(vertices)

    exclude_vertices = {a1, a2, b1, b2}

    if DEBUG:
        print(f"\nПОИСК ПУТИ: от {edge1} до {edge2}")
        print(f"  Исключаемые вершины: {exclude_vertices}")

    best_path = None
    best_length = float('inf')

    for start in [a1, a2]:
        for end in [b1, b2]:
            if start == end:
                continue

            # Путь по часовой стрелке
            path_cw = []
            i = start
            while True:
                if i == start or i == end or i not in exclude_vertices:
                    path_cw.append(i)
                if i == end:
                    break
                i = (i + 1) % n

            # Путь против часовой стрелки
            path_ccw = []
            i = start
            while True:
                if i == start or i == end or i not in exclude_vertices:
                    path_ccw.append(i)
                if i == end:
                    break
                i = (i - 1) % n

            if DEBUG:
                print(f"  start={start}, end={end}: CW={path_cw}, CCW={path_ccw}")

            for path in [path_cw, path_ccw]:
                if len(path) >= 2:
                    path_length = 0.0
                    for k in range(len(path) - 1):
                        p1 = np.array(vertices[path[k]])
                        p2 = np.array(vertices[path[k + 1]])
                        path_length += np.linalg.norm(p2 - p1)

                    if path_length < best_length:
                        best_length = path_length
                        best_path = path

    if best_path is None:
        best_path = [a1, b1]

    if DEBUG:
        print(f"  ВЫБРАН ПУТЬ: {best_path}")

    return best_path

def get_offset_points_on_edge(edge_start: Tuple[float, float],
                              edge_end: Tuple[float, float],
                              offset_distance: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Вычисляет две точки на ребре, смещённые от концов ребра на заданное расстояние.
    """
    start = np.array(edge_start)
    end = np.array(edge_end)
    edge_vec = end - start
    edge_length = np.linalg.norm(edge_vec)

    if edge_length < offset_distance * 2:
        mid = (start + end) / 2
        if DEBUG:
            print(f"  Ребро {edge_start}->{edge_end} короткое, середина={tuple(mid)}")
        return (tuple(mid), tuple(mid))

    direction = edge_vec / edge_length
    offset_from_start = start + direction * offset_distance
    offset_from_end = end - direction * offset_distance

    if DEBUG:
        print(f"  Ребро {edge_start}->{edge_end}:")
        print(f"    offset_from_start={tuple(offset_from_start)}")
        print(f"    offset_from_end={tuple(offset_from_end)}")

    return (tuple(offset_from_start), tuple(offset_from_end))


def build_polygon_from_lines(vertices: List[Tuple[float, float]],
                             offset_distance: float = 2.0) -> List[Tuple[float, float]]:
    """
    Строит многоугольник, образованный двумя прямыми и участками рёбер.
    """
    n = len(vertices)
    if n < 4:
        return vertices

    e1_idx, e2_idx = find_most_parallel_edges(vertices)

    if e1_idx is None or e2_idx is None:
        return vertices

    a1 = e1_idx
    a2 = (e1_idx + 1) % n
    b1 = e2_idx
    b2 = (e2_idx + 1) % n

    if DEBUG:
        print(f"\nВЫБРАННЫЕ РЁБРА:")
        print(f"  Ребро {e1_idx}: вершины {vertices[a1]} и {vertices[a2]}")
        print(f"  Ребро {e2_idx}: вершины {vertices[b1]} и {vertices[b2]}")

    edge1_vertices = (vertices[a1], vertices[a2])
    edge2_vertices = (vertices[b1], vertices[b2])

    path_indices = find_path_between_edges(vertices, (a1, a2), (b1, b2))
    path_points = [vertices[i] for i in path_indices]

    if DEBUG:
        print(f"\nПУТЬ МЕЖДУ РЁБРАМИ: {path_points}")

    if len(path_points) < 2:
        path_points = [edge1_vertices[0], edge2_vertices[0]]

    # Внутренняя прямая
    line1_start = path_points[0]
    line1_end = path_points[-1]

    # Получаем смещённые точки
    offset_start1, offset_end1 = get_offset_points_on_edge(edge1_vertices[0], edge1_vertices[1], offset_distance)
    offset_start2, offset_end2 = get_offset_points_on_edge(edge2_vertices[0], edge2_vertices[1], offset_distance)

    # Внешняя прямая: берём смещения с противоположных сторон от пути
    dist_a1_to_start = np.linalg.norm(np.array(edge1_vertices[0]) - np.array(line1_start))
    dist_a2_to_start = np.linalg.norm(np.array(edge1_vertices[1]) - np.array(line1_start))
    dist_b1_to_end = np.linalg.norm(np.array(edge2_vertices[0]) - np.array(line1_end))
    dist_b2_to_end = np.linalg.norm(np.array(edge2_vertices[1]) - np.array(line1_end))

    if dist_a1_to_start < dist_a2_to_start:
        line2_start = offset_start1
    else:
        line2_start = offset_end1

    if dist_b1_to_end < dist_b2_to_end:
        line2_end = offset_end2
    else:
        line2_end = offset_start2

    if DEBUG:
        print(f"\nline1: {line1_start} -> {line1_end}")
        print(f"line2: {line2_start} -> {line2_end}")

    # Строим многоугольник (без дубликатов)
    polygon = []

    def add_unique(point):
        if point not in polygon:
            polygon.append(point)

    # 1. Внутренняя прямая
    add_unique(line1_start)

    # 2. Промежуточные точки пути
    for pt in path_points[1:-1]:
        add_unique(pt)

    # 3. Конец внутренней прямой
    add_unique(line1_end)

    # 4. Вершина второго ребра (ближайшая к line1_end)
    dist_to_b1 = np.linalg.norm(np.array(line1_end) - np.array(edge2_vertices[0]))
    dist_to_b2 = np.linalg.norm(np.array(line1_end) - np.array(edge2_vertices[1]))
    if dist_to_b1 < dist_to_b2:
        add_unique(edge2_vertices[0])
    else:
        add_unique(edge2_vertices[1])

    # 5. Внешняя прямая
    add_unique(line2_end)
    add_unique(line2_start)

    # 6. Вершина первого ребра (ближайшая к line2_start)
    dist_to_a1 = np.linalg.norm(np.array(line2_start) - np.array(edge1_vertices[0]))
    dist_to_a2 = np.linalg.norm(np.array(line2_start) - np.array(edge1_vertices[1]))
    if dist_to_a1 < dist_to_a2:
        add_unique(edge1_vertices[0])
    else:
        add_unique(edge1_vertices[1])

    # Замыкаем
    if len(polygon) > 0 and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    if DEBUG:
        print(f"\nМНОГОУГОЛЬНИК ({len(polygon)} точек):")
        for i, pt in enumerate(polygon):
            print(f"  [{i}]: {pt}")

    return polygon


def build_two_lines(vertices: List[Tuple[float, float]],
                    offset_distance: float = 2.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Строит две прямые и возвращает их вместе с многоугольником.
    """
    if DEBUG:
        print("\n" + "=" * 60)
        print("ПОСТРОЕНИЕ ПРЯМЫХ")
        print("=" * 60)

    polygon = build_polygon_from_lines(vertices, offset_distance)

    if len(polygon) < 4:
        line1 = np.array([vertices[0], vertices[1]])
        line2 = np.array([vertices[0], vertices[1]])
        if DEBUG:
            print("\nМногоугольник слишком маленький")
        return line1, line2, polygon

    # Внутренняя прямая
    line1 = np.array([polygon[0], polygon[1]])

    # Внешняя прямая - ищем две уникальные точки, не являющиеся вершинами
    line2 = None
    for i in range(len(polygon) - 1):
        pt = polygon[i]
        is_vertex = any(np.linalg.norm(np.array(pt) - np.array(v)) < 0.1 for v in vertices)
        if not is_vertex:
            if line2 is None:
                line2_start = pt
            else:
                line2_end = pt
                line2 = np.array([line2_start, line2_end])
                break

    if line2 is None:
        line2 = np.array([polygon[2], polygon[3]] if len(polygon) > 3 else [polygon[0], polygon[1]])

    if DEBUG:
        print(f"\nВНУТРЕННЯЯ ПРЯМАЯ (line1): {line1[0]} -> {line1[1]}")
        print(f"ВНЕШНЯЯ ПРЯМАЯ (line2): {line2[0]} -> {line2[1]}")

    return line1, line2, polygon
def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Проверяет, находится ли точка внутри полигона (алгоритм с лучом).
    """
    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n - 1):  # polygon уже замкнут, последняя точка = первой
        x1, y1 = polygon[i]
        x2, y2 = polygon[i + 1]

        # Проверка, пересекает ли горизонтальный луч ребро
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside

    return inside

def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """
    Вычисляет расстояние от точки до отрезка.
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-6:
        return np.linalg.norm(point - line_start)

    # Проекция точки на прямую
    t = np.dot(point - line_start, line_vec) / (line_len * line_len)
    t = max(0, min(1, t))  # Ограничиваем в пределах отрезка

    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def is_point_between_lines(point: Tuple[float, float],
                           line1: np.ndarray,
                           line2: np.ndarray) -> bool:
    """
    Проверяет, находится ли точка между двумя прямыми.
    Точка считается между прямыми, если расстояние до line1 <= расстояние до line2
    (line1 - внутренняя линия, line2 - внешняя)
    """
    pt = np.array(point)

    dist_to_line1 = point_to_line_distance(pt, line1[0], line1[1])
    dist_to_line2 = point_to_line_distance(pt, line2[0], line2[1])

    # Точка между прямыми, если она ближе к line1 или между ними
    return dist_to_line1 <= dist_to_line2


def process_superpixel_with_lines(superpixel_id: int, data: Dict[str, Any],
                                  processed_ids: set = None,
                                  offset_distance: float = 2.0) -> Dict[int, List[Tuple[float, float]]]:
    """
    Обрабатывает суперпиксель: точки внутри построенного многоугольника остаются,
    остальные перераспределяются соседям.
    """
    sp = next((s for s in data["superpixels"] if s["id"] == superpixel_id), None)
    if sp is None:
        raise ValueError(f"Суперпиксель с id {superpixel_id} не найден.")

    # Получаем границу суперпикселя
    boundary_points = [(p["x"], p["y"]) for p in sp["boundary_points"]]
    ordered_boundary = order_boundary_points(boundary_points)
    simplified_boundary = simplify_boundary(ordered_boundary, epsilon=2.2)
    vertices = filter_close_points(simplified_boundary)

    # Строим многоугольник из двух прямых и участков рёбер
    line1, line2, polygon = build_two_lines(vertices, offset_distance)

    all_points = [(p["x"], p["y"]) for p in sp["all_points"]]
    neighbors = sp["neighbors"]
    sp_dict = {s["id"]: s for s in data["superpixels"]}

    # Фильтруем соседей: исключаем уже обработанные
    if processed_ids is None:
        processed_ids = set()

    available_neighbors = [nid for nid in neighbors if nid not in processed_ids]
    neighbor_centers = {nid: (sp_dict[nid]["center"]["x"], sp_dict[nid]["center"]["y"])
                        for nid in available_neighbors if nid in sp_dict}

    reassigned = {superpixel_id: []}

    for pt in all_points:
        # Проверка, находится ли точка внутри многоугольника
        if point_in_polygon(pt, polygon):
            reassigned.setdefault(superpixel_id, []).append(pt)
        else:
            # Найти ближайшего доступного соседа
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

    # Сохраняем линии и многоугольник для визуализации
    reassigned['_lines'] = (line1, line2, polygon, vertices)

    return reassigned


def visualize_result_with_lines(original_sp: Dict[str, Any],
                                reassigned: Dict[int, List[Tuple[float, float]]],
                                neighbor_reassigned: Dict[int, List[Tuple[float, float]]] = None,
                                data: Dict[str, Any] = None,
                                offset_distance: float = 2.0):
    """
    Отображает три стадии обработки с использованием многоугольника.
    """
    # Исходный суперпиксель
    orig_boundary = [(p["x"], p["y"]) for p in original_sp["boundary_points"]]
    orig_ordered = order_boundary_points(orig_boundary)
    orig_vertices = simplify_boundary(orig_ordered, epsilon=2.2)
    orig_vertices = filter_close_points(orig_vertices)

    # Строим линии и многоугольник для исходного суперпикселя
    line1, line2, polygon = build_two_lines(orig_vertices, offset_distance)

    # Соседи исходного суперпикселя
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

    # Определяем первого соседа
    first_neighbor_id = neighbor_ids[0] if neighbor_ids else None

    # Если есть первый сосед, получаем его данные
    first_neighbor_sp = None
    first_neighbor_vertices = None
    first_neighbor_line1 = None
    first_neighbor_line2 = None
    first_neighbor_polygon = None
    first_neighbor_center = None
    first_neighbor_points = []  # Точки, принадлежащие first_neighbor после обработки

    if first_neighbor_id and data:
        first_neighbor_sp = next((s for s in data["superpixels"] if s["id"] == first_neighbor_id), None)
        if first_neighbor_sp:
            # Получаем границу первого соседа
            first_neighbor_bound = [(p["x"], p["y"]) for p in first_neighbor_sp["boundary_points"]]
            first_neighbor_ordered = order_boundary_points(first_neighbor_bound)
            first_neighbor_simplified = simplify_boundary(first_neighbor_ordered, epsilon=2.2)
            first_neighbor_vertices = filter_close_points(first_neighbor_simplified)

            # Строим линии и многоугольник для первого соседа
            first_neighbor_line1, first_neighbor_line2, first_neighbor_polygon = build_two_lines(
                first_neighbor_vertices, offset_distance)
            first_neighbor_center = (first_neighbor_sp["center"]["x"], first_neighbor_sp["center"]["y"])

    # Объединяем результаты перераспределения
    all_reassigned = {}
    if reassigned:
        for k, v in reassigned.items():
            if k != '_lines':
                all_reassigned.setdefault(k, []).extend(v)
    if neighbor_reassigned:
        for k, v in neighbor_reassigned.items():
            if k != '_lines':
                all_reassigned.setdefault(k, []).extend(v)
                # Собираем точки, принадлежащие first_neighbor
                if k == first_neighbor_id:
                    first_neighbor_points.extend(v)

    # Также добавляем точки, которые остались в first_neighbor из первого этапа
    if neighbor_reassigned and first_neighbor_id in neighbor_reassigned:
        first_neighbor_points.extend(neighbor_reassigned.get(first_neighbor_id, []))

    # Создаём фигуру с тремя подграфиками
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ------------------------------------------------------------
    # График 1: Исходное состояние с многоугольником
    # ------------------------------------------------------------
    ax1 = axes[0]

    # Исходный суперпиксель
    poly_orig = np.array(orig_vertices + [orig_vertices[0]])
    ax1.plot(poly_orig[:, 0], poly_orig[:, 1], 'b-', linewidth=2, label=f'SP {original_sp["id"]}')

    # Прямые
    ax1.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=2, alpha=0.8, label='Inner line')
    ax1.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=2, alpha=0.8, label='Outer line')

    # Многоугольник (заполненная область)
    poly_array = np.array(polygon)
    ax1.fill(poly_array[:, 0], poly_array[:, 1], alpha=0.3, color='yellow', label='Keep polygon')
    ax1.plot(poly_array[:, 0], poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

    # Соседи
    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        ax1.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.5,
                 label=f'Neighbor {nid}' if nid == neighbor_ids[0] else "")

    # Точки исходного суперпикселя
    all_pts = [(p["x"], p["y"]) for p in original_sp["all_points"]]
    all_pts = np.array(all_pts)
    ax1.scatter(all_pts[:, 0], all_pts[:, 1], c='blue', s=5, alpha=0.5)

    # Центр
    ax1.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=80,
                marker='*', label='Center', zorder=5)
    ax1.annotate(str(original_sp["id"]),
                 xy=(center[0], center[1]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold',
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))

    # Отображаем номера соседей
    for nid, ncenter in neighbor_centers.items():
        ax1.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=50, marker='o', alpha=0.7)
        ax1.annotate(str(nid),
                     xy=(ncenter[0], ncenter[1]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

    ax1.set_title("Этап 1: Исходное состояние\n(Жёлтый многоугольник - сохраняемые точки)")
    ax1.axis('equal')
    ax1.legend(loc='upper right', fontsize=8)

    # ------------------------------------------------------------
    # График 2: После обработки выбранного суперпикселя
    # ------------------------------------------------------------
    ax2 = axes[1]

    # Границы соседей
    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        ax2.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.5, linewidth=1)

    # Граница исходного суперпикселя
    ax2.plot(poly_orig[:, 0], poly_orig[:, 1], 'b--', linewidth=1.5, alpha=0.5,
             label=f'Original boundary SP {original_sp["id"]}')

    # Прямые и многоугольник
    ax2.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=2, alpha=0.8, label='Inner line')
    ax2.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=2, alpha=0.8, label='Outer line')
    poly_array = np.array(polygon)
    ax2.fill(poly_array[:, 0], poly_array[:, 1], alpha=0.3, color='yellow')
    ax2.plot(poly_array[:, 0], poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

    # Центр
    ax2.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=80,
                marker='*', label='Center', zorder=5)
    ax2.annotate(str(original_sp["id"]),
                 xy=(center[0], center[1]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold',
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))

    # Отображаем номера соседей
    for nid, ncenter in neighbor_centers.items():
        ax2.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=50, marker='o', alpha=0.7)
        ax2.annotate(str(nid),
                     xy=(ncenter[0], ncenter[1]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

    # Цветовая палитра
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')

    unique_ids = [k for k in reassigned.keys() if k != '_lines']
    num_colors = len(unique_ids)
    if num_colors > 10:
        try:
            cmap = plt.colormaps['tab20']
        except AttributeError:
            cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(num_colors)]

    # Отображаем точки
    for idx, sp_id in enumerate(unique_ids):
        points = reassigned[sp_id]
        if points:
            pts = np.array(points)
            ax2.scatter(pts[:, 0], pts[:, 1],
                        c=[colors[idx]],
                        s=8,
                        alpha=0.8,
                        label=f'SP {sp_id}',
                        edgecolors='none')

    ax2.set_title(f"Этап 2: После обработки SP {original_sp['id']}")
    ax2.axis('equal')
    ax2.legend(loc='upper right', fontsize=8, markerscale=1.5)

    # ------------------------------------------------------------
    # График 3: После обработки первого соседа
    # ------------------------------------------------------------
    ax3 = axes[2]

    # Собираем все центры
    all_centers = {}
    for s in data["superpixels"]:
        all_centers[s["id"]] = (s["center"]["x"], s["center"]["y"])

    # Отображаем границы других соседей (кроме первого)
    for nid, verts in neighbor_boundaries.items():
        if nid != first_neighbor_id:
            poly = np.array(verts + [verts[0]])
            ax3.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.3, linewidth=1)

    # Граница исходного суперпикселя
    ax3.plot(poly_orig[:, 0], poly_orig[:, 1], 'b--', linewidth=1.5, alpha=0.5,
             label=f'Boundary SP {original_sp["id"]}')

    # Прямые и многоугольник исходного суперпикселя
    ax3.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=1.5, alpha=0.5)
    ax3.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=1.5, alpha=0.5)

    # Граница первого соседа
    if first_neighbor_vertices is not None:
        poly_neighbor = np.array(first_neighbor_vertices + [first_neighbor_vertices[0]])
        ax3.plot(poly_neighbor[:, 0], poly_neighbor[:, 1], 'orange', linewidth=2, alpha=0.8,
                 label=f'Boundary SP {first_neighbor_id}')

        # Прямые и многоугольник первого соседа
        if first_neighbor_line1 is not None and first_neighbor_line2 is not None:
            ax3.plot(first_neighbor_line1[:, 0], first_neighbor_line1[:, 1], 'orange',
                     linewidth=2, alpha=0.8, linestyle='-', label=f'Inner line SP {first_neighbor_id}')
            ax3.plot(first_neighbor_line2[:, 0], first_neighbor_line2[:, 1], 'orange',
                     linewidth=2, alpha=0.8, linestyle='--', label=f'Outer line SP {first_neighbor_id}')

        # Заполняем многоугольник первого соседа
        if first_neighbor_polygon is not None:
            neighbor_poly_array = np.array(first_neighbor_polygon)
            ax3.fill(neighbor_poly_array[:, 0], neighbor_poly_array[:, 1], alpha=0.2, color='orange')
            ax3.plot(neighbor_poly_array[:, 0], neighbor_poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

        # Центр соседа
        if first_neighbor_center is not None:
            ax3.scatter(first_neighbor_center[0], first_neighbor_center[1], c='orange', edgecolors='black',
                        linewidth=1.5, s=100, marker='*', label=f'Center SP {first_neighbor_id}', zorder=5)
            ax3.annotate(str(first_neighbor_id),
                         xy=(first_neighbor_center[0], first_neighbor_center[1]),
                         xytext=(8, 8),
                         textcoords='offset points',
                         fontsize=11,
                         fontweight='bold',
                         color='orange',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange", alpha=0.8))

    # Центр исходного суперпикселя
    ax3.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1.5, s=100,
                marker='*', label=f'Center SP {original_sp["id"]}', zorder=5)
    ax3.annotate(str(original_sp["id"]),
                 xy=(center[0], center[1]),
                 xytext=(8, 8),
                 textcoords='offset points',
                 fontsize=11,
                 fontweight='bold',
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))

    # Отображаем точки first_neighbor и соединяем их линией
    if first_neighbor_points:
        pts = np.array(first_neighbor_points)
        # Сортируем точки по углу относительно центра first_neighbor
        if first_neighbor_center:
            center_np = np.array(first_neighbor_center)
            angles = np.arctan2(pts[:, 1] - center_np[1], pts[:, 0] - center_np[0])
            sorted_indices = np.argsort(angles)
            sorted_pts = pts[sorted_indices]
            # Замыкаем контур
            sorted_pts = np.vstack([sorted_pts, sorted_pts[0]])
            # Рисуем линию, соединяющую точки
            ax3.plot(sorted_pts[:, 0], sorted_pts[:, 1], 'orange', linewidth=2, alpha=0.8,
                     label=f'Points connection SP {first_neighbor_id}')

        # Отображаем сами точки
        ax3.scatter(pts[:, 0], pts[:, 1], c='orange', s=15, alpha=0.8,
                    edgecolors='black', linewidth=0.5, label=f'Points SP {first_neighbor_id}')

    # Отображаем номера других суперпикселей, которые получили точки
    other_ids = set(all_reassigned.keys()) - {original_sp["id"], first_neighbor_id}
    for oid in other_ids:
        if oid in all_centers:
            oc = all_centers[oid]
            ax3.scatter(oc[0], oc[1], c='gray', edgecolors='black', linewidth=0.5, s=40, marker='o', alpha=0.5)
            ax3.annotate(str(oid),
                         xy=(oc[0], oc[1]),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=7,
                         color='gray',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.5))

    # Отображаем остальные точки (не first_neighbor)
    unique_ids_final = list(all_reassigned.keys())
    colors_final = [cmap(i % cmap.N) for i in range(len(unique_ids_final))]

    for idx, sp_id in enumerate(unique_ids_final):
        if sp_id == first_neighbor_id:
            continue  # Пропускаем, так как уже отобразили
        points = all_reassigned[sp_id]
        if points:
            pts = np.array(points)
            ax3.scatter(pts[:, 0], pts[:, 1],
                        c=[colors_final[idx]],
                        s=8,
                        alpha=0.8,
                        label=f'SP {sp_id}',
                        edgecolors='none')

    ax3.set_title(f"Этап 3: После обработки SP {first_neighbor_id}\n(Оранжевая линия - точки SP {first_neighbor_id})")
    ax3.axis('equal')
    ax3.legend(loc='upper right', fontsize=8, markerscale=1.5)

    plt.tight_layout()
    plt.show()
def visualize_intermediate_with_lines(original_sp: Dict[str, Any],
                                      reassigned: Dict[int, List[Tuple[float, float]]],
                                      data: Dict[str, Any],
                                      offset_distance: float = 2.0):
    """
    Визуализация промежуточного этапа с многоугольником.
    """
    orig_boundary = [(p["x"], p["y"]) for p in original_sp["boundary_points"]]
    orig_ordered = order_boundary_points(orig_boundary)
    orig_vertices = simplify_boundary(orig_ordered, epsilon=2.2)
    orig_vertices = filter_close_points(orig_vertices)

    # Строим линии и многоугольник
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

    # Границы соседей
    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        plt.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.5, linewidth=1)

    # Граница исходного
    poly_orig = np.array(orig_vertices + [orig_vertices[0]])
    plt.plot(poly_orig[:, 0], poly_orig[:, 1], 'b-', linewidth=2, alpha=0.7)

    # Прямые и многоугольник
    plt.plot(line1[:, 0], line1[:, 1], 'r-', linewidth=2, alpha=0.8, label='Inner line')
    plt.plot(line2[:, 0], line2[:, 1], 'g-', linewidth=2, alpha=0.8, label='Outer line')
    poly_array = np.array(polygon)
    plt.fill(poly_array[:, 0], poly_array[:, 1], alpha=0.3, color='yellow', label='Keep polygon')
    plt.plot(poly_array[:, 0], poly_array[:, 1], 'orange', linewidth=1, alpha=0.5)

    # Центр
    plt.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=100,
                marker='*', label=f'Center SP {original_sp["id"]}', zorder=5)
    plt.annotate(str(original_sp["id"]),
                 xy=(center[0], center[1]),
                 xytext=(8, 8),
                 textcoords='offset points',
                 fontsize=12,
                 fontweight='bold',
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8))

    # Номера соседей
    for nid, ncenter in neighbor_centers.items():
        plt.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=60, marker='o', alpha=0.7)
        plt.annotate(str(nid),
                     xy=(ncenter[0], ncenter[1]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=9,
                     color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

    # Цветовая палитра
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')

    unique_ids = [k for k in reassigned.keys() if k != '_lines']
    colors = [cmap(i % cmap.N) for i in range(len(unique_ids))]

    # Отображаем точки
    for idx, sp_id in enumerate(unique_ids):
        points = reassigned[sp_id]
        if points:
            pts = np.array(points)
            plt.scatter(pts[:, 0], pts[:, 1],
                        c=[colors[idx]],
                        s=10,
                        alpha=0.8,
                        label=f'SP {sp_id}',
                        edgecolors='none')

    plt.title(f"Промежуточный этап: после обработки SP {original_sp['id']}\n(Жёлтый многоугольник - сохраняемые точки)")
    plt.axis('equal')
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------
# Основная функция
# ------------------------------------------------------------

if __name__ == "__main__":
    # Загружаем данные из файла
    with open("superpixels_full_4.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Параметры
    OFFSET_DISTANCE = 2.0  # Расстояние между прямыми

    # Пользователь вводит id суперпикселя
    try:
        sp_id = int(input("Введите id суперпикселя: "))
    except ValueError:
        print("Некорректный ввод.")
        exit(1)

    # Ищем исходный суперпиксель
    original_sp = next((s for s in data["superpixels"] if s["id"] == sp_id), None)
    if original_sp is None:
        print(f"Суперпиксель с id {sp_id} не найден.")
        exit(1)

    # Получаем первого соседа
    neighbors = original_sp["neighbors"]
    if not neighbors:
        print(f"У суперпикселя {sp_id} нет соседей. Обработка только одного суперпикселя.")
        reassigned = process_superpixel_with_lines(sp_id, data, set(), OFFSET_DISTANCE)
        visualize_intermediate_with_lines(original_sp, reassigned, data, OFFSET_DISTANCE)
    else:
        first_neighbor_id = neighbors[0]
        print(f"Будет обработан суперпиксель {sp_id} и его сосед {first_neighbor_id}")

        # Этап 1: Обработка выбранного суперпикселя
        print(f"\n--- Этап 1: Обработка суперпикселя {sp_id} ---")
        reassigned_first = process_superpixel_with_lines(sp_id, data, set(), OFFSET_DISTANCE)

        # Визуализация первого этапа
        visualize_intermediate_with_lines(original_sp, reassigned_first, data, OFFSET_DISTANCE)

        # Формируем обновлённые данные для второго этапа
        updated_data = json.loads(json.dumps(data))

        # Обновляем точки для обработанного суперпикселя
        sp_updated = next((s for s in updated_data["superpixels"] if s["id"] == sp_id), None)
        if sp_updated:
            kept_points = reassigned_first.get(sp_id, [])
            sp_updated["all_points"] = [{"x": p[0], "y": p[1]} for p in kept_points]

        # Обновляем точки для соседей
        for neighbor_id, points in reassigned_first.items():
            if neighbor_id != sp_id and neighbor_id != '_lines':
                neighbor_sp = next((s for s in updated_data["superpixels"] if s["id"] == neighbor_id), None)
                if neighbor_sp:
                    existing_points = [(p["x"], p["y"]) for p in neighbor_sp["all_points"]]
                    new_points = [(p[0], p[1]) for p in points]
                    all_points = existing_points + new_points
                    neighbor_sp["all_points"] = [{"x": p[0], "y": p[1]} for p in all_points]

        # Этап 2: Обработка первого соседа
        print(f"\n--- Этап 2: Обработка суперпикселя {first_neighbor_id} ---")
        print(f"(Суперпиксель {sp_id} уже обработан, его точки не будут перераспределяться)")

        processed_ids = {sp_id}
        reassigned_second = process_superpixel_with_lines(first_neighbor_id, updated_data, processed_ids,
                                                          OFFSET_DISTANCE)

        # Визуализация финального результата
        visualize_result_with_lines(original_sp, reassigned_first, reassigned_second, updated_data, OFFSET_DISTANCE)

        # Вывод статистики
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