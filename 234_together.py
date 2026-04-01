import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from rdp import rdp
from typing import List, Tuple, Dict, Any


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
    Критерий: максимизация (длина1 + длина2) * косинус.
    """
    edges = edges_from_vertices(vertices)
    n = len(edges)
    best_pair = None
    best_score = -1.0
    for i in range(n):
        for j in range(i + 1, n):
            v1, l1, _ = edges[i]
            v2, l2, _ = edges[j]
            if l1 * l2 == 0:
                continue
            cos = abs(np.dot(v1, v2)) / (l1 * l2)
            score = (l1 + l2) * cos
            if score > best_score:
                best_score = score
                best_pair = (i, j)
    return best_pair


def two_directions_paths(start: int, end: int, n: int) -> Tuple[List[int], List[int]]:
    """Возвращает два пути по границе от start до end (включительно)"""
    # По часовой стрелке
    path_cw = []
    i = start
    while True:
        path_cw.append(i)
        if i == end:
            break
        i = (i + 1) % n
    # Против часовой
    path_ccw = []
    i = start
    while True:
        path_ccw.append(i)
        if i == end:
            break
        i = (i - 1) % n
    return path_cw, path_ccw


def path_length(path_indices: List[int], vertices: List[Tuple[float, float]]) -> float:
    length = 0.0
    for k in range(len(path_indices) - 1):
        p1 = np.array(vertices[path_indices[k]])
        p2 = np.array(vertices[path_indices[k + 1]])
        length += np.linalg.norm(p2 - p1)
    return length


def bezier_curve(points: List[Tuple[float, float]], num_points: int = 100) -> np.ndarray:
    """
    Строит кривую Безье по контрольным точкам (первая и последняя – концевые).
    """
    n = len(points) - 1
    if n == 0:
        return np.array([points[0]] * num_points)
    if n == 1:
        t = np.linspace(0, 1, num_points)[:, np.newaxis]
        return (1 - t) * np.array(points[0]) + t * np.array(points[1])
    coeffs = [math.comb(n, i) for i in range(n + 1)]
    t = np.linspace(0, 1, num_points)
    basis = np.array([(1 - t) ** (n - i) * t ** i for i in range(n + 1)]).T
    weighted = basis * coeffs
    curve = np.dot(weighted, np.array(points))
    return curve


def build_curve_inside(vertices: List[Tuple[float, float]], num_points: int = 100) -> Tuple[
    np.ndarray, List[Tuple[float, float]]]:
    """
    Построение первой кривой Безье внутри суперпикселя по алгоритму из файла.
    Возвращает (кривая, список точек пути, использованных для построения).
    """
    n = len(vertices)
    if n < 3:
        return np.array(vertices), vertices
    e1, e2 = find_most_parallel_edges(vertices)
    a1, a2 = e1, (e1 + 1) % n
    b1, b2 = e2, (e2 + 1) % n
    best_info = None
    for p in (a1, a2):
        for q in (b1, b2):
            if p == q:
                continue
            path_cw, path_ccw = two_directions_paths(p, q, n)
            for path in (path_cw, path_ccw):
                if len(path) > 2:
                    length = path_length(path, vertices)
                    if best_info is None or length < best_info[0]:
                        best_info = (length, path)
    if best_info is None:
        # запасной вариант – просто отрезок между первыми двумя вершинами
        return np.array([vertices[0], vertices[1]]), [vertices[0], vertices[1]]
    _, best_path = best_info
    path_points = [vertices[i] for i in best_path]
    curve = bezier_curve(path_points, num_points)
    return curve, path_points


def compute_inward_normal(p: Tuple[float, float], prev: Tuple[float, float],
                          nxt: Tuple[float, float], center: Tuple[float, float]) -> np.ndarray:
    """
    Вычисляет единичную нормаль, направленную внутрь суперпикселя, для точки p.
    Используется средний вектор двух соседних отрезков (prev->p и p->nxt),
    затем нормаль, ориентированная в сторону центра.
    """
    v1 = np.array(p) - np.array(prev)
    v2 = np.array(nxt) - np.array(p)
    # усреднённый вектор направления
    avg_dir = (v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2)) / 2
    if np.linalg.norm(avg_dir) < 1e-6:
        avg_dir = np.array([1.0, 0.0])
    # нормаль (поворот на 90 градусов)
    norm = np.array([-avg_dir[1], avg_dir[0]])
    norm = norm / np.linalg.norm(norm)
    # определяем знак: вектор от p к центру должен быть сонаправлен с нормалью
    to_center = np.array(center) - np.array(p)
    if np.dot(norm, to_center) < 0:
        norm = -norm
    return norm


def offset_path(path_points: List[Tuple[float, float]], offset_dist: float,
                center: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Смещает каждую точку пути внутрь суперпикселя на расстояние offset_dist.
    """
    if len(path_points) < 2:
        return path_points
    new_points = []
    for i, p in enumerate(path_points):
        prev = path_points[i - 1] if i > 0 else path_points[0]  # для начала используем следующую
        nxt = path_points[i + 1] if i < len(path_points) - 1 else path_points[-1]
        # для конца и начала можно взять только один отрезок
        if i == 0:
            v = np.array(nxt) - np.array(p)
            norm = np.array([-v[1], v[0]]) / np.linalg.norm(v)
            # проверить направление внутрь
            to_center = np.array(center) - np.array(p)
            if np.dot(norm, to_center) < 0:
                norm = -norm
        elif i == len(path_points) - 1:
            v = np.array(p) - np.array(prev)
            norm = np.array([-v[1], v[0]]) / np.linalg.norm(v)
            to_center = np.array(center) - np.array(p)
            if np.dot(norm, to_center) < 0:
                norm = -norm
        else:
            norm = compute_inward_normal(p, prev, nxt, center)
        new_point = np.array(p) + norm * offset_dist
        new_points.append(tuple(new_point))
    return new_points


def point_in_polygon(point: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """
    Проверяет, находится ли точка внутри полигона (алгоритм с лучом).
    """
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # проверка, пересекает ли горизонтальный луч ребро
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def create_polygon_between_curves(curve1: np.ndarray, curve2: np.ndarray) -> List[Tuple[float, float]]:
    """
    Строит замкнутый полигон между двумя кривыми.
    """
    # Идём по curve1 от начала к концу, затем по curve2 в обратном порядке, замыкаем
    poly = list(curve1) + list(curve2[::-1])
    # замыкаем (первая точка = последняя)
    poly.append(poly[0])
    return poly


def reassign_points(points: List[Tuple[float, float]], poly: List[Tuple[float, float]],
                    current_id: int, current_center: Tuple[float, float], neighbors: List[int],
                    all_superpixels: Dict[int, Any]) -> Dict[int, List[Tuple[float, float]]]:
    """
    Перераспределяет точки: если точка не внутри полигона, она присваивается ближайшему соседу.
    Возвращает словарь {id_superpixel: [точки]} для всех затронутых суперпикселей.
    """
    reassigned = {current_id: []}  # здесь будем собирать точки, оставшиеся в текущем
    # Сначала подготовим центры соседей
    neighbor_centers = {}
    for nid in neighbors:
        sp = all_superpixels.get(nid)
        if sp:
            neighbor_centers[nid] = (sp["center"]["x"], sp["center"]["y"])
    # Для каждой точки
    for pt in points:
        if point_in_polygon(pt, poly):
            reassigned.setdefault(current_id, []).append(pt)
        else:
            # найти ближайший центр среди соседей
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
                # если нет соседей, оставляем в текущем
                reassigned.setdefault(current_id, []).append(pt)
    return reassigned


def process_superpixel(superpixel_id: int, data: Dict[str, Any],
                       processed_ids: set = None, radius: float = 2.0) -> Dict[int, List[Tuple[float, float]]]:
    """
    Обрабатывает суперпиксель: точки внутри круга радиуса radius остаются,
    остальные перераспределяются соседям.

    Args:
        superpixel_id: ID обрабатываемого суперпикселя
        data: данные со всеми суперпикселями
        processed_ids: множество уже обработанных ID (чтобы не перераспределять им точки)
        radius: радиус круга

    Returns:
        Словарь {id_superpixel: [точки]} для всех затронутых суперпикселей
    """
    sp = next((s for s in data["superpixels"] if s["id"] == superpixel_id), None)
    if sp is None:
        raise ValueError(f"Суперпиксель с id {superpixel_id} не найден.")

    center = (sp["center"]["x"], sp["center"]["y"])
    all_points = [(p["x"], p["y"]) for p in sp["all_points"]]
    neighbors = sp["neighbors"]
    sp_dict = {s["id"]: s for s in data["superpixels"]}

    # Перераспределение
    reassigned = {superpixel_id: []}

    # Фильтруем соседей: исключаем уже обработанные
    available_neighbors = [nid for nid in neighbors if nid not in processed_ids]
    neighbor_centers = {nid: (sp_dict[nid]["center"]["x"], sp_dict[nid]["center"]["y"])
                        for nid in available_neighbors if nid in sp_dict}

    for pt in all_points:
        # Проверка, находится ли точка внутри круга
        if np.linalg.norm(np.array(pt) - np.array(center)) <= radius:
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
                # Если нет доступных соседей, оставляем в текущем
                reassigned.setdefault(superpixel_id, []).append(pt)

    return reassigned


def visualize_result(original_sp: Dict[str, Any],
                     reassigned: Dict[int, List[Tuple[float, float]]],
                     neighbor_reassigned: Dict[int, List[Tuple[float, float]]] = None,
                     data: Dict[str, Any] = None,
                     radius: float = 2.0):
    """
    Отображает три стадии обработки:
    1. Исходный суперпиксель и его соседи
    2. После обработки выбранного суперпикселя
    3. После обработки первого соседнего суперпикселя
    """
    # Исходный суперпиксель
    orig_boundary = [(p["x"], p["y"]) for p in original_sp["boundary_points"]]
    orig_ordered = order_boundary_points(orig_boundary)
    orig_vertices = simplify_boundary(orig_ordered, epsilon=2.2)
    orig_vertices = filter_close_points(orig_vertices)

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

    # Координаты центра выбранного суперпикселя
    center = (original_sp["center"]["x"], original_sp["center"]["y"])

    # Определяем первый соседний суперпиксель для отображения на третьем графике
    first_neighbor_id = neighbor_ids[0] if neighbor_ids else None
    first_neighbor_sp = None
    first_neighbor_center = neighbor_centers[first_neighbor_id]
    if first_neighbor_id and data:
        first_neighbor_sp = next((s for s in data["superpixels"] if s["id"] == first_neighbor_id), None)
        if first_neighbor_sp:
            first_neighbor_center = (first_neighbor_sp["center"]["x"], first_neighbor_sp["center"]["y"])

    # Объединяем результаты перераспределения (определяем ЗДЕСЬ, до использования)
    all_reassigned = {}
    if reassigned:
        for k, v in reassigned.items():
            all_reassigned.setdefault(k, []).extend(v)
    if neighbor_reassigned:
        for k, v in neighbor_reassigned.items():
            all_reassigned.setdefault(k, []).extend(v)

    # Создаём фигуру с тремя подграфиками
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ------------------------------------------------------------
    # График 1: Исходное состояние
    # ------------------------------------------------------------
    ax1 = axes[0]

    # Исходный суперпиксель
    poly_orig = np.array(orig_vertices + [orig_vertices[0]])
    ax1.plot(poly_orig[:, 0], poly_orig[:, 1], 'b-', linewidth=2, label=f'SP {original_sp["id"]}')

    # Соседи
    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        ax1.plot(poly[:, 0], poly[:, 1], 'g--', alpha=0.7,
                 label=f'Neighbor {nid}' if nid == neighbor_ids[0] else "")

    # Точки исходного суперпикселя
    all_pts = [(p["x"], p["y"]) for p in original_sp["all_points"]]
    all_pts = np.array(all_pts)
    ax1.scatter(all_pts[:, 0], all_pts[:, 1], c='blue', s=5, alpha=0.5)

    # Центр и круг
    ax1.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=80,
                marker='*', label='Center', zorder=5)
    # Добавляем номер суперпикселя над центром
    ax1.annotate(str(original_sp["id"]),
                 xy=(center[0], center[1]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold',
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))

    circle = plt.Circle(center, radius, color='red', fill=False, linewidth=2, linestyle='--', alpha=0.7)
    ax1.add_patch(circle)

    # Отображаем номера соседних суперпикселей
    for nid, ncenter in neighbor_centers.items():
        ax1.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=50, marker='o', alpha=0.7)
        ax1.annotate(str(nid),
                     xy=(ncenter[0], ncenter[1]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

    ax1.set_title("Этап 1: Исходное состояние\n(★ - центр суперпикселя)")
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

    # Граница исходного суперпикселя (пунктиром, т.к. он уже частично изменён)
    ax2.plot(poly_orig[:, 0], poly_orig[:, 1], 'b--', linewidth=1.5, alpha=0.5,
             label=f'Original boundary SP {original_sp["id"]}')

    # Цветовая палитра
    try:
        cmap = plt.colormaps['tab10']
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')

    unique_ids = list(reassigned.keys())
    num_colors = len(unique_ids)
    if num_colors > 10:
        try:
            cmap = plt.colormaps['tab20']
        except AttributeError:
            cmap = plt.cm.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(num_colors)]

    # Круг
    circle2 = plt.Circle(center, radius, color='red', fill=False, linewidth=2, linestyle='--')
    ax2.add_patch(circle2)

    # Центр с номером
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

    # Отображаем номера соседних суперпикселей
    for nid, ncenter in neighbor_centers.items():
        ax2.scatter(ncenter[0], ncenter[1], c='green', edgecolors='black', linewidth=0.5, s=50, marker='o', alpha=0.7)
        ax2.annotate(str(nid),
                     xy=(ncenter[0], ncenter[1]),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     color='green',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="green", alpha=0.6))

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
    # График 3: После обработки первого соседнего суперпикселя
    # ------------------------------------------------------------
    ax3 = axes[2]

    # Собираем все центры для отображения на третьем графике
    all_centers = {}
    for s in data["superpixels"]:
        all_centers[s["id"]] = (s["center"]["x"], s["center"]["y"])

    # Отображаем границы всех суперпикселей, которые участвуют
    # Границы соседей (кроме обработанного)
    for nid, verts in neighbor_boundaries.items():
        if nid != first_neighbor_id:
            poly = np.array(verts + [verts[0]])
            ax3.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.3, linewidth=1)

    # Граница исходного суперпикселя
    ax3.plot(poly_orig[:, 0], poly_orig[:, 1], 'b--', linewidth=1, alpha=0.3)

    # Граница первого соседа (пунктиром)
    if first_neighbor_sp:
        neighbor_bound = [(p["x"], p["y"]) for p in first_neighbor_sp["boundary_points"]]
        neighbor_ordered = order_boundary_points(neighbor_bound)
        neighbor_vertices = simplify_boundary(neighbor_ordered, epsilon=2.2)
        neighbor_vertices = filter_close_points(neighbor_vertices)
        poly_neighbor = np.array(neighbor_vertices + [neighbor_vertices[0]])
        ax3.plot(poly_neighbor[:, 0], poly_neighbor[:, 1], 'orange', linestyle='--',
                 linewidth=1.5, alpha=0.7, label=f'Boundary SP {first_neighbor_id}')

        # Центр соседа с номером
        if first_neighbor_center:
            ax3.scatter(first_neighbor_center[0], first_neighbor_center[1], c='orange', edgecolors='black',
                        linewidth=1, s=80, marker='*', label=f'Center SP {first_neighbor_id}', zorder=5)
            ax3.annotate(str(first_neighbor_id),
                         xy=(first_neighbor_center[0], first_neighbor_center[1]),
                         xytext=(5, 5),
                         textcoords='offset points',
                         fontsize=10,
                         fontweight='bold',
                         color='orange',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="orange", alpha=0.7))

        # Круг для соседа
        neighbor_circle = plt.Circle(first_neighbor_center, radius, color='orange',
                                     fill=False, linewidth=2, linestyle='--', alpha=0.7)
        ax3.add_patch(neighbor_circle)

    # Круг для исходного суперпикселя
    circle3 = plt.Circle(center, radius, color='red', fill=False, linewidth=2, linestyle='--', alpha=0.5)
    ax3.add_patch(circle3)
    ax3.scatter(center[0], center[1], c='red', edgecolors='black', linewidth=1, s=80,
                marker='*', label=f'Center SP {original_sp["id"]}', zorder=5)
    ax3.annotate(str(original_sp["id"]),
                 xy=(center[0], center[1]),
                 xytext=(5, 5),
                 textcoords='offset points',
                 fontsize=10,
                 fontweight='bold',
                 color='red',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.7))
    # Круг
    circle4 = plt.Circle(first_neighbor_center, radius, color='red', fill=False, linewidth=2, linestyle='--')
    ax3.add_patch(circle4)
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

    # Отображаем точки
    unique_ids_final = list(all_reassigned.keys())
    colors_final = [cmap(i % cmap.N) for i in range(len(unique_ids_final))]

    for idx, sp_id in enumerate(unique_ids_final):
        points = all_reassigned[sp_id]
        if points:
            pts = np.array(points)
            ax3.scatter(pts[:, 0], pts[:, 1],
                        c=[colors_final[idx]],
                        s=8,
                        alpha=0.8,
                        label=f'SP {sp_id}',
                        edgecolors='none')

    ax3.set_title(f"Этап 3: После обработки SP {first_neighbor_id}\n(финальное распределение)")
    ax3.axis('equal')
    ax3.legend(loc='upper right', fontsize=8, markerscale=1.5)

    plt.tight_layout()
    plt.show()

def visualize_intermediate(original_sp: Dict[str, Any],
                           reassigned: Dict[int, List[Tuple[float, float]]],
                           data: Dict[str, Any],
                           radius: float = 2.0,
                           stage: str = "first"):
    """
    Визуализация промежуточного этапа (после обработки одного суперпикселя)
    """
    # Исходный суперпиксель
    orig_boundary = [(p["x"], p["y"]) for p in original_sp["boundary_points"]]
    orig_ordered = order_boundary_points(orig_boundary)
    orig_vertices = simplify_boundary(orig_ordered, epsilon=2.2)
    orig_vertices = filter_close_points(orig_vertices)

    # Соседи
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

    plt.figure(figsize=(10, 8))

    # Границы соседей
    for nid, verts in neighbor_boundaries.items():
        poly = np.array(verts + [verts[0]])
        plt.plot(poly[:, 0], poly[:, 1], 'gray', linestyle='--', alpha=0.5, linewidth=1)

    # Граница исходного суперпикселя
    poly_orig = np.array(orig_vertices + [orig_vertices[0]])
    plt.plot(poly_orig[:, 0], poly_orig[:, 1], 'b--', linewidth=1.5, alpha=0.5)

    # Круг
    circle = plt.Circle(center, radius, color='red', fill=False, linewidth=2, linestyle='--')
    plt.gca().add_patch(circle)

    # Центр с номером
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

    # Отображаем номера соседних суперпикселей
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

    unique_ids = list(reassigned.keys())
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

    plt.title(f"Промежуточный этап: после обработки SP {original_sp['id']}")
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
        reassigned = process_superpixel(sp_id, data, set())
        visualize_intermediate(original_sp, reassigned, data)
    else:
        first_neighbor_id = neighbors[0]
        print(f"Будет обработан суперпиксель {sp_id} и его сосед {first_neighbor_id}")

        # Этап 1: Обработка выбранного суперпикселя
        print(f"\n--- Этап 1: Обработка суперпикселя {sp_id} ---")
        reassigned_first = process_superpixel(sp_id, data, set())

        # Визуализация первого этапа
        visualize_intermediate(original_sp, reassigned_first, data, stage="first")

        # Формируем обновлённые данные для второго этапа
        # Создаём копию данных и обновляем точки для обработанного суперпикселя
        updated_data = json.loads(json.dumps(data))  # глубокое копирование

        # Обновляем точки для обработанного суперпикселя
        sp_updated = next((s for s in updated_data["superpixels"] if s["id"] == sp_id), None)
        if sp_updated:
            # Оставляем только точки, которые остались в этом суперпикселе
            kept_points = reassigned_first.get(sp_id, [])
            sp_updated["all_points"] = [{"x": p[0], "y": p[1]} for p in kept_points]

        # Обновляем точки для соседей, которые получили новые точки
        for neighbor_id, points in reassigned_first.items():
            if neighbor_id != sp_id:
                neighbor_sp = next((s for s in updated_data["superpixels"] if s["id"] == neighbor_id), None)
                if neighbor_sp:
                    # Добавляем новые точки к существующим
                    existing_points = [(p["x"], p["y"]) for p in neighbor_sp["all_points"]]
                    new_points = [(p[0], p[1]) for p in points]
                    all_points = existing_points + new_points
                    neighbor_sp["all_points"] = [{"x": p[0], "y": p[1]} for p in all_points]

        # Этап 2: Обработка первого соседа
        # Важно: исключаем уже обработанный суперпиксель из списка доступных для перераспределения
        print(f"\n--- Этап 2: Обработка суперпикселя {first_neighbor_id} ---")
        print(f"(Суперпиксель {sp_id} уже обработан, его точки не будут перераспределяться)")

        processed_ids = {sp_id}  # уже обработанные суперпиксели
        reassigned_second = process_superpixel(first_neighbor_id, updated_data, processed_ids)

        # Визуализация финального результата (все три этапа)
        visualize_result(original_sp, reassigned_first, reassigned_second, updated_data)

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

    # При желании можно сохранить результат в JSON
    # with open("reassigned_points_full.json", "w") as f:
    #     # Здесь нужно сохранить финальное состояние
    #     pass