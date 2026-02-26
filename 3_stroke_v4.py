import json
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.patches import Polygon
import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
import json


def reassign_superpixel_pixels(labels, centers, spline_pixels, processed_superpixel_id):
    """
    Переопределяет пиксели суперпикселя после создания сплайнов.

    Parameters:
    labels - матрица меток суперпикселей (H, W)
    centers - центры суперпикселей (N, 5)
    spline_pixels - список пикселей сплайнов [(y1, x1), (y2, x2), ...]
    processed_superpixel_id - ID обработанного суперпикселя

    Returns:
    new_labels - новая матрица меток
    new_centers - обновленные центры суперпикселей
    """
    height, width = labels.shape

    # 1. Создаем копию меток для модификации
    new_labels = labels.copy()

    # 2. Получаем маску обработанного суперпикселя
    original_mask = labels == processed_superpixel_id

    # 3. Создаем маску сплайнов (пиксели, которые остаются в обработанном суперпикселе)
    spline_mask = np.zeros_like(original_mask, dtype=bool)

    # Если заданы конкретные пиксели сплайнов
    if spline_pixels is not None:
        for y, x in spline_pixels:
            if 0 <= y < height and 0 <= x < width:
                spline_mask[y, x] = True

    # 4. Определяем пиксели для перераспределения (все кроме сплайнов)
    pixels_to_redistribute = original_mask & ~spline_mask

    # Если пикселей для перераспределения нет, возвращаем исходные данные
    if not np.any(pixels_to_redistribute):
        return new_labels, centers

    # 5. Находим соседние суперпиксели
    neighbor_ids = find_superpixel_neighbors_for_redistribution(
        labels, processed_superpixel_id
    )

    if not neighbor_ids:
        # Если соседей нет, оставляем все как есть
        return new_labels, centers

    # 6. Собираем характеристики соседних суперпикселей для перераспределения
    neighbor_features = []
    valid_neighbor_ids = []

    for neighbor_id in neighbor_ids:
        neighbor_mask = labels == neighbor_id
        if np.any(neighbor_mask):
            # Получаем координаты и цвет пикселей соседа
            y_coords, x_coords = np.where(neighbor_mask)
            neighbor_center = centers[neighbor_id]

            # Вычисляем средний цвет соседа
            neighbor_features.append({
                'id': neighbor_id,
                'center_coords': (neighbor_center[0], neighbor_center[1]),  # y, x
                'center_color': neighbor_center[2:],  # Lab
                'pixel_count': len(y_coords)
            })
            valid_neighbor_ids.append(neighbor_id)

    # 7. Перераспределяем пиксели
    redistribute_pixels_to_neighbors(
        new_labels,
        pixels_to_redistribute,
        valid_neighbor_ids,
        centers,
        labels
    )

    # 8. Обновляем центры суперпикселей
    new_centers = update_centers_after_redistribution(
        new_labels, centers, valid_neighbor_ids + [processed_superpixel_id]
    )

    return new_labels, new_centers


def find_superpixel_neighbors_for_redistribution(labels, superpixel_id):
    """
    Находит соседние суперпиксели для перераспределения пикселей.

    Parameters:
    labels - матрица меток
    superpixel_id - ID суперпикселя

    Returns:
    list - список ID соседних суперпикселей
    """
    height, width = labels.shape
    neighbors = set()

    # Получаем границы суперпикселя
    superpixel_mask = labels == superpixel_id

    # Используем морфологические операции для нахождения границ
    structure = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(superpixel_mask, structure)
    boundaries = superpixel_mask & ~eroded

    # Находим соседей на границах
    boundary_y, boundary_x = np.where(boundaries)

    for y, x in zip(boundary_y, boundary_x):
        # Проверяем 8-связных соседей
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_id = labels[ny, nx]
                    if neighbor_id != superpixel_id and neighbor_id != -1:
                        neighbors.add(neighbor_id)

    return list(neighbors)


def redistribute_pixels_to_neighbors(new_labels, pixels_mask, neighbor_ids,
                                     centers, original_labels):
    """
    Перераспределяет пиксели между соседними суперпикселями.

    Parameters:
    new_labels - матрица меток для модификации
    pixels_mask - маска пикселей для перераспределения
    neighbor_ids - список ID соседних суперпикселей
    centers - центры суперпикселей
    original_labels - исходная матрица меток
    """
    height, width = new_labels.shape

    # Получаем координаты пикселей для перераспределения
    y_coords, x_coords = np.where(pixels_mask)

    if len(y_coords) == 0:
        return

    # Подготовка данных для k-NN
    # Используем пространственные координаты и цвет
    neighbor_features = []
    neighbor_indices = []

    for neighbor_id in neighbor_ids:
        neighbor_mask = original_labels == neighbor_id
        if np.any(neighbor_mask):
            n_y, n_x = np.where(neighbor_mask)
            # Берем случайную выборку пикселей для обучения
            sample_size = min(100, len(n_y))
            indices = np.random.choice(len(n_y), sample_size, replace=False)

            for idx in indices:
                y, x = n_y[idx], n_x[idx]
                neighbor_features.append([y, x, centers[neighbor_id, 2],
                                          centers[neighbor_id, 3], centers[neighbor_id, 4]])
                neighbor_indices.append(neighbor_id)

    if not neighbor_features:
        return

    # Обучаем k-NN
    from sklearn.neighbors import KNeighborsClassifier

    X_train = np.array(neighbor_features)
    y_train = np.array(neighbor_indices)

    # Используем взвешенное расстояние (пространство + цвет)
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean'
    )
    knn.fit(X_train, y_train)

    # Классифицируем пиксели для перераспределения
    for y, x in zip(y_coords, x_coords):
        # Получаем цвет пикселя (нужен доступ к изображению)
        # Вместо этого используем усредненный цвет суперпикселя
        original_id = original_labels[y, x]
        if original_id < len(centers):
            pixel_features = np.array([[
                y, x,
                centers[original_id, 2],  # L
                centers[original_id, 3],  # a
                centers[original_id, 4]  # b
            ]])

            # Предсказываем ближайшего соседа
            predicted_id = knn.predict(pixel_features)[0]
            new_labels[y, x] = predicted_id


def update_centers_after_redistribution(labels, old_centers, affected_ids):
    """
    Обновляет центры суперпикселей после перераспределения.

    Parameters:
    labels - новая матрица меток
    old_centers - старые центры
    affected_ids - список затронутых суперпикселей

    Returns:
    np.array - обновленные центры
    """
    height, width = labels.shape
    new_centers = old_centers.copy()

    for superpixel_id in affected_ids:
        mask = labels == superpixel_id

        if np.any(mask):
            # Получаем координаты пикселей
            y_coords, x_coords = np.where(mask)

            # Обновляем пространственные координаты центра
            new_centers[superpixel_id, 0] = np.mean(y_coords)  # y
            new_centers[superpixel_id, 1] = np.mean(x_coords)  # x

            # Цветовые каналы не обновляем, так как нет доступа к изображению
            # В реальном использовании здесь нужно пересчитывать средний цвет
        else:
            # Если суперпиксель стал пустым, помечаем его для удаления
            new_centers[superpixel_id] = [-1, -1, -1, -1, -1]

    return new_centers


def save_redistribution_results(labels, centers, filename='redistributed_superpixels.json'):
    """
    Сохраняет результаты перераспределения в JSON.

    Parameters:
    labels - матрица меток после перераспределения
    centers - центры суперпикселей
    filename - имя файла для сохранения
    """
    height, width = labels.shape
    data = {
        'image_dimensions': {
            'height': int(height),
            'width': int(width)
        },
        'num_superpixels': len(centers),
        'superpixels': []
    }

    # Находим соседей для новой конфигурации
    neighbors = find_superpixel_neighbors(labels)

    for k in range(len(centers)):
        mask = labels == k

        if not np.any(mask):
            continue

        # Находим граничные точки
        from scipy import ndimage
        structure = ndimage.generate_binary_structure(2, 2)
        eroded = ndimage.binary_erosion(mask, structure)
        boundaries = mask & ~eroded

        boundary_y, boundary_x = np.where(boundaries)
        all_y, all_x = np.where(mask)

        # Формируем данные
        superpixel_data = {
            'id': int(k),
            'center': {
                'y': float(centers[k, 0]),
                'x': float(centers[k, 1])
            },
            'boundary_points': [{'x': int(x), 'y': int(y)}
                                for y, x in zip(boundary_y, boundary_x)],
            'all_points': [{'x': int(x), 'y': int(y)}
                           for y, x in zip(all_y, all_x)],
            'area': int(len(all_y)),
            'neighbors': sorted(list(neighbors[k])) if k < len(neighbors) else []
        }

        data['superpixels'].append(superpixel_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Результаты перераспределения сохранены в {filename}")


# Пример использования в 3_stroke_v4.py
def apply_superpixel_redistribution(result_data, original_labels, original_centers,
                                    processed_superpixel_id):
    """
    Применяет перераспределение пикселей для одного суперпикселя.

    Parameters:
    result_data - данные из analyze_and_plot_polygon
    original_labels - исходные метки суперпикселей
    original_centers - исходные центры
    processed_superpixel_id - ID обработанного суперпикселя

    Returns:
    tuple - (new_labels, new_centers)
    """
    # 1. Получаем пиксели сплайнов
    spline_pixels = []

    # Добавляем точки исходного сплайна
    if result_data['x_spline'] is not None and result_data['y_spline'] is not None:
        for x, y in zip(result_data['x_spline'], result_data['y_spline']):
            spline_pixels.append((int(round(y)), int(round(x))))

    # Добавляем точки параллельного сплайна
    if result_data['x_parallel'] is not None and result_data['y_parallel'] is not None:
        for x, y in zip(result_data['x_parallel'], result_data['y_parallel']):
            spline_pixels.append((int(round(y)), int(round(x))))

    # 2. Применяем перераспределение
    new_labels, new_centers = reassign_superpixel_pixels(
        original_labels,
        original_centers,
        spline_pixels,
        processed_superpixel_id
    )

    return new_labels, new_centers


# Вспомогательная функция для поиска соседей (из 1_segmentation.py)
def find_superpixel_neighbors(labels):
    """
    Находит соседние суперпиксели для каждого суперпикселя
    """
    height, width = labels.shape
    num_superpixels = np.max(labels) + 1
    neighbors = [set() for _ in range(num_superpixels)]

    # Проверяем соседей по горизонтали и вертикали
    for i in range(height - 1):
        for j in range(width - 1):
            current = labels[i, j]
            right = labels[i, j + 1]
            down = labels[i + 1, j]

            if current != right:
                neighbors[current].add(right)
                neighbors[right].add(current)
            if current != down:
                neighbors[current].add(down)
                neighbors[down].add(current)

    return neighbors

def load_data(filename):
    """Загрузка данных из JSON файла"""
    with open(filename, 'r') as f:
        return json.load(f)


def calculate_angle_to_center(edge_points, center):
    """Вычисление угла между векторами от центра к вершинам ребра"""
    v1, v2 = edge_points

    vec1 = np.array([v1[0] - center[0], v1[1] - center[1]])
    vec2 = np.array([v2[0] - center[0], v2[1] - center[1]])

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 180

    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_angle))
    return abs(angle - 90)


def find_best_edges(polygon_data):
    """Нахождение двух лучших рёбер с углами, ближайшими к 90 градусам"""
    vertices = [(v['x'], v['y']) for v in polygon_data['vertices']]
    center = (polygon_data['center']['x'], polygon_data['center']['y'])
    n = len(vertices)

    edges = []
    for i in range(n):
        v1_idx = i
        v2_idx = (i + 1) % n
        edge_points = (vertices[v1_idx], vertices[v2_idx])
        deviation = calculate_angle_to_center(edge_points, center)
        edge_length = math.dist(edge_points[0], edge_points[1])

        edges.append({
            'indices': (v1_idx, v2_idx),
            'points': edge_points,
            'deviation': deviation,
            'length': edge_length
        })

    edges.sort(key=lambda x: x['deviation'])

    selected_edges = []
    used_vertices = set()

    for edge in edges:
        v1_idx, v2_idx = edge['indices']

        if v1_idx not in used_vertices and v2_idx not in used_vertices:
            selected_edges.append(edge)
            used_vertices.add(v1_idx)
            used_vertices.add(v2_idx)

            if len(selected_edges) == 2:
                break

    if len(selected_edges) < 2:
        selected_edges = edges[:2]

    # Определяем какое ребро меньше (по длине)
    if selected_edges[0]['length'] <= selected_edges[1]['length']:
        return vertices, center, selected_edges[0], selected_edges[1]
    else:
        return vertices, center, selected_edges[1], selected_edges[0]


def find_path_to_first_vertex_of_second_edge(vertices, edge1, edge2):
    """
    Нахождение пути от вершины первого ребра до первой вершины второго ребра
    """
    n = len(vertices)
    edge1_idx1, edge1_idx2 = edge1['indices']
    edge2_idx1, edge2_idx2 = edge2['indices']

    start_candidates = [edge1_idx1, edge1_idx2]

    best_path = []
    max_intermediate = -1

    for start_idx in start_candidates:
        forbidden_idx = edge1_idx2 if start_idx == edge1_idx1 else edge1_idx1

        for direction in [1, -1]:
            path = []
            current = start_idx
            found_end = False

            while True:
                path.append(current)

                if current in [edge2_idx1, edge2_idx2]:
                    found_end = True
                    break

                if current == forbidden_idx:
                    break

                current = (current + direction) % n

                if current == start_idx:
                    break

            if found_end and len(path) > 1:
                if forbidden_idx in path:
                    continue

                intermediate_count = len(path) - 2
                if intermediate_count > max_intermediate:
                    max_intermediate = intermediate_count
                    best_path = path.copy()

    return best_path


def create_spline_from_path(path_points, smoothness=100):
    """Создание кубического сплайна через точки пути"""
    if len(path_points) < 2:
        return None, None, None, None, None

    x_vals = [p[0] for p in path_points]
    y_vals = [p[1] for p in path_points]

    distances = [0]
    for i in range(1, len(path_points)):
        dist = math.dist(path_points[i - 1], path_points[i])
        distances.append(distances[-1] + dist)

    try:
        t = np.array(distances)
        cs_x = CubicSpline(t, x_vals)
        cs_y = CubicSpline(t, y_vals)

        t_smooth = np.linspace(t[0], t[-1], smoothness)
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)

        return x_smooth, y_smooth, cs_x, cs_y, t
    except Exception as e:
        print(f"Ошибка при создании сплайна: {e}")
        return x_vals, y_vals, None, None, None


def find_point_on_circle_on_edge(start_point, edge_points, radius, center):
    """
    Нахождение точки на ребре, находящейся на расстоянии radius от начальной точки
    и лежащей на окружности с центром в start_point
    """
    v1, v2 = edge_points

    # Уравнение ребра: параметрическая форма
    # x = v1.x + t*(v2.x - v1.x)
    # y = v1.y + t*(v2.y - v1.y), где t ∈ [0, 1]

    # Уравнение окружности: (x - start_point.x)^2 + (y - start_point.y)^2 = radius^2

    # Подставляем параметрическое уравнение прямой в уравнение окружности
    dx = v2[0] - v1[0]
    dy = v2[1] - v1[1]

    # Параметры квадратного уравнения
    a = dx ** 2 + dy ** 2
    b = 2 * (dx * (v1[0] - start_point[0]) + dy * (v1[1] - start_point[1]))
    c = (v1[0] - start_point[0]) ** 2 + (v1[1] - start_point[1]) ** 2 - radius ** 2

    # Решаем квадратное уравнение
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        # Нет пересечения, возвращаем точку на ребре, ближайшую к start_point
        return find_closest_point_on_edge(start_point, v1, v2)

    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    # Выбираем t в пределах [0, 1]
    valid_ts = []
    for t in [t1, t2]:
        if 0 <= t <= 1:
            valid_ts.append(t)

    if not valid_ts:
        # Если нет пересечений в пределах отрезка, возвращаем ближайшую точку
        return find_closest_point_on_edge(start_point, v1, v2)

    # Из всех возможных точек выбираем ту, которая дальше от центра
    # (чтобы двигаться наружу от многоугольника)
    points = []
    for t in valid_ts:
        point = (
            v1[0] + t * dx,
            v1[1] + t * dy
        )
        # Вычисляем расстояние от центра
        dist_to_center = math.dist(point, center)
        points.append((dist_to_center, point))

    # Выбираем точку, наиболее удаленную от центра (внешнюю)
    points.sort(reverse=True)
    return points[0][1]


def find_closest_point_on_edge(point, edge_start, edge_end):
    """Нахождение ближайшей точки на ребре к заданной точке"""
    # Вектор ребра
    edge_vec = np.array([edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]])
    edge_length_sq = np.dot(edge_vec, edge_vec)

    if edge_length_sq == 0:
        return edge_start

    # Вектор от начала ребра к точке
    point_vec = np.array([point[0] - edge_start[0], point[1] - edge_start[1]])

    # Проекция
    t = np.dot(point_vec, edge_vec) / edge_length_sq
    t = np.clip(t, 0, 1)

    # Ближайшая точка на ребре
    closest_x = edge_start[0] + t * edge_vec[0]
    closest_y = edge_start[1] + t * edge_vec[1]

    return (closest_x, closest_y)


def create_parallel_spline_with_circle_method(vertices, center, edge1, edge2, path_indices, offset_distance):
    """Создание параллельного сплайна методом окружностей"""
    # Получаем точки пути
    path_points = [vertices[i] for i in path_indices]

    # Начальная точка пути (на ребре edge1)
    start_point = path_points[0]
    # Конечная точка пути (на ребре edge2)
    end_point = path_points[-1]

    # Находим новые точки на рёбрах с помощью метода окружностей
    new_start_point = find_point_on_circle_on_edge(start_point, edge1['points'], offset_distance, center)
    new_end_point = find_point_on_circle_on_edge(end_point, edge2['points'], offset_distance, center)

    # Смещаем промежуточные точки по тому же вектору смещения
    # Вычисляем вектор смещения для начальной точки
    start_shift_x = new_start_point[0] - start_point[0]
    start_shift_y = new_start_point[1] - start_point[1]

    # Вычисляем вектор смещения для конечной точки
    end_shift_x = new_end_point[0] - end_point[0]
    end_shift_y = new_end_point[1] - end_point[1]

    # Для промежуточных точек интерполируем смещение
    offset_points = []
    for i, point in enumerate(path_points):
        if i == 0:
            # Начальная точка
            offset_points.append(new_start_point)
        elif i == len(path_points) - 1:
            # Конечная точка
            offset_points.append(new_end_point)
        else:
            # Промежуточные точки
            # Интерполируем смещение в зависимости от позиции в пути
            t = i / (len(path_points) - 1)
            shift_x = start_shift_x * (1 - t) + end_shift_x * t
            shift_y = start_shift_y * (1 - t) + end_shift_y * t

            new_point = (
                point[0] + shift_x,
                point[1] + shift_y
            )
            offset_points.append(new_point)

    # Создаём сплайн через смещённые точки
    if len(offset_points) < 2:
        return None, None, offset_points

    x_vals = [p[0] for p in offset_points]
    y_vals = [p[1] for p in offset_points]

    distances = [0]
    for i in range(1, len(offset_points)):
        dist = math.dist(offset_points[i - 1], offset_points[i])
        distances.append(distances[-1] + dist)

    try:
        t = np.array(distances)
        cs_x_parallel = CubicSpline(t, x_vals)
        cs_y_parallel = CubicSpline(t, y_vals)

        t_smooth = np.linspace(t[0], t[-1], 100)
        x_parallel = cs_x_parallel(t_smooth)
        y_parallel = cs_y_parallel(t_smooth)

        return x_parallel, y_parallel, offset_points
    except Exception as e:
        print(f"Ошибка при создании параллельного сплайна: {e}")
        return None, None, offset_points


def analyze_and_plot_polygon(polygon_data):
    """Анализ и отрисовка одного многоугольника"""
    vertices, center, edge1, edge2 = find_best_edges(polygon_data)

    print(f"Найдены рёбра:")
    print(f"  Меньшее ребро: вершины {edge1['indices']}, длина: {edge1['length']:.2f}")
    print(f"  Второе ребро: вершины {edge2['indices']}, длина: {edge2['length']:.2f}")

    # Находим путь от вершины первого ребра до первой вершины второго ребра
    path_indices = find_path_to_first_vertex_of_second_edge(vertices, edge1, edge2)

    if not path_indices:
        print("Не удалось найти подходящий путь!")
        return None

    print(f"Путь для сплайна: {path_indices}")
    print(f"Начальная точка: вершина {path_indices[0]} = {vertices[path_indices[0]]}")
    print(f"Конечная точка: вершина {path_indices[-1]} = {vertices[path_indices[-1]]}")

    # Получаем точки пути
    path_points = [vertices[i] for i in path_indices]

    # Создаём исходный сплайн
    x_spline, y_spline, cs_x, cs_y, t_values = create_spline_from_path(path_points)

    # Вычисляем смещение (длина меньшего ребра - 1), округляем
    offset_distance = round(edge1['length'] - 1)
    print(f"Смещение для параллельного сплайна (округлено): {offset_distance}")

    # Создаём параллельный сплайн методом окружностей
    x_parallel, y_parallel, offset_points = create_parallel_spline_with_circle_method(
        vertices, center, edge1, edge2, path_indices, offset_distance
    )

    return {
        'vertices': vertices,
        'center': center,
        'edge1': edge1,
        'edge2': edge2,
        'path_indices': path_indices,
        'path_points': path_points,
        'x_spline': x_spline,
        'y_spline': y_spline,
        'x_parallel': x_parallel,
        'y_parallel': y_parallel,
        'offset_points': offset_points,
        'offset_distance': offset_distance
    }


def plot_polygon_with_parallel_spline(result, polygon_id):
    """Отрисовка многоугольника с параллельным сплайном"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    vertices = result['vertices']
    center = result['center']
    edge1 = result['edge1']
    edge2 = result['edge2']
    path_indices = result['path_indices']
    path_points = result['path_points']
    x_spline = result['x_spline']
    y_spline = result['y_spline']
    x_parallel = result['x_parallel']
    y_parallel = result['y_parallel']
    offset_points = result['offset_points']
    offset_distance = result['offset_distance']

    # ===== ЛЕВЫЙ ГРАФИК: Общий вид =====
    # 1. Рисуем многоугольник
    poly = Polygon(vertices, closed=True, fill=False,
                   edgecolor='blue', linewidth=2, alpha=0.7)
    ax1.add_patch(poly)

    # 2. Рисуем вершины с номерами
    for i, (x, y) in enumerate(vertices):
        ax1.plot(x, y, 'bo', markersize=8)
        ax1.text(x, y, f'{i}', fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 3. Рисуем центр
    ax1.plot(center[0], center[1], 'ro', markersize=10, label='Center')

    # 4. Выделяем выбранные рёбра
    edge1_x = [edge1['points'][0][0], edge1['points'][1][0]]
    edge1_y = [edge1['points'][0][1], edge1['points'][1][1]]
    ax1.plot(edge1_x, edge1_y, 'g-', linewidth=4, label='Edge 1 (меньшее)')

    edge2_x = [edge2['points'][0][0], edge2['points'][1][0]]
    edge2_y = [edge2['points'][0][1], edge2['points'][1][1]]
    ax1.plot(edge2_x, edge2_y, 'y-', linewidth=4, label='Edge 2')

    # 5. Подсвечиваем вершины пути
    for i, idx in enumerate(path_indices):
        color = 'red' if i == 0 else 'green' if i == len(path_indices) - 1 else 'orange'
        marker = 's' if i == 0 else 'D' if i == len(path_indices) - 1 else 'o'
        label = 'Начало пути' if i == 0 else 'Конец пути' if i == len(path_indices) - 1 else 'Промежут. вершина'
        ax1.plot(vertices[idx][0], vertices[idx][1], marker=marker,
                 color=color, markersize=12, markeredgecolor='black',
                 markeredgewidth=2, label=label if i < 3 else '')

    # 6. Рисуем исходный сплайн
    if x_spline is not None and y_spline is not None:
        ax1.plot(x_spline, y_spline, 'r-', linewidth=3, label='Исходный сплайн')

    # 7. Рисуем параллельный сплайн
    if x_parallel is not None and y_parallel is not None:
        ax1.plot(x_parallel, y_parallel, 'm--', linewidth=3, label='Параллельный сплайн')

    # 8. Отмечаем смещённые точки
    if offset_points:
        for i, point in enumerate(offset_points):
            color = 'magenta' if i == 0 or i == len(offset_points) - 1 else 'purple'
            marker = 's' if i == 0 else 'D' if i == len(offset_points) - 1 else 'o'
            label = 'Начало парал.' if i == 0 else 'Конец парал.' if i == len(offset_points) - 1 else ''
            ax1.plot(point[0], point[1], marker=marker, color=color,
                     markersize=10, markeredgecolor='black', markeredgewidth=1,
                     label=label if i == 0 or i == len(offset_points) - 1 else '')

    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_title(f"Многоугольник ID: {polygon_id} - Общий вид")

    # ===== ПРАВЫЙ ГРАФИК: Детали сплайнов и окружностей =====
    # 1. Рисуем рёбра
    ax2.plot(edge1_x, edge1_y, 'g-', linewidth=3, label='Ребро 1 (меньшее)', alpha=0.7)
    ax2.plot(edge2_x, edge2_y, 'y-', linewidth=3, label='Ребро 2', alpha=0.7)

    # 2. Рисуем окружности для нахождения точек
    if offset_points:
        # Окружность для начальной точки
        circle1 = plt.Circle(path_points[0], offset_distance, color='lightgray',
                             fill=False, linestyle=':', linewidth=1)
        ax2.add_patch(circle1)

        # Окружность для конечной точки
        circle2 = plt.Circle(path_points[-1], offset_distance, color='lightgray',
                             fill=False, linestyle=':', linewidth=1)
        ax2.add_patch(circle2)

    # 3. Рисуем исходный сплайн
    if x_spline is not None and y_spline is not None:
        ax2.plot(x_spline, y_spline, 'b-', linewidth=2, label='Исходный сплайн', alpha=0.7)

    # 4. Рисуем параллельный сплайн
    if x_parallel is not None and y_parallel is not None:
        ax2.plot(x_parallel, y_parallel, 'r--', linewidth=3, label='Параллельный сплайн')

    # 5. Рисуем исходные вершины пути
    for i, (x, y) in enumerate(path_points):
        color = 'red' if i == 0 else 'green' if i == len(path_points) - 1 else 'orange'
        ax2.plot(x, y, 'o', color=color, markersize=8, alpha=0.7)
        ax2.text(x, y, f'{i}', fontsize=10, ha='right', va='bottom')

    # 6. Рисуем смещённые точки
    if offset_points:
        for i, point in enumerate(offset_points):
            color = 'magenta' if i == 0 or i == len(offset_points) - 1 else 'purple'
            ax2.plot(point[0], point[1], 'x', color=color, markersize=10)
            ax2.text(point[0], point[1], f'({int(round(point[0]))},{int(round(point[1]))})',
                     fontsize=8, ha='left', va='bottom')

    # 7. Показываем линии смещения для ключевых точек
    if offset_points and len(path_points) == len(offset_points):
        for i in [0, len(path_points) // 2, -1]:
            idx = i if i >= 0 else len(path_points) - 1
            ax2.plot([path_points[idx][0], offset_points[idx][0]],
                     [path_points[idx][1], offset_points[idx][1]],
                     'k:', linewidth=1, alpha=0.5)

    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    ax2.set_title(f"Детали сплайнов (смещение: {offset_distance})")

    # Информация
    info_text = f"ID: {polygon_id}\n"
    info_text += f"Меньшее ребро: {edge1['indices']}, длина: {edge1['length']:.1f}\n"
    info_text += f"Второе ребро: {edge2['indices']}, длина: {edge2['length']:.1f}\n"
    info_text += f"Смещение (округлено): {offset_distance}\n"
    info_text += f"Путь: {path_indices}\n"

    if offset_points:
        info_text += f"Начало парал.: ({int(round(offset_points[0][0]))}, {int(round(offset_points[0][1]))})\n"
        info_text += f"Конец парал.: ({int(round(offset_points[-1][0]))}, {int(round(offset_points[-1][1]))})"

    plt.figtext(0.02, 0.02, info_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    plt.tight_layout()
    plt.show()


def main():
    # 1. Загружаем данные обработанных многоугольников
    data = load_data('processed_superpixels.json')

    # 2. ЗАГРУЖАЕМ ИСХОДНЫЕ СУПЕРПИКСЕЛИ из файла, созданного 1_segmentation.py
    original_labels, original_centers = load_labels_and_centers_from_json(
        'superpixels_full_4.json'  # Убедитесь, что имя файла совпадает с вашим
    )

    # Создаем копии для модификации
    current_labels = original_labels.copy()
    current_centers = original_centers.copy()

    # Выбираем многоугольники для анализа (ID из примеров)
    target_ids = [10, 25, 50]

    for polygon_id in target_ids:
        print(f"\n{'=' * 60}")
        print(f"Анализ многоугольника ID: {polygon_id}")
        print('=' * 60)

        # Находим многоугольник по ID
        polygon_data = None
        for poly in data['processed_superpixels']:
            if poly['id'] == polygon_id:
                polygon_data = poly
                break

        if polygon_data is None:
            print(f"Многоугольник с ID {polygon_id} не найден!")
            continue

        # Анализируем и строим графики
        result = analyze_and_plot_polygon(polygon_data)

        if result:
            # Отображаем исходные графики
            plot_polygon_with_parallel_spline(result, polygon_id)

            # !!! ПЕРЕРАСПРЕДЕЛЕНИЕ ПИКСЕЛЕЙ ДЛЯ ЭТОГО СУПЕРПИКСЕЛЯ !!!
            print(f"\n--- Перераспределение пикселей для суперпикселя {polygon_id} ---")

            # Получаем пиксели сплайнов
            spline_pixels = []

            # Добавляем точки исходного сплайна
            if result['x_spline'] is not None and result['y_spline'] is not None:
                for x, y in zip(result['x_spline'], result['y_spline']):
                    spline_pixels.append((int(round(y)), int(round(x))))

            # Добавляем точки параллельного сплайна
            if result['x_parallel'] is not None and result['y_parallel'] is not None:
                for x, y in zip(result['x_parallel'], result['y_parallel']):
                    spline_pixels.append((int(round(y)), int(round(x))))

            # Применяем перераспределение
            new_labels, new_centers = reassign_superpixel_pixels(
                current_labels,
                current_centers,
                spline_pixels,
                polygon_id
            )

            # Обновляем для следующей итерации
            current_labels = new_labels
            current_centers = new_centers

            # Визуализируем результат перераспределения
            visualize_redistribution_result(
                original_labels,  # исходные для сравнения
                current_labels,  # новые после перераспределения
                polygon_id,
                spline_pixels
            )

    # Сохраняем финальный результат после обработки всех target_ids
    save_redistribution_results(
        current_labels,
        current_centers,
        'redistributed_target_superpixels.json'
    )

    print(f"\nОбработано суперпикселей: {target_ids}")
    print(f"Результаты сохранены в 'redistributed_target_superpixels.json'")


def visualize_redistribution_result(original_labels, new_labels, superpixel_id, spline_pixels):
    """
    Визуализация результата перераспределения пикселей
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Исходный суперпиксель
    original_mask = original_labels == superpixel_id
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title(f'Исходный суперпиксель ID: {superpixel_id}')
    axes[0].axis('off')

    # 2. Суперпиксель после перераспределения
    new_mask = new_labels == superpixel_id
    axes[1].imshow(new_mask, cmap='gray')
    axes[1].set_title(f'После перераспределения')
    axes[1].axis('off')

    # 3. Разница (какие пиксели были перераспределены)
    redistributed = original_mask & ~new_mask
    axes[2].imshow(redistributed, cmap='hot')
    axes[2].set_title(f'Перераспределенные пиксели (всего: {np.sum(redistributed)})')
    axes[2].axis('off')

    # Добавляем сплайны на изображение разницы
    if spline_pixels:
        spline_y, spline_x = zip(*spline_pixels)
        axes[2].plot(spline_x, spline_y, 'g.', markersize=1, alpha=0.5, label='Сплайны')
        axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Статистика
    original_count = np.sum(original_mask)
    new_count = np.sum(new_mask)
    redistributed_count = np.sum(redistributed)

    print(f"  Статистика суперпикселя {superpixel_id}:")
    print(f"    - Исходное количество пикселей: {original_count}")
    print(f"    - После перераспределения: {new_count}")
    print(f"    - Перераспределено пикселей: {redistributed_count}")
    print(f"    - Осталось пикселей сплайнов: {len(spline_pixels)}")


def load_labels_and_centers_from_json(json_file):
    """
    Загружает матрицу меток и центры из JSON файла, созданного 1_segmentation.py
    """
    print(f"Загрузка суперпикселей из файла: {json_file}")

    with open(json_file, 'r') as f:
        data = json.load(f)

    height = data['image_dimensions']['height']
    width = data['image_dimensions']['width']
    num_superpixels = data['num_superpixels']

    # Создаем матрицу меток
    labels = -1 * np.ones((height, width), dtype=np.int32)

    # Создаем массив центров
    centers = np.zeros((num_superpixels, 5), dtype=np.float64)

    # Заполняем данные
    for sp in data['superpixels']:
        sp_id = sp['id']
        center_y = sp['center']['y']
        center_x = sp['center']['x']

        centers[sp_id, 0] = center_y
        centers[sp_id, 1] = center_x
        # Цветовые каналы оставляем 0, они не критичны

        # Заполняем метки пикселей
        for point in sp['all_points']:
            y, x = point['y'], point['x']
            labels[y, x] = sp_id

    print(f"  Загружено: {num_superpixels} суперпикселей")
    print(f"  Размер изображения: {height}x{width}")

    return labels, centers


if __name__ == "__main__":
    main()