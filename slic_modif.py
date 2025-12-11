import cv2
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.colors import ListedColormap


def slic_modif(imgcol, u, v, p=2.0, num_superpixels=100, compactness=10, max_iterations=10, random_init=True,
               poisson_radius_factor=0.5):
    """
    Модифицированный алгоритм SLIC с учетом градиента и вытянутости

    Parameters:
    imgcol - исходное цветное изображение (H, W, 3)
    u, v - массивы с cos(theta) и sin(theta) градиентов
    p - коэффициент вытянутости (p >= 1)
    num_superpixels - количество суперпикселей
    compactness - баланс между цветом и пространственным расстоянием
    max_iterations - максимальное количество итераций
    random_init - если True, центры инициализируются через распределение Пуассона
    poisson_radius_factor - коэффициент радиуса для распределения Пуассона (0.3-0.8)
    """

    height, width = imgcol.shape[:2]

    # Преобразуем изображение в Lab цветовое пространство
    img_lab = cv2.cvtColor(imgcol, cv2.COLOR_RGB2LAB).astype(np.float64)

    # Вычисляем шаг между центрами суперпикселей
    step = int(np.sqrt((height * width) / num_superpixels))

    # Инициализируем центры суперпикселей
    centers = []

    if random_init:
        # РАСПРЕДЕЛЕНИЕ ПУАССОНА
        # Вычисляем минимальное расстояние между центрами
        min_distance = step * poisson_radius_factor  # Используем параметр

        # Создаем объект PoissonDisk
        radius = min_distance / max(height, width)  # нормализуем радиус
        engine = qmc.PoissonDisk(
            d=2,
            radius=radius,
            hypersphere='volume',
            ncandidates=30,
            seed=42
        )

        # Генерируем точки в нормализованном пространстве [0, 1]
        points_normalized = engine.random(num_superpixels)

        # Масштабируем точки к размерам изображения
        for point in points_normalized:
            i = int(point[0] * (height - 1))
            j = int(point[1] * (width - 1))

            # Гарантируем, что координаты в пределах изображения
            i = np.clip(i, 0, height - 1)
            j = np.clip(j, 0, width - 1)

            centers.append([i, j, img_lab[i, j, 0], img_lab[i, j, 1], img_lab[i, j, 2]])
    else:
        # РАВНОМЕРНАЯ ИНИЦИАЛИЗАЦИЯ (оригинальная)
        for i in range(step // 2, height, step):
            for j in range(step // 2, width, step):
                centers.append([i, j, img_lab[i, j, 0], img_lab[i, j, 1], img_lab[i, j, 2]])

    centers = np.array(centers)
    num_centers = len(centers)

    # Создаем сетку координат
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    coordinates = np.dstack([y_coords, x_coords])

    # Матрицы для хранения меток и расстояний
    labels = -1 * np.ones((height, width), dtype=np.int32)
    distances = np.full((height, width), np.inf)

    for iteration in range(max_iterations):
        # СБРОС РАССТОЯНИЙ НА КАЖДОЙ ИТЕРАЦИИ - ВАЖНО!
        distances.fill(np.inf)

        # Для каждого центра
        for k in range(num_centers):
            center_y, center_x = int(centers[k, 0]), int(centers[k, 1])

            # УВЕЛИЧИВАЕМ ОБЛАСТЬ ПОИСКА ДЛЯ ПОКРЫТИЯ ВСЕХ ПИКСЕЛЕЙ
            search_radius = step * 2
            y_min = max(0, center_y - search_radius)
            y_max = min(height, center_y + search_radius)
            x_min = max(0, center_x - search_radius)
            x_max = min(width, center_x + search_radius)

            # Извлекаем регион интереса
            region_y = coordinates[y_min:y_max, x_min:x_max, 0]
            region_x = coordinates[y_min:y_max, x_min:x_max, 1]
            region_lab = img_lab[y_min:y_max, x_min:x_max]
            region_u = u[center_y, center_x]
            region_v = v[center_y, center_x]

            # Вычисляем модифицированное расстояние
            dy = region_y - center_y
            dx = region_x - center_x

            # Вычисляем x и y в повернутой системе координат
            x_rotated = dx * region_u + dy * region_v
            y_rotated = -dx * region_v + dy * region_u

            # Модифицированное евклидово расстояние с вытянутостью
            spatial_dist = np.sqrt(x_rotated ** 2 + (p * y_rotated) ** 2)

            # Цветовое расстояние в Lab пространстве
            color_dist = np.sqrt(
                (region_lab[:, :, 0] - centers[k, 2]) ** 2 +
                (region_lab[:, :, 1] - centers[k, 3]) ** 2 +
                (region_lab[:, :, 2] - centers[k, 4]) ** 2
            )

            # Комбинированное расстояние
            total_dist = np.sqrt((spatial_dist / step) ** 2 + (color_dist / compactness) ** 2)

            # Обновляем метки
            update_mask = total_dist < distances[y_min:y_max, x_min:x_max]
            labels[y_min:y_max, x_min:x_max][update_mask] = k
            distances[y_min:y_max, x_min:x_max][update_mask] = total_dist[update_mask]

        # Обновляем центры
        new_centers = np.zeros_like(centers)
        counts = np.zeros(num_centers)

        for k in range(num_centers):
            mask = labels == k
            if np.any(mask):
                y_coords_k = coordinates[:, :, 0][mask]
                x_coords_k = coordinates[:, :, 1][mask]
                lab_k = img_lab[mask]

                new_centers[k, 0] = np.mean(y_coords_k)  # y
                new_centers[k, 1] = np.mean(x_coords_k)  # x
                new_centers[k, 2:] = np.mean(lab_k, axis=0)  # Lab
                counts[k] = len(y_coords_k)

        # Заменяем только валидные центры
        valid_centers = counts > 0
        centers[valid_centers] = new_centers[valid_centers]

    # ПОСТ-ОБРАБОТКА: заполняем оставшиеся пустоты
    # labels = fill_unassigned_pixels(labels, centers)
    #labels = enforce_connectivity(labels, min_size=20)

    convex_hulls = build_convex_hulls(labels, centers)
    visualize_convex_hulls(labels, centers, convex_hulls,
                           show_points=True, show_centers=False,
                           alpha=0.3, figsize=(10, 10))

    return labels, centers


def fill_unassigned_pixels(labels, centers):
    """Заполняет пиксели, которые не были присвоены ни одному суперпикселю"""
    height, width = labels.shape

    # Находим незаполненные пиксели
    unassigned_mask = labels == -1

    if not np.any(unassigned_mask):
        return labels

    # Создаем копию меток для заполнения
    filled_labels = labels.copy()

    # Находим координаты незаполненных пикселей
    unassigned_y, unassigned_x = np.where(unassigned_mask)

    # Для каждого незаполненного пикселя находим ближайший центр
    for y, x in zip(unassigned_y, unassigned_x):
        min_dist = float('inf')
        best_label = -1

        for k, center in enumerate(centers):
            if center[0] >= 0 and center[1] >= 0:  # Проверяем валидность центра
                dist = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_label = k

        if best_label != -1:
            filled_labels[y, x] = best_label

    return filled_labels


def draw_superpixel_boundaries(image, labels):
    """Рисует границы суперпикселей на изображении"""
    boundaries = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Проверяем границы по вертикали
    vertical_boundaries = labels[:-1, :] != labels[1:, :]
    boundaries[:-1, :] = np.logical_or(boundaries[:-1, :], vertical_boundaries)
    boundaries[1:, :] = np.logical_or(boundaries[1:, :], vertical_boundaries)

    # Проверяем границы по горизонтали
    horizontal_boundaries = labels[:, :-1] != labels[:, 1:]
    boundaries[:, :-1] = np.logical_or(boundaries[:, :-1], horizontal_boundaries)
    boundaries[:, 1:] = np.logical_or(boundaries[:, 1:], horizontal_boundaries)

    # Создаем изображение с границами
    result = image.copy()
    result[boundaries > 0] = [255, 0, 0]  # Красные границы

    return result


def draw_gradient_vectors(image, labels, centers, u, v, scale=5):
    """Рисует векторы градиента в центроидах суперпикселей"""
    result = image.copy()

    for center in centers:
        y, x = int(center[0]), int(center[1])
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            # Получаем градиент в центроиде
            grad_u = u[y, x]
            grad_v = v[y, x]

            # Вычисляем конечную точку вектора
            end_y = int(y + scale * grad_v)
            end_x = int(x + scale * grad_u)

            # Рисуем вектор
            cv2.arrowedLine(result, (x, y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3)
            cv2.circle(result, (x, y), 2, (255, 0, 255), -1)  # Центроид

    return result


def draw_gradient_vectors_quiver(image, labels, centers, u, v, scale=2.0):
    """Рисует векторы градиента в центроидах суперпикселей с использованием quiver"""

    # Собираем координаты и направления для центроидов
    x_coords = []
    y_coords = []
    u_vectors = []
    v_vectors = []

    for center in centers:
        y, x = int(center[0]), int(center[1])
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            x_coords.append(x)
            y_coords.append(y)
            u_vectors.append(u[y, x] * scale)
            v_vectors.append(v[y, x] * scale)

    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(12, 8))

    # Отображаем исходное изображение
    ax.imshow(image)

    # Рисуем векторы градиента
    ax.quiver(x_coords, y_coords, u_vectors, v_vectors,
              color='blue', scale=15, width=0.003, headwidth=4,
              headlength=5, headaxislength=4.5)

    # Отмечаем центроиды точками
    ax.scatter(x_coords, y_coords, c='yellow', s=20, marker='o', alpha=0.7)

    ax.set_title('Векторы градиента в центроидах суперпикселей')
    ax.axis('off')

    return fig


def enforce_connectivity(labels, min_size=20):
    """
    Обеспечивает связность суперпикселей через анализ связных компонент

    Parameters:
    labels - метки суперпикселей
    min_size - минимальный размер компоненты для сохранения

    Returns:
    labels - исправленные метки с связными областями
    """
    from scipy import ndimage

    height, width = labels.shape
    new_labels = -1 * np.ones_like(labels)
    label_counter = 0

    # Проходим по всем уникальным меткам
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label

        # Находим связные компоненты для этой метки
        num_components, components = cv2.connectedComponents(
            mask.astype(np.uint8), connectivity=4
        )

        # plt.figure()
        # plt.imshow(components, cmap='rainbow')
        # plt.gca().invert_yaxis();
        # plt.axis('off')
        # plt.show()

        # Для каждой связной компоненты
        for comp_id in range(1, num_components):
            component_mask = components == comp_id
            component_size = np.sum(component_mask)

            if component_size < min_size:
                # Маленькие компоненты присоединяем к соседям
                # Находим соседние метки
                dilated = cv2.dilate(component_mask.astype(np.uint8),
                                     kernel=np.ones((3, 3), np.uint8))
                neighbors = dilated & ~component_mask
                neighbor_labels = labels[neighbors]
                if len(neighbor_labels) > 0:
                    # Присваиваем наиболее частую метку соседа
                    unique, counts = np.unique(neighbor_labels, return_counts=True)
                    new_label = unique[np.argmax(counts)]
                    new_labels[component_mask] = new_label
                else:
                    # Если нет соседей, сохраняем как новую метку
                    new_labels[component_mask] = label_counter
                    label_counter += 1
            else:
                # Большие компоненты сохраняем как отдельные суперпиксели
                new_labels[component_mask] = label_counter
                label_counter += 1

    return new_labels


def build_convex_hulls(labels, centers):
    """
    Построить выпуклые оболочки для каждого кластера.

    Parameters:
    -----------
    labels : ndarray, shape (H, W) или (N,)
        Массив меток кластеров
    centers : ndarray, shape (n_clusters, 2)
        Центры кластеров: centers[i, 0] = x, centers[i, 1] = y

    Returns:
    --------
    convex_hulls : dict
        Словарь, где ключ - номер кластера,
        значение - вершины выпуклой оболочки (ndarray shape (m, 2))
    """
    n_clusters = len(centers)

    # Если labels 2D (изображение), преобразуем в плоский вид
    if labels.ndim == 2:
        h, w = labels.shape
        y_coords, x_coords = np.mgrid[:h, :w]
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        flat_labels = labels.ravel()
    else:
        # Для 1D массива предполагаем, что координаты - это индексы
        n_points = len(labels)
        y_coords, x_coords = np.divmod(np.arange(n_points), labels.shape[1] if labels.ndim > 1 else n_points)
        points = np.column_stack([x_coords, y_coords])
        flat_labels = labels

    convex_hulls = {}

    # Для каждого кластера строим выпуклую оболочку
    for cluster_id in range(n_clusters):
        # Получаем точки текущего кластера
        mask = flat_labels == cluster_id
        cluster_points = points[mask]

        # Если точек достаточно для построения выпуклой оболочки
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
                # Сохраняем вершины выпуклой оболочки
                convex_hulls[cluster_id] = cluster_points[hull.vertices]
            except:
                # Если не удалось построить выпуклую оболочку, используем bounding box
                convex_hulls[cluster_id] = cluster_points
        elif len(cluster_points) > 0:
            # Для 1-2 точек сохраняем их как есть
            convex_hulls[cluster_id] = cluster_points

    return convex_hulls


def visualize_convex_hulls(labels, centers, convex_hulls=None,
                           show_points=True, show_centers=True,
                           alpha=0.3, figsize=(10, 10)):
    """
    Визуализация выпуклых оболочек кластеров.

    Parameters:
    -----------
    labels : ndarray
        Массив меток кластеров
    centers : ndarray
        Центры кластеров
    convex_hulls : dict, optional
        Предварительно вычисленные выпуклые оболочки
    show_points : bool
        Показывать ли точки кластеров
    show_centers : bool
        Показывать ли центры кластеров
    alpha : float
        Прозрачность заполнения выпуклых оболочек
    figsize : tuple
        Размер фигуры
    """
    if convex_hulls is None:
        convex_hulls = build_convex_hulls(labels, centers)

    matlab_lines_colors = [
            (0, 0.4470, 0.7410),  # Blue
            (0.8500, 0.3250, 0.0980),  # Orange
            (0.9290, 0.6940, 0.1250),  # Yellow
            (0.4940, 0.1840, 0.5560),  # Purple
            (0.4660, 0.6740, 0.1880),  # Green
            (0.3010, 0.7450, 0.9330),  # Light Blue
            (0.6350, 0.0780, 0.1840),  # Red
        ]

    colors = [matlab_lines_colors[i % 7] for i in range(len(convex_hulls))]

    # Создаем фигуру
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax0 = axes[0]
    # Отображаем результаты
    matlab_lines_cmap = ListedColormap(colors)
    ax0.imshow(labels, cmap=matlab_lines_cmap, interpolation='nearest')
    ax0.invert_yaxis()

    ax = axes[1]

    # Если нужно показать точки
    if show_points and labels.ndim == 2:
        h, w = labels.shape
        y_coords, x_coords = np.mgrid[:h, :w]
        flat_labels = labels.ravel()

        # Показываем все точки с цветами по кластерам
        scatter = ax.scatter(x_coords.ravel(), y_coords.ravel(),
                             c=flat_labels, cmap=matlab_lines_cmap, s=1, alpha=0.5)

    # Рисуем выпуклые оболочки
    for cluster_id, hull_points in convex_hulls.items():
        if len(hull_points) > 0:
            # Создаем полигон
            from matplotlib.patches import Polygon

            # Если точки образуют замкнутый контур
            if len(hull_points) >= 3:
                # Замыкаем полигон
                closed_points = np.vstack([hull_points, hull_points[0]])

                # Создаем и добавляем полигон
                polygon = Polygon(closed_points,
                                  closed=True,
                                  edgecolor=colors[cluster_id % len(colors)],
                                  facecolor=colors[cluster_id % len(colors)],
                                  alpha=alpha,
                                  linewidth=2)
                ax.add_patch(polygon)

            # Рисуем границы
            ax.plot(hull_points[:, 0], hull_points[:, 1],
                    color=colors[cluster_id % len(colors)],
                    linewidth=2, alpha=0.8)

    # Если нужно показать центры
    if show_centers:
        ax.scatter(centers[:, 0], centers[:, 1],
                   c='red', s=100, marker='X',
                   edgecolor='black', linewidth=2,
                   label='Centers')
        ax.legend()

    # Настройка осей
    if labels.ndim == 2:
        ax.set_xlim(0, labels.shape[1])
        ax.set_ylim(labels.shape[0], 0)  # Инвертируем ось Y для изображений
    else:
        # Автоматическое масштабирование
        ax.set_aspect('equal')

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Convex Hulls of {len(convex_hulls)} Clusters')
    ax.invert_yaxis()
    plt.tight_layout()
    return fig, ax
