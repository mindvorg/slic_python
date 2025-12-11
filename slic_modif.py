import cv2
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt


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

    # Отображаем результаты
    plt.figure(dpi=100)
    plt.imshow(labels, cmap='gist_ncar', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.title('Labels Map')
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    plt.show()

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
              color='red', scale=15, width=0.003, headwidth=4,
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

