import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import cv2

# Импорт функций визуализации из исходного модуля (они не зависят от алгоритма)
from slic_modif import (draw_superpixel_boundaries,
                        draw_gradient_vectors,
                        draw_gradient_vectors_quiver)

# Импорт нового алгоритма
from voronoi_segmentation import voronoi_segmentation


def numpy_to_python(obj):
    """Преобразует NumPy типы в стандартные Python типы для JSON сериализации"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def find_superpixel_neighbors(labels):
    """Находит соседние суперпиксели для каждого суперпикселя"""
    height, width = labels.shape
    num_superpixels = np.max(labels) + 1
    neighbors = [set() for _ in range(num_superpixels)]

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


def save_superpixels_to_json(labels, centers, imgcol, filename='superpixels_data.json'):
    """Сохраняет данные о суперпикселях в JSON, включая усреднённый RGB."""
    height, width = labels.shape
    superpixels_data = {
        'image_dimensions': {'height': height, 'width': width},
        'num_superpixels': len(centers),
        'superpixels': []
    }
    neighbors = find_superpixel_neighbors(labels)

    for k in range(len(centers)):
        mask = labels == k
        if not np.any(mask):
            continue

        # Средний RGB из исходного изображения
        avg_rgb = np.mean(imgcol[mask], axis=0).astype(int)
        rgb_color = avg_rgb.tolist()

        # Граничные точки
        boundaries = np.zeros((height, width), dtype=np.uint8)
        vertical_boundaries = labels[:-1, :] != labels[1:, :]
        boundaries[:-1, :] = np.logical_or(boundaries[:-1, :], vertical_boundaries)
        boundaries[1:, :] = np.logical_or(boundaries[1:, :], vertical_boundaries)
        horizontal_boundaries = labels[:, :-1] != labels[:, 1:]
        boundaries[:, :-1] = np.logical_or(boundaries[:, :-1], horizontal_boundaries)
        boundaries[:, 1:] = np.logical_or(boundaries[:, 1:], horizontal_boundaries)

        boundary_mask = np.logical_and(boundaries, mask)
        boundary_y, boundary_x = np.where(boundary_mask)
        all_y, all_x = np.where(mask)

        boundary_points = [{'x': int(x), 'y': int(y)} for y, x in zip(boundary_y, boundary_x)]
        all_points = [{'x': int(x), 'y': int(y)} for y, x in zip(all_y, all_x)]

        superpixel_data = {
            'id': int(k),
            'center': {
                'y': float(centers[k, 0]),
                'x': float(centers[k, 1])
            },
            'color_rgb': {
                'R': int(rgb_color[0]),
                'G': int(rgb_color[1]),
                'B': int(rgb_color[2])
            },
            'boundary_points': boundary_points,
            'all_points': all_points,
            'area': int(len(all_y)),
            'neighbors': sorted(list(neighbors[k]))
        }
        superpixels_data['superpixels'].append(superpixel_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(superpixels_data, f, indent=2, ensure_ascii=False, default=numpy_to_python)

    print(f"Данные суперпикселей (средний RGB) сохранены в {filename}")
    return superpixels_data


def main():
    filename = 'Lena_playboy.jpg'

    # Загружаем серое изображение и цветное (BGR -> RGB)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    img_bgr = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение {filename}")
    imgcol = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Вычисление градиентов (как в оригинале)
    kernely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernelx = -np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    u = cv2.filter2D(img, cv2.CV_8U, kernelx).astype(float)
    v = cv2.filter2D(img, cv2.CV_8U, kernely).astype(float)

    # Сглаживание
    nsize = 24
    s = 6
    ax = np.linspace(-(nsize - 1) / 2., (nsize - 1) / 2., nsize)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(s))
    kernel = np.outer(gauss, gauss)
    h = kernel / np.sum(kernel)
    u = sig.convolve2d(u, h, mode="same")
    v = sig.convolve2d(v, h, mode="same")

    # Нормализация направлений
    scl = np.sqrt(np.power(u, 2) + np.power(v, 2))
    scl = (scl == 0) + scl
    u = np.divide(u, scl)
    v = np.divide(v, scl)

    # Переворот цветного изображения (как в оригинале)
    imgcol = np.flipud(imgcol)

    # ----- ЗАМЕНА АЛГОРИТМА СЕГМЕНТАЦИИ -----
    labels, centers = voronoi_segmentation(
        imgcol, u, v,
        num_superpixels=5000,       # можно подбирать под размер изображения
        random_init=True,
        poisson_radius_factor=1.0   # влияет на минимальное расстояние между центрами
    )
    print(f"Финальное количество центров: {len(centers)}")

    # Визуализация границ
    img_with_boundaries = draw_superpixel_boundaries(imgcol, labels)
    img_with_vectors = draw_gradient_vectors(imgcol, labels, centers, u, v)

    fig_quiver = draw_gradient_vectors_quiver(imgcol, labels, centers, u, v, scale=0.5)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # Основная визуализация
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].imshow(imgcol)
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title('Исходное изображение')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.arctan2(-v, u), cmap='hsv')   # направления градиентов
    axes[0, 1].set_title('Направления градиентов')
    axes[0, 1].axis('off')
    axes[0, 1].invert_yaxis()

    axes[1, 0].imshow(img_with_boundaries)
    axes[1, 0].set_title('Границы суперпикселей (Вороной)')
    axes[1, 0].axis('off')
    axes[1, 0].invert_yaxis()

    axes[1, 1].imshow(img_with_vectors)
    axes[1, 1].set_title('Векторы градиента в центрах')
    axes[1, 1].axis('off')
    axes[1, 1].invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Сегментированное изображение (каждый суперпиксель – средний цвет)
    segmented_img = np.zeros_like(imgcol)
    for i in range(np.max(labels) + 1):
        mask = labels == i
        if np.any(mask):
            segmented_img[mask] = np.mean(imgcol[mask], axis=0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(segmented_img)
    plt.gca().invert_yaxis()
    plt.title('Сегментированное изображение (Вороной)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(draw_superpixel_boundaries(segmented_img, labels))
    plt.gca().invert_yaxis()
    plt.title('Сегментация с границами')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Сохранение JSON
    print("Сохранение данных суперпикселей...")
    full_data = save_superpixels_to_json(
        labels, centers, imgcol,
        f'superpixels_voronoi_{filename.split(".")[0]}.json'
    )
    print(f"Сохранено {len(full_data['superpixels'])} суперпикселей")


if __name__ == "__main__":
    main()