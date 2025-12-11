import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import cv2
from model import create_hexagonal_superpixel, create_stroke_along_edges, create_stroke_polygon
import matplotlib.patches as patches
from scipy.spatial import Voronoi, ConvexHull
from sklearn.cluster import KMeans
import numpy as np
from slic_modif import slic_modif, draw_gradient_vectors, draw_superpixel_boundaries, draw_gradient_vectors_quiver

#filename = '4.jpg'
filename = 'Lenna256.jpg'

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
imgcol = plt.imread(filename)

# differentiation kernels
kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
kernelx = -np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# apply differentiation
u = cv2.filter2D(img, cv2.CV_8U, kernelx).astype(float)
v = cv2.filter2D(img, cv2.CV_8U, kernely).astype(float)

# filter via averaging
nsize = 24
s = 6
ax = np.linspace(-(nsize - 1) / 2., (nsize - 1) / 2., nsize)
gauss = np.exp(-0.5 * np.square(ax) / np.square(s))
kernel = np.outer(gauss, gauss)
h = kernel / np.sum(kernel)
u = sig.convolve2d(u, h, mode="same")
v = sig.convolve2d(v, h, mode="same")

# get only directions
scl = np.sqrt(np.power(u, 2) + np.power(v, 2))  # scaling factor
scl = (scl == 0) + scl
u = np.divide(u, scl)
v = np.divide(v, scl)

N = 10  # downsampling, 20 for ball
s = 1  # scaling factor for arrows
ud = u[::N, ::N]
vd = v[::N, ::N]

imgcol = np.flipud(imgcol)

h, w = ud.shape
fig, ax = plt.subplots()
ax.imshow(imgcol, extent=[0, w, 0, h])

# ax.quiver(ud, vd, headwidth = 3*s, headlength = 5*s, headaxislength = 4*s, color = 'red',scale = 50)
ax.quiver(ud, vd, color='red')
# ax.quiver(u, v)
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.show()

fig.savefig('UPD_quiver_'+filename, transparent=True, dpi=300)

fig, ax = plt.subplots()
a = -vd
b = ud
ax.imshow(imgcol, extent=[0, w, 0, h])
# ax.quiver(-ud, vd, headwidth = 3*s, headlength = 5*s, headaxislength = 4*s, color = 'blue', scale = 50)
ax.quiver(a, b, color='blue')
plt.gca().set_aspect('equal', adjustable='box')
plt.gca().invert_yaxis()
plt.xticks([])
plt.yticks([])
plt.show()
fig.savefig('UPD_quiver2_' + filename, transparent=True, dpi=300)

a = -v[::-1, :]
b = u[::-1, :]
magnitude, angle = cv2.cartToPolar(a, b)

# Применяем модифицированный SLIC с распределением Пуассона от SciPy
labels, centers = slic_modif(
    imgcol, u, v,
    p=3.0,
    num_superpixels=1400,
    compactness=30,
    max_iterations=50,
    poisson_radius_factor=1  # Меньше = плотнее распределение
)


# модель берет отрезки, шестиугольники, и превращает в мазки
print(f"Финальное количество центров: {len(centers)}")
# Создаем визуализации
img_with_boundaries = draw_superpixel_boundaries(imgcol, labels)
img_with_vectors = draw_gradient_vectors(imgcol, labels, centers, u, v)

fig_quiver = draw_gradient_vectors_quiver(imgcol, labels, centers, u, v, scale=0.5)
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Отображаем результаты
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Исходное изображение
axes[0, 0].imshow(imgcol)
axes[0, 0].invert_yaxis()
axes[0, 0].set_title('Исходное изображение')
axes[0, 0].axis('off')

# Направления градиентов
axes[0, 1].imshow(angle, cmap='hsv')
axes[0, 1].set_title('Направления градиентов')
axes[0, 1].axis('off')
axes[0, 1].invert_yaxis()

# Изображение с границами суперпикселей
axes[1, 0].imshow(img_with_boundaries)
axes[1, 0].set_title('Границы суперпикселей')
axes[1, 0].axis('off')
axes[1, 0].invert_yaxis()

# Изображение с векторами градиента
axes[1, 1].imshow(img_with_vectors)
axes[1, 1].set_title('Векторы градиента в центроидах')
axes[1, 1].axis('off')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()

# Дополнительная визуализация: сегментированное изображение
segmented_img = np.zeros_like(imgcol)
for i in range(np.max(labels) + 1):
    mask = labels == i
    if np.any(mask):
        segmented_img[mask] = np.mean(imgcol[mask], axis=0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(segmented_img)
plt.gca().invert_yaxis()
plt.title('Сегментированное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(draw_superpixel_boundaries(segmented_img, labels))
plt.gca().invert_yaxis()
plt.title('Сегментация с границами')
plt.axis('off')

plt.tight_layout()
plt.show()

# Сохранение в json
def numpy_to_python(obj):
    """
    Преобразует NumPy типы в стандартные Python типы для JSON сериализации
    """
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

def save_superpixels_to_json(labels, centers, filename='superpixels_data.json'):
    """
    Сохраняет данные о суперпикселях в JSON файл

    Parameters:
    labels - матрица меток суперпикселей
    centers - центры суперпикселей
    filename - имя файла для сохранения
    """

    height, width = labels.shape
    superpixels_data = {
        'image_dimensions': {
            'height': int(height),
            'width': int(width)
        },
        'num_superpixels': len(centers),
        'superpixels': []
    }
    neighbors = find_superpixel_neighbors(labels)

    # Для каждого суперпикселя собираем координаты граничных точек
    for k in range(len(centers)):
        mask = labels == k
        if not np.any(mask):
            continue

        # Находим граничные точки суперпикселя
        boundaries = np.zeros((height, width), dtype=np.uint8)

        # Проверяем границы по вертикали
        vertical_boundaries = labels[:-1, :] != labels[1:, :]
        boundaries[:-1, :] = np.logical_or(boundaries[:-1, :], vertical_boundaries)
        boundaries[1:, :] = np.logical_or(boundaries[1:, :], vertical_boundaries)

        # Проверяем границы по горизонтали
        horizontal_boundaries = labels[:, :-1] != labels[:, 1:]
        boundaries[:, :-1] = np.logical_or(boundaries[:, :-1], horizontal_boundaries)
        boundaries[:, 1:] = np.logical_or(boundaries[:, 1:], horizontal_boundaries)

        # Получаем координаты граничных точек для этого суперпикселя
        boundary_mask = np.logical_and(boundaries, mask)
        boundary_y, boundary_x = np.where(boundary_mask)

        # Получаем все точки суперпикселя (опционально)
        all_y, all_x = np.where(mask)

        # Преобразуем координаты в стандартные Python типы
        boundary_points = [{'x': int(x), 'y': int(y)} for y, x in zip(boundary_y, boundary_x)]
        all_points = [{'x': int(x), 'y': int(y)} for y, x in zip(all_y, all_x)]

        # Собираем данные суперпикселя
        superpixel_data = {
            'id': int(k),
            'center': {
                'y': float(centers[k, 0]),
                'x': float(centers[k, 1])
            },
            'boundary_points': boundary_points,
            'all_points': all_points,
            'area': int(len(all_y)),
            'neighbors': sorted(list(neighbors[k]))  # ДОБАВЬ ЭТУ СТРОКУ
        }

        superpixels_data['superpixels'].append(superpixel_data)

    # Сохраняем в JSON файл с использованием кастомного обработчика
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(superpixels_data, f, indent=2, ensure_ascii=False, default=numpy_to_python)

    print(f"Данные суперпикселей сохранены в {filename}")
    return superpixels_data


def save_superpixels_compact(labels, centers, filename='superpixels_compact.json'):
    """
    Сохраняет компактные данные о суперпикселях (только граничные точки)
    """

    height, width = labels.shape
    superpixels_data = {
        'image_dimensions': {
            'height': int(height),
            'width': int(width)
        },
        'num_superpixels': int(len(centers)),
        'superpixels': []
    }

    for k in range(len(centers)):
        mask = labels == k
        if not np.any(mask):
            continue

        # Находим граничные точки (упрощенный метод)
        from scipy import ndimage

        # Вычисляем границы с помощью морфологических операций
        structure = ndimage.generate_binary_structure(2, 2)
        eroded = ndimage.binary_erosion(mask, structure)
        boundaries = mask & ~eroded

        boundary_y, boundary_x = np.where(boundaries)

        # Преобразуем координаты в стандартные Python типы
        boundary_points = [{'x': int(x), 'y': int(y)} for y, x in zip(boundary_y, boundary_x)]

        superpixel_data = {
            'id': int(k),
            'center': {
                'y': float(centers[k, 0]),
                'x': float(centers[k, 1])
            },
            'boundary_points': boundary_points,
            'area': int(np.sum(mask))
        }

        superpixels_data['superpixels'].append(superpixel_data)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(superpixels_data, f, indent=2, ensure_ascii=False, default=numpy_to_python)

    print(f"Компактные данные суперпикселей сохранены в {filename}")
    return superpixels_data


# ВЫЗЫВАЕМ ФУНКЦИИ СОХРАНЕНИЯ ПОСЛЕ ОСНОВНОЙ ОБРАБОТКИ

# После получения labels и centers из slic_modif добавляем:
print("Сохранение данных суперпикселей...")

# Полная версия с всеми точками
full_data = save_superpixels_to_json(labels, centers, f'superpixels_full_{filename.split(".")[0]}.json')

# Компактная версия только с граничными точками
# compact_data = save_superpixels_compact(labels, centers, f'superpixels_compact_{filename.split(".")[0]}.json')

print(f"Сохранено {len(full_data['superpixels'])} суперпикселей")

def save_superpixels_simple(labels, centers, filename='superpixels_simple.json'):
    """
    Упрощенный метод сохранения - только основные данные
    """
    height, width = labels.shape

    # Создаем упрощенную структуру данных
    simple_data = {
        'image_dimensions': {'height': int(height), 'width': int(width)},
        'num_superpixels': int(len(centers)),
        'centers': [],
        'superpixel_areas': []
    }

    # Сохраняем центры
    for k, center in enumerate(centers):
        simple_data['centers'].append({
            'id': int(k),
            'y': float(center[0]),
            'x': float(center[1])
        })

    # Сохраняем площади суперпикселей
    for k in range(len(centers)):
        mask = labels == k
        area = int(np.sum(mask)) if np.any(mask) else 0
        simple_data['superpixel_areas'].append({
            'id': int(k),
            'area': area
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False, default=numpy_to_python)

    print(f"Упрощенные данные сохранены в {filename}")
    return simple_data