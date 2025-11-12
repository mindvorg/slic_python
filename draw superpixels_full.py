import json
import matplotlib.pyplot as plt

# Загрузка данных из JSON файла
with open('superpixels_full_4.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Получение размеров изображения
height = data['image_dimensions']['height']
width = data['image_dimensions']['width']
superpixels = data['superpixels']

# Создание фигуры
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.invert_yaxis()

# Отрисовка boundary_points и центров с ID
for superpixel in superpixels:
    # Отрисовка boundary_points красным цветом
    if 'boundary_points' in superpixel and superpixel['boundary_points']:
        boundary_points = superpixel['boundary_points']
        x_coords = [point['x'] for point in boundary_points]
        y_coords = [point['y'] for point in boundary_points]
        ax.scatter(x_coords, y_coords, c='red', s=2, alpha=0.8)

    # Отрисовка центра и ID
    center_x = superpixel['center']['x']
    center_y = superpixel['center']['y']
    sp_id = superpixel['id']

    # Синяя точка в центре
    ax.plot(center_x, center_y, 'bo', markersize=3)

    # Текст с ID
    ax.text(center_x, center_y, str(sp_id),
            fontsize=6, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))

# Настройка внешнего вида
ax.set_title('Boundary_points (красные) и центры суперпикселей с ID', fontsize=14)
ax.set_xlabel('X координата')
ax.set_ylabel('Y координата')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()