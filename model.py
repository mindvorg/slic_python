import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate


def create_hexagonal_superpixel(center, size=30):
    """
    Создает шестиугольный суперпиксель
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 углов
    hexagon = np.column_stack([
        center[0] + size * np.cos(angles),
        center[1] + size * np.sin(angles)
    ])
    return hexagon


def create_stroke_along_edges(hexagon, start_edge, num_edges=4, stroke_width=8):
    """
    Создает мазок, где один сплайн идет вдоль затронутых ребер от начала до конца,
    а второй - параллельно ему внутри шестиугольника
    """
    # Определяем затронутые ребра
    edges = [(start_edge + i) % 6 for i in range(num_edges)]

    # Центр шестиугольника
    hex_center = np.mean(hexagon, axis=0)

    # Создаем точки для внешнего сплайна (точно по вершинам ребер)
    outer_points = []

    # Начинаем с начала первого ребра
    outer_points.append(hexagon[edges[0]])

    # Добавляем все промежуточные вершины
    for i in range(1, num_edges):
        edge_idx = edges[i]
        outer_points.append(hexagon[edge_idx])

    # Заканчиваем концом последнего ребра
    outer_points.append(hexagon[(edges[-1] + 1) % 6])

    outer_points = np.array(outer_points)

    # Создаем внутренний сплайн (параллельный внешнему, но смещенный внутрь)
    inner_points = []

    for i, outer_point in enumerate(outer_points):
        # Для каждой точки внешнего сплайна находим направление смещения
        if i == 0:
            # Первая точка - смещение от первого ребра
            edge_vector = hexagon[(edges[0] + 1) % 6] - hexagon[edges[0]]
            normal = np.array([-edge_vector[1], edge_vector[0]])
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0, 0])
            # Направляем нормаль внутрь
            to_center = hex_center - outer_point
            if np.dot(normal, to_center) < 0:
                normal = -normal
        elif i == len(outer_points) - 1:
            # Последняя точка - смещение от последнего ребра
            edge_vector = hexagon[(edges[-1] + 1) % 6] - hexagon[edges[-1]]
            normal = np.array([-edge_vector[1], edge_vector[0]])
            normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) > 0 else np.array([0, 0])
            # Направляем нормаль внутрь
            to_center = hex_center - outer_point
            if np.dot(normal, to_center) < 0:
                normal = -normal
        else:
            # Промежуточные точки - смещение к центру
            to_center = hex_center - outer_point
            normal = to_center / np.linalg.norm(to_center) if np.linalg.norm(to_center) > 0 else np.array([0, 0])

        # Смещаем точку внутрь
        inner_point = outer_point + normal * stroke_width
        inner_points.append(inner_point)

    inner_points = np.array(inner_points)

    # Создаем сплайны для плавных кривых
    t_outer = np.linspace(0, 1, len(outer_points))
    t_inner = np.linspace(0, 1, len(inner_points))

    # Внешний сплайн (вдоль ребер от начала до конца)
    outer_spline_x = interpolate.CubicSpline(t_outer, outer_points[:, 0])
    outer_spline_y = interpolate.CubicSpline(t_outer, outer_points[:, 1])

    # Внутренний сплайн (параллельный)
    inner_spline_x = interpolate.CubicSpline(t_inner, inner_points[:, 0])
    inner_spline_y = interpolate.CubicSpline(t_inner, inner_points[:, 1])

    # Генерируем плотные точки для визуализации
    t_dense = np.linspace(0, 1, 50)

    outer_curve = np.column_stack([outer_spline_x(t_dense), outer_spline_y(t_dense)])
    inner_curve = np.column_stack([inner_spline_x(t_dense), inner_spline_y(t_dense)])

    return (outer_spline_x, outer_spline_y), (inner_spline_x, inner_spline_y), outer_curve, inner_curve, outer_points


def create_stroke_polygon(outer_spline, inner_spline, num_points=50):
    """
    Создает полигон мазка из двух сплайнов
    """
    t_samples = np.linspace(0, 1, num_points)

    outer_points = np.column_stack([outer_spline[0](t_samples), outer_spline[1](t_samples)])
    inner_points = np.column_stack([inner_spline[0](t_samples), inner_spline[1](t_samples)])

    # Создаем полигон мазка (соединяем внешний и внутренний сплайны)
    stroke_polygon = np.vstack([outer_points, inner_points[::-1]])

    return stroke_polygon, outer_points, inner_points


def visualize_stroke_along_edges():
    """
    Визуализирует мазок, где один сплайн идет вдоль ребер от начала до конца
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    center = [50, 50]
    hexagon = create_hexagonal_superpixel(center, size=25)

    for start_edge in range(3):  # Покажем 3 варианта
        ax = axes[start_edge]
        ax2 = axes[start_edge + 3]

        # Рисуем исходный шестиугольник
        hex_patch = plt.Polygon(hexagon, alpha=0.3, color='lightblue', label='Исходный сегмент')
        ax.add_patch(hex_patch)
        ax.plot(hexagon[:, 0], hexagon[:, 1], 'bo-', linewidth=1, markersize=4)

        # Подписываем вершины
        for i in range(6):
            ax.text(hexagon[i, 0], hexagon[i, 1], f'V{i}',
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))

        # Создаем мазок вдоль ребер
        outer_spline, inner_spline, outer_curve, inner_curve, outer_vertices = create_stroke_along_edges(
            hexagon, start_edge, num_edges=4, stroke_width=10
        )

        # Создаем полигон мазка
        stroke_polygon, outer_points, inner_points = create_stroke_polygon(
            outer_spline, inner_spline, 50
        )

        # Рисуем мазок
        stroke_patch = plt.Polygon(stroke_polygon, alpha=0.7, color='red', label='Мазок кисти')
        ax.add_patch(stroke_patch)

        # Подсвечиваем затронутые ребра
        affected_edges = [(start_edge + i) % 6 for i in range(4)]
        for edge_idx in affected_edges:
            edge_start = hexagon[edge_idx]
            edge_end = hexagon[(edge_idx + 1) % 6]
            ax.plot([edge_start[0], edge_end[0]], [edge_start[1], edge_end[1]],
                    'g-', linewidth=3, alpha=0.7, label='Затронутые ребра' if edge_idx == affected_edges[0] else "")

        # Показываем начальную и конечную точки сплайнов
        ax.plot(outer_vertices[0, 0], outer_vertices[0, 1], 'ro', markersize=8, label='Начало сплайна')
        ax.plot(outer_vertices[-1, 0], outer_vertices[-1, 1], 'go', markersize=8, label='Конец сплайна')

        ax.set_aspect('equal')
        ax.set_title(f'Мазок от V{affected_edges[0]} до V{(affected_edges[-1] + 1) % 6}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Детальная визуализация
        hex_patch2 = plt.Polygon(hexagon, alpha=0.2, color='lightblue')
        ax2.add_patch(hex_patch2)
        ax2.plot(hexagon[:, 0], hexagon[:, 1], 'bo-', linewidth=1, markersize=4)

        # Рисуем отдельные сплайны
        ax2.plot(outer_points[:, 0], outer_points[:, 1], 'b-', linewidth=3,
                 label='Внешний сплайн (вдоль ребер)')
        ax2.plot(inner_points[:, 0], inner_points[:, 1], 'g-', linewidth=3,
                 label='Внутренний сплайн')

        # Показываем контрольные точки сплайнов
        ax2.plot(outer_vertices[:, 0], outer_vertices[:, 1], 'bo', markersize=6,
                 label='Контрольные точки')

        # Рисуем мазок полупрозрачным
        stroke_patch2 = plt.Polygon(stroke_polygon, alpha=0.5, color='red')
        ax2.add_patch(stroke_patch2)

        # Подписываем начальную и конечную точки
        ax2.text(outer_vertices[0, 0], outer_vertices[0, 1], 'START',
                 ha='right', va='bottom', fontweight='bold', color='blue')
        ax2.text(outer_vertices[-1, 0], outer_vertices[-1, 1], 'END',
                 ha='left', va='top', fontweight='bold', color='blue')

        ax2.set_aspect('equal')
        ax2.set_title(f'Детали сплайнов: V{affected_edges[0]} → V{(affected_edges[-1] + 1) % 6}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def demonstrate_edge_to_edge_strokes():
    """
    Демонстрирует мазки от начала до конца ребер для разных конфигураций
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    center = [50, 50]
    hexagon = create_hexagonal_superpixel(center, size=25)

    configurations = [
        (0, 3, "От V0 до V3"),  # Через 3 ребра
        (1, 4, "От V1 до V4"),  # Через 3 ребра
        (2, 5, "От V2 до V5"),  # Через 3 ребра
        (0, 4, "От V0 до V4")  # Через 4 ребра
    ]

    for idx, (start_edge, num_edges, title) in enumerate(configurations):
        ax = axes[idx]

        # Рисуем шестиугольник
        hex_patch = plt.Polygon(hexagon, alpha=0.2, color='lightblue')
        ax.add_patch(hex_patch)
        ax.plot(hexagon[:, 0], hexagon[:, 1], 'ko-', linewidth=1, markersize=3)

        # Подписываем вершины
        for i in range(6):
            ax.text(hexagon[i, 0], hexagon[i, 1], f'V{i}',
                    ha='center', va='center', fontsize=8, fontweight='bold')

        # Создаем мазок
        outer_spline, inner_spline, outer_curve, inner_curve, outer_vertices = create_stroke_along_edges(
            hexagon, start_edge, num_edges=num_edges, stroke_width=8
        )

        stroke_polygon, outer_points, inner_points = create_stroke_polygon(
            outer_spline, inner_spline, 50
        )

        # Рисуем мазок
        stroke_patch = plt.Polygon(stroke_polygon, alpha=0.8, color='coral')
        ax.add_patch(stroke_patch)

        # Рисуем сплайны
        ax.plot(outer_points[:, 0], outer_points[:, 1], 'b-', linewidth=2)
        ax.plot(inner_points[:, 0], inner_points[:, 1], 'g-', linewidth=2)

        # Подсвечиваем начальную и конечную точки
        ax.plot(outer_vertices[0, 0], outer_vertices[0, 1], 'ro', markersize=8)
        ax.plot(outer_vertices[-1, 0], outer_vertices[-1, 1], 'go', markersize=8)

        ax.text(outer_vertices[0, 0], outer_vertices[0, 1], 'START',
                ha='right', va='bottom', fontweight='bold')
        ax.text(outer_vertices[-1, 0], outer_vertices[-1, 1], 'END',
                ha='left', va='top', fontweight='bold')

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Запуск демонстрации
if __name__ == "__main__":
    print("Демонстрация мазков от начала до конца ребер...")

    # Визуализация мазков вдоль ребер
    visualize_stroke_along_edges()

    # Демонстрация разных конфигураций
    demonstrate_edge_to_edge_strokes()