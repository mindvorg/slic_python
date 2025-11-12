import json
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os


def validate_and_fix_json(filename: str) -> dict:
    """Пытается исправить и загрузить JSON файл."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Попытка загрузить как есть
        return json.loads(content)

    except json.JSONDecodeError as e:
        print(f"Ошибка в JSON файле: {e}")
        print("Пытаемся исправить автоматически...")

        # Простые исправления распространенных ошибок
        content = content.replace(',asd', '')  # Удаляем ,asd
        content = content.replace(', asd', '')  # Удаляем , asd
        content = content.replace('asd', '')  # Удаляем asd

        # Удаляем лишние запятые в конце массивов и объектов
        lines = content.split('\n')
        fixed_lines = []
        for line in lines:
            # Удаляем запятые перед закрывающими скобками
            line = line.replace(',}', '}')
            line = line.replace(',]', ']')
            fixed_lines.append(line)

        content = '\n'.join(fixed_lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError as e2:
            print(f"Не удалось исправить JSON: {e2}")
            raise


def sort_points_by_angle(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Сортирует точки по углу относительно центра масс для создания правильного контура."""
    if len(points) <= 1:
        return points

    # Находим центр масс
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)

    # Сортируем точки по углу относительно центра
    def angle_from_center(point):
        return math.atan2(point[1] - center_y, point[0] - center_x)

    return sorted(points, key=angle_from_center)


def perpendicular_distance(point: Tuple[float, float], line_start: Tuple[float, float],
                           line_end: Tuple[float, float]) -> float:
    """Вычисляет перпендикулярное расстояние от точки до линии."""
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    if x1 == x2 and y1 == y2:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

    numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return numerator / denominator


def ramer_douglas_peucker(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """Упрощает кривую с помощью алгоритма Рамера-Дугласа-Пекера."""
    if len(points) <= 2:
        return points

    dmax = 0.0
    index = 0
    start, end = points[0], points[-1]

    for i in range(1, len(points) - 1):
        d = perpendicular_distance(points[i], start, end)
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left = ramer_douglas_peucker(points[:index + 1], epsilon)
        right = ramer_douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def find_optimal_epsilon(points: List[Tuple[float, float]], target_vertices: int) -> float:
    """Находит оптимальное значение epsilon для получения target_vertices вершин."""
    epsilon_min = 0.0
    epsilon_max = max(math.hypot(points[-1][0] - points[0][0], points[-1][1] - points[0][1]), 1.0)

    for _ in range(50):  # Бинарный поиск
        epsilon = (epsilon_min + epsilon_max) / 2
        simplified = ramer_douglas_peucker(points, epsilon)

        if len(simplified) <= target_vertices:
            epsilon_max = epsilon
        else:
            epsilon_min = epsilon

    return (epsilon_min + epsilon_max) / 2


def ensure_five_vertices(points: List[Tuple[float, float]], original_points: List[Tuple[float, float]]) -> List[
    Tuple[float, float]]:
    """Гарантирует, что в результате будет ровно 5 вершин."""
    if len(points) == 5:
        return points

    # Если вершин меньше 5, добавляем важные точки из оригинала
    if len(points) < 5:
        # Находим точки с наибольшим отклонением от упрощенного контура
        important_points = []

        # Создаем замкнутый контур из упрощенных точек
        closed_simplified = points + [points[0]]

        # Для каждой точки оригинала вычисляем минимальное расстояние до упрощенного контура
        distances = []
        for orig_point in original_points:
            min_dist = float('inf')
            for i in range(len(closed_simplified) - 1):
                dist = perpendicular_distance(orig_point, closed_simplified[i], closed_simplified[i + 1])
                if dist < min_dist:
                    min_dist = dist
            distances.append((orig_point, min_dist))

        # Сортируем по расстоянию (наибольшее расстояние = наиболее важная точка)
        distances.sort(key=lambda x: x[1], reverse=True)

        # Добавляем наиболее важные точки, пока не получим 5 вершин
        result = points.copy()
        for point, dist in distances:
            if len(result) >= 5:
                break
            if point not in result:
                result.append(point)

        return result

    # Если вершин больше 5, удаляем наименее важные
    if len(points) > 5:
        # Вычисляем важность каждой точки как минимальное расстояние до линии между соседями
        importance = []
        closed_points = points + [points[0], points[1]]  # Добавляем для удобства вычислений

        for i in range(len(points)):
            prev = closed_points[i]
            curr = closed_points[i + 1]
            next = closed_points[i + 2]

            # Важность точки - расстояние до линии между соседями
            imp = perpendicular_distance(curr, prev, next)
            importance.append((curr, imp))

        # Сортируем по важности (наименее важные сначала)
        importance.sort(key=lambda x: x[1])

        # Удаляем наименее важные точки, пока не останется 5
        result = points.copy()
        for point, imp in importance:
            if len(result) <= 5:
                break
            if point in result:
                result.remove(point)

        return result


def simplify_to_pentagon(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Упрощает фигуру до пятиугольника."""
    # Сортируем точки для создания правильного контура
    sorted_points = sort_points_by_angle(points)

    # Замыкаем фигуру
    if sorted_points[0] != sorted_points[-1]:
        closed_points = sorted_points + [sorted_points[0]]
    else:
        closed_points = sorted_points

    # Находим оптимальный epsilon для 5 вершин
    epsilon = find_optimal_epsilon(closed_points, 5)
    simplified = ramer_douglas_peucker(closed_points, epsilon)

    # Гарантируем ровно 5 вершин
    final_points = ensure_five_vertices(simplified, sorted_points)

    return final_points


def process_json_file(input_file: str, output_file: str, save_plots: bool = True):
    """Обрабатывает JSON файл и сохраняет результаты."""

    # Читаем и валидируем JSON файл
    data = validate_and_fix_json(input_file)

    results = []

    # Обрабатываем каждый суперпиксель
    for superpixel in data['superpixels']:
        sp_id = superpixel['id']
        center = superpixel['center']
        boundary_points = superpixel['boundary_points']

        # Преобразуем boundary_points в список кортежей
        points = [(p['x'], p['y']) for p in boundary_points]

        # Упрощаем до пятиугольника
        simplified_vertices = simplify_to_pentagon(points)

        # Сохраняем результат
        result = {
            'id': sp_id,
            'center': center,
            'vertices': [{'x': float(v[0]), 'y': float(v[1])} for v in simplified_vertices]
        }
        results.append(result)

        # Создаем визуализацию если нужно
        if save_plots:
            plot_superpixel(points, simplified_vertices, sp_id, center)

    # Сохраняем результаты в JSON файл
    output_data = {
        'image_dimensions': data['image_dimensions'],
        'num_superpixels': data['num_superpixels'],
        'processed_superpixels': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Обработано {len(results)} суперпикселей")
    print(f"Результаты сохранены в {output_file}")


def plot_superpixel(original_points: List[Tuple[float, float]],
                    simplified_points: List[Tuple[float, float]],
                    sp_id: int,
                    center: Dict):
    """Создает визуализацию для одного суперпикселя."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Исходная фигура
    orig_x = [p[0] for p in original_points]
    orig_y = [p[1] for p in original_points]

    # Замыкаем для отрисовки
    if original_points[0] != original_points[-1]:
        orig_x.append(original_points[0][0])
        orig_y.append(original_points[0][1])

    ax1.plot(orig_x, orig_y, 'b-', linewidth=2, label='Исходная граница')
    ax1.scatter(orig_x, orig_y, c='blue', s=20, alpha=0.6)
    ax1.scatter([center['x']], [center['y']], c='red', s=50, marker='x', label='Центр')
    ax1.set_title(f'Суперпиксель {sp_id} - Исходная форма\n({len(original_points)} точек)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Упрощенная фигура
    simp_x = [p[0] for p in simplified_points]
    simp_y = [p[1] for p in simplified_points]

    # Замыкаем упрощенную фигуру для отрисовки
    if simplified_points[0] != simplified_points[-1]:
        simp_x.append(simplified_points[0][0])
        simp_y.append(simplified_points[0][1])

    ax2.plot(simp_x, simp_y, 'r-', linewidth=2, label='Упрощенная граница')
    ax2.scatter(simp_x, simp_y, c='red', s=50, label='Вершины')
    ax2.scatter([center['x']], [center['y']], c='red', s=50, marker='x', label='Центр')
    ax2.set_title(f'Суперпиксель {sp_id} - Упрощенный пятиугольник\n({len(simplified_points)} вершин)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Создаем папку для графиков если её нет
    os.makedirs('superpixel_plots', exist_ok=True)
    plt.savefig(f'superpixel_plots/superpixel_{sp_id}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Создана визуализация для суперпикселя {sp_id}")


def plot_all_superpixels_from_json(json_file: str, output_image: str = "all_superpixels.png"):
    """Отрисовывает все суперпиксели из JSON файла на одном изображении."""

    # Читаем JSON файл
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Обрабатываем каждый суперпиксель
    for superpixel in data['processed_superpixels']:
        sp_id = superpixel['id']
        center = superpixel['center']
        vertices = superpixel['vertices']

        # Извлекаем координаты вершин
        vert_x = [v['x'] for v in vertices]
        vert_y = [v['y'] for v in vertices]

        # Замыкаем полигон
        if vertices[0] != vertices[-1]:
            vert_x.append(vertices[0]['x'])
            vert_y.append(vertices[0]['y'])

        # Рисуем полигон
        ax.plot(vert_x, vert_y, 'b-', linewidth=1, alpha=0.7)
        ax.fill(vert_x, vert_y, alpha=0.2)

        # Рисуем вершины
        ax.scatter(vert_x, vert_y, c='red', s=20, alpha=0.8)

        # Добавляем ID в центре
        ax.text(center['x'], center['y'], str(sp_id),
                fontsize=8, ha='center', va='center')

    ax.set_title(f'Все суперпиксели ({len(data["processed_superpixels"])} шт.)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Визуализация всех суперпикселей сохранена в {output_image}")


def plot_all_superpixels_original_from_json(json_file: str, output_image: str = "all_original_superpixels.png"):
    """Отрисовывает все исходные суперпиксели из JSON файла на одном изображении."""

    # Читаем JSON файл
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Обрабатываем каждый суперпиксель
    for superpixel in data['superpixels']:
        sp_id = superpixel['id']
        center = superpixel['center']
        boundary_points = superpixel['boundary_points']

        # Извлекаем координаты граничных точек
        bp_x = [p['x'] for p in boundary_points]
        bp_y = [p['y'] for p in boundary_points]

        # Замыкаем полигон
        if boundary_points[0] != boundary_points[-1]:
            bp_x.append(boundary_points[0]['x'])
            bp_y.append(boundary_points[0]['y'])

        # Рисуем полигон
        ax.plot(bp_x, bp_y, 'b-', linewidth=0.5, alpha=0.5)
        ax.fill(bp_x, bp_y, alpha=0.1)

        # Добавляем ID в центре
        ax.text(center['x'], center['y'], str(sp_id),
                fontsize=6, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', alpha=0.7))

    ax.set_title(f'Все исходные суперпиксели ({len(data["superpixels"])} шт.)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Визуализация всех исходных суперпикселей сохранена в {output_image}")
def main():
    """Основная функция для запуска скрипта."""

    # Настройки
    input_json_file = 'superpixels_full_4.json'  # Замените на путь к вашему JSON файлу
    output_json_file = 'processed_superpixels.json'

    # Проверяем существование входного файла
    if not os.path.exists(input_json_file):
        print(f"Файл {input_json_file} не найден!")
        return

    # Обрабатываем файл
    try:
        process_json_file(input_json_file, output_json_file, save_plots=True)
        print("Обработка завершена успешно!")
        # Отрисовываем все суперпиксели из обработанного файла

    except Exception as e:
        print(f"Ошибка при обработке файла: {e}")

plot_all_superpixels_from_json('processed_superpixels.json')
# if __name__ == "__main__":
#     main()