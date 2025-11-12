import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9

    def distance(self, other) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"({self.x:.2f}, {self.y:.2f})"


class ShapeRecognizer:
    def __init__(self, angle_threshold: float = 0.4, distance_threshold: float = 5.0):
        self.angle_threshold = angle_threshold
        self.distance_threshold = distance_threshold
        self.angle_history = []  # Для хранения истории углов
        self.center = None  # Центр фигуры
        self.polar_data = []  # Данные в полярных координатах

    def recognize_shape(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Распознает вершины n-угольника из списка точек

        Args:
            points: Список координат точек [(x1, y1), (x2, y2), ...]

        Returns:
            Список вершин многоугольника [(x1, y1), (x2, y2), ...]
        """
        # Преобразуем в объекты Point
        point_objects = [Point(x, y) for x, y in points]

        # Находим центр фигуры
        self.center = self._calculate_center(point_objects)

        # Вычисляем полярные координаты для ВСЕХ исходных точек
        self._calculate_polar_coordinates_all_points(point_objects)

        # Находим внешний контур (выпуклую оболочку)
        contour_points = self._find_contour(point_objects)

        # Упрощаем контур
        simplified_points = self._reduce_points(contour_points)

        # Находим вершины
        vertices = self._find_vertices(simplified_points)

        return [(p.x, p.y) for p in vertices]

    def _calculate_center(self, points: List[Point]) -> Point:
        """Вычисляет центр масс фигуры"""
        if not points:
            return Point(0, 0)

        avg_x = sum(p.x for p in points) / len(points)
        avg_y = sum(p.y for p in points) / len(points)

        return Point(avg_x, avg_y)

    def _calculate_polar_coordinates_all_points(self, points: List[Point]):
        """Вычисляет полярные координаты для ВСЕХ исходных точек относительно центра"""
        self.polar_data = []

        for point in points:
            # Вектор от центра к точке
            dx = point.x - self.center.x
            dy = point.y - self.center.y

            # Угол в полярных координатах (от -π до π)
            angle = math.atan2(dy, dx)
            # Нормализуем угол от 0 до 2π
            if angle < 0:
                angle += 2 * math.pi

            # Расстояние от центра
            distance = math.sqrt(dx ** 2 + dy ** 2)

            self.polar_data.append((angle, distance, point))

    def _find_contour(self, points: List[Point]) -> List[Point]:
        """Находит внешний контур точек с помощью выпуклой оболочки"""
        if len(points) <= 2:
            return points

        # Используем выпуклую оболочку для нахождения внешнего контура
        points_array = np.array([(p.x, p.y) for p in points])

        try:
            hull = ConvexHull(points_array)
            # Получаем точки выпуклой оболочки в правильном порядке
            hull_points = [Point(points_array[vertex, 0], points_array[vertex, 1])
                           for vertex in hull.vertices]
            # Замыкаем контур
            hull_points.append(hull_points[0])
            return hull_points
        except:
            # Если выпуклая оболочка не работает, используем минимальную ограничивающую рамку
            return self._find_bounding_box(points)

    def _find_bounding_box(self, points: List[Point]) -> List[Point]:
        """Находит ограничивающую рамку как запасной вариант"""
        if not points:
            return []

        min_x = min(p.x for p in points)
        max_x = max(p.x for p in points)
        min_y = min(p.y for p in points)
        max_y = max(p.y for p in points)

        return [
            Point(min_x, min_y),
            Point(max_x, min_y),
            Point(max_x, max_y),
            Point(min_x, max_y),
            Point(min_x, min_y)  # Замыкаем
        ]

    def _reduce_points(self, points: List[Point]) -> List[Point]:
        """Алгоритм Рамера-Дугласа-Пекера для упрощения контура"""
        if len(points) <= 2:
            return points

        def douglas_pecker(points, epsilon):
            if len(points) <= 2:
                return points

            # Находим точку с максимальным расстоянием
            dmax = 0
            index = 0
            start, end = points[0], points[-1]

            for i in range(1, len(points) - 1):
                d = self._perpendicular_distance(points[i], start, end)
                if d > dmax:
                    index = i
                    dmax = d

            # Если максимальное расстояние больше epsilon, рекурсивно упрощаем
            if dmax > epsilon:
                rec_results1 = douglas_pecker(points[:index + 1], epsilon)
                rec_results2 = douglas_pecker(points[index:], epsilon)
                return rec_results1[:-1] + rec_results2
            else:
                return [start, end]

        return douglas_pecker(points, self.distance_threshold)

    def _perpendicular_distance(self, point: Point, line_start: Point, line_end: Point) -> float:
        """Вычисляет перпендикулярное расстояние от точки до линии"""
        if line_start == line_end:
            return point.distance(line_start)

        area = abs(
            (line_end.x - line_start.x) * (line_start.y - point.y) -
            (line_start.x - point.x) * (line_end.y - line_start.y)
        )
        line_len = line_start.distance(line_end)
        return area / line_len if line_len > 0 else 0

    def _find_vertices(self, points: List[Point]) -> List[Point]:
        """Находит вершины многоугольника на основе анализа углов"""
        if len(points) < 3:
            return points

        vertices = []
        self.angle_history = []  # Сбрасываем историю углов
        n = len(points)

        for i in range(n):
            # Предыдущая, текущая и следующая точки (с учетом цикличности)
            prev = points[(i - 1) % n]
            curr = points[i]
            next_p = points[(i + 1) % n]

            # Векторы от текущей точки к соседним
            v1 = Point(prev.x - curr.x, prev.y - curr.y)
            v2 = Point(next_p.x - curr.x, next_p.y - curr.y)

            # Вычисляем угол между векторами
            angle = self._vector_angle(v1, v2)
            self.angle_history.append(angle)  # Сохраняем угол для графика

            # Если угол значительно отличается от 180°, это вершина
            if abs(angle - math.pi) > self.angle_threshold:
                vertices.append(curr)

        # Убираем близко расположенные вершины
        cleaned_vertices = self._remove_close_vertices(vertices)

        # Если вершин слишком много, дополнительно упрощаем
        if len(cleaned_vertices) > 8:
            return self._reduce_points(cleaned_vertices)

        return cleaned_vertices

    def _vector_angle(self, v1: Point, v2: Point) -> float:
        """Вычисляет угол между двумя векторами в радианах"""
        dot_product = v1.x * v2.x + v1.y * v2.y
        mag1 = math.sqrt(v1.x ** 2 + v1.y ** 2)
        mag2 = math.sqrt(v2.x ** 2 + v2.y ** 2)

        if mag1 * mag2 == 0:
            return 0

        # Обеспечиваем значение в допустимом диапазоне для acos
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.acos(cos_angle)

    def _remove_close_vertices(self, vertices: List[Point]) -> List[Point]:
        """Удаляет вершины, расположенные слишком близко друг к другу"""
        if len(vertices) <= 1:
            return vertices

        result = [vertices[0]]
        for i in range(1, len(vertices)):
            if vertices[i].distance(result[-1]) > self.distance_threshold:
                result.append(vertices[i])

        return result

    def plot_recognition_diagram(self, original_points: List[Tuple[float, float]],
                                 vertices: List[Tuple[float, float]],
                                 save_path: Optional[str] = None):
        """
        Создает диаграмму, показывающую процесс распознавания
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Первый график: исходные точки
        orig_x, orig_y = zip(*original_points)
        ax1.scatter(orig_x, orig_y, c='blue', alpha=0.6, s=50, label='Исходные точки')

        # Отмечаем центр
        if self.center:
            ax1.scatter([self.center.x], [self.center.y], c='green', s=100, marker='x',
                        label='Центр', linewidth=2)

        ax1.set_title('Исходные точки')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')

        # Второй график: распознанные вершины
        vert_x, vert_y = zip(*vertices)
        ax2.scatter(orig_x, orig_y, c='blue', alpha=0.3, s=30, label='Исходные точки')
        ax2.plot(vert_x, vert_y, 'ro-', linewidth=2, markersize=8,
                 label='Распознанные вершины')

        # Подписываем вершины
        for i, (x, y) in enumerate(vertices):
            ax2.annotate(f'V{i + 1}', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=12, color='red')

        ax2.set_title(f'Распознан {len(vertices)}-угольник')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')

        # Третий график: диаграмма углов (как в Java-проекте)
        if self.angle_history:
            angles_degrees = [math.degrees(angle) for angle in self.angle_history]
            ax3.plot(angles_degrees, 'g-', linewidth=2, label='Углы между векторами')
            ax3.axhline(y=180, color='r', linestyle='--', alpha=0.7, label='180° (прямая линия)')

            # Отмечаем пороговые значения
            threshold_deg = math.degrees(self.angle_threshold)
            ax3.axhline(y=180 + threshold_deg, color='orange', linestyle=':',
                        alpha=0.5, label=f'Порог ±{threshold_deg:.1f}°')
            ax3.axhline(y=180 - threshold_deg, color='orange', linestyle=':', alpha=0.5)

            # Закрашиваем области, где углы считаются вершинами
            x = range(len(angles_degrees))
            ax3.fill_between(x, 180 + threshold_deg, angles_degrees,
                             where=(np.array(angles_degrees) > 180 + threshold_deg),
                             color='red', alpha=0.3, label='Области вершин')
            ax3.fill_between(x, 180 - threshold_deg, angles_degrees,
                             where=(np.array(angles_degrees) < 180 - threshold_deg),
                             color='red', alpha=0.3)

            ax3.set_title('Диаграмма углов распознавания')
            ax3.set_xlabel('Индекс точки в контуре')
            ax3.set_ylabel('Угол (градусы)')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Диаграмма сохранена в {save_path}")
        else:
            plt.show()

    def plot_polar_diagram(self, original_points: List[Tuple[float, float]],
                           vertices: List[Tuple[float, float]],
                           save_path: Optional[str] = None):
        """
        Создает полярную диаграмму для ВСЕХ исходных точек:
        угол относительно центра (рад) по X, расстояние по Y
        """
        if not self.polar_data:
            print("Нет данных для полярной диаграммы.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Сортируем данные по углу для непрерывного графика
        sorted_polar = sorted(self.polar_data, key=lambda x: x[0])
        angles = [item[0] for item in sorted_polar]
        distances = [item[1] for item in sorted_polar]

        # Строим график для всех исходных точек
        ax.scatter(angles, distances, c='blue', s=50, alpha=0.6, label='Все исходные точки')

        # Также строим линию, соединяющую точки по порядку углов
        ax.plot(angles, distances, 'b-', linewidth=1, alpha=0.3)

        # Отмечаем распознанные вершины на полярной диаграмме
        if vertices:
            vertex_angles = []
            vertex_distances = []

            for x, y in vertices:
                dx = x - self.center.x
                dy = y - self.center.y
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi
                distance = math.sqrt(dx ** 2 + dy ** 2)

                vertex_angles.append(angle)
                vertex_distances.append(distance)

            # Сортируем вершины по углу для правильного отображения
            vertex_data = sorted(zip(vertex_angles, vertex_distances), key=lambda x: x[0])
            vertex_angles, vertex_distances = zip(*vertex_data)

            ax.plot(vertex_angles, vertex_distances, 'ro-', linewidth=2,
                    markersize=8, label='Распознанные вершины')

            # Подписываем вершины
            for i, (angle, dist) in enumerate(zip(vertex_angles, vertex_distances)):
                ax.annotate(f'V{i + 1}', (angle, dist), xytext=(5, 5),
                            textcoords='offset points', fontsize=12, color='red', weight='bold')

        # Добавляем информацию о центре
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        ax.set_xlabel('Угол относительно центра (радианы)')
        ax.set_ylabel('Расстояние от центра')
        ax.set_title('Полярная диаграмма: расстояние от центра в зависимости от угла\n(все исходные точки)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Добавляем метки для особых углов
        special_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
        special_labels = ['0', 'π/2', 'π', '3π/2', '2π']
        ax.set_xticks(special_angles)
        ax.set_xticklabels(special_labels)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Полярная диаграмма сохранена в {save_path}")
        else:
            plt.show()

    def plot_comparison_diagram(self, original_points: List[Tuple[float, float]],
                                vertices: List[Tuple[float, float]],
                                save_path: Optional[str] = None):
        """
        Создает сравнительную диаграмму с обычными и полярными координатами
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Исходные точки
        orig_x, orig_y = zip(*original_points)
        ax1.scatter(orig_x, orig_y, c='blue', alpha=0.6, s=50, label='Исходные точки')
        if self.center:
            ax1.scatter([self.center.x], [self.center.y], c='green', s=100, marker='x',
                        label='Центр', linewidth=2)
        ax1.set_title('Исходные точки')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')

        # 2. Распознанные вершины
        vert_x, vert_y = zip(*vertices)
        ax2.scatter(orig_x, orig_y, c='blue', alpha=0.3, s=30, label='Исходные точки')
        ax2.plot(vert_x, vert_y, 'ro-', linewidth=2, markersize=8,
                 label='Распознанные вершины')
        for i, (x, y) in enumerate(vertices):
            ax2.annotate(f'V{i + 1}', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=12, color='red')
        ax2.set_title(f'Распознан {len(vertices)}-угольник')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')

        # 3. Полярная диаграмма всех точек
        if self.polar_data:
            sorted_polar = sorted(self.polar_data, key=lambda x: x[0])
            angles = [item[0] for item in sorted_polar]
            distances = [item[1] for item in sorted_polar]

            ax3.scatter(angles, distances, c='blue', s=50, alpha=0.6, label='Все исходные точки')
            ax3.plot(angles, distances, 'b-', linewidth=1, alpha=0.3)

            # Вершины на полярной диаграмме
            if vertices:
                vertex_angles = []
                vertex_distances = []
                for x, y in vertices:
                    dx = x - self.center.x
                    dy = y - self.center.y
                    angle = math.atan2(dy, dx)
                    if angle < 0:
                        angle += 2 * math.pi
                    distance = math.sqrt(dx ** 2 + dy ** 2)
                    vertex_angles.append(angle)
                    vertex_distances.append(distance)

                vertex_data = sorted(zip(vertex_angles, vertex_distances), key=lambda x: x[0])
                vertex_angles, vertex_distances = zip(*vertex_data)
                ax3.plot(vertex_angles, vertex_distances, 'ro-', linewidth=2,
                         markersize=8, label='Распознанные вершины')

            ax3.set_xlabel('Угол относительно центра (радианы)')
            ax3.set_ylabel('Расстояние от центра')
            ax3.set_title('Полярная диаграмма всех точек')
            ax3.grid(True, alpha=0.3)
            ax3.legend()

        # 4. Диаграмма углов распознавания
        if self.angle_history:
            angles_degrees = [math.degrees(angle) for angle in self.angle_history]
            ax4.plot(angles_degrees, 'g-', linewidth=2, label='Углы между векторами')
            ax4.axhline(y=180, color='r', linestyle='--', alpha=0.7, label='180° (прямая линия)')

            threshold_deg = math.degrees(self.angle_threshold)
            ax4.axhline(y=180 + threshold_deg, color='orange', linestyle=':',
                        alpha=0.5, label=f'Порог ±{threshold_deg:.1f}°')
            ax4.axhline(y=180 - threshold_deg, color='orange', linestyle=':', alpha=0.5)

            x = range(len(angles_degrees))
            ax4.fill_between(x, 180 + threshold_deg, angles_degrees,
                             where=(np.array(angles_degrees) > 180 + threshold_deg),
                             color='red', alpha=0.3, label='Области вершин')
            ax4.fill_between(x, 180 - threshold_deg, angles_degrees,
                             where=(np.array(angles_degrees) < 180 - threshold_deg),
                             color='red', alpha=0.3)

            ax4.set_title('Диаграмма углов распознавания')
            ax4.set_xlabel('Индекс точки в контуре')
            ax4.set_ylabel('Угол (градусы)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сравнительная диаграмма сохранена в {save_path}")
        else:
            plt.show()


# Тестирование с вашими данными
def test_with_your_data():
    points1 = [
        (68, 149), (69, 150), (70, 150), (71, 150), (69, 151), (72, 151), (73, 151), (75, 151),
        (76, 151), (77, 151), (78, 151), (69, 152), (74, 152), (78, 152), (70, 153), (79, 153),
        (70, 154), (79, 154), (71, 155), (80, 155), (71, 156), (80, 156), (72, 157), (73, 157),
        (80, 157), (81, 157), (74, 158), (75, 158), (79, 158), (73, 159), (76, 159), (77, 159),
        (78, 159)
    ]
    points2 = [
        (80, 154), (81, 155), (81, 156), (82, 156), (83, 156), (84, 156), (82, 157), (84, 157),
        (80, 158), (81, 158), (85, 158), (86, 158), (79, 159), (87, 159), (88, 159), (78, 160),
        (89, 160), (79, 161), (90, 161), (91, 161), (80, 162), (92, 162), (93, 162), (81, 163),
        (82, 163), (91, 163), (83, 164), (90, 164), (83, 165), (84, 165), (85, 165), (86, 165),
        (87, 165), (90, 165), (88, 166), (89, 166), (90, 166), (80, 154)
    ]

    # Создаем распознаватель с настройками для ваших данных
    recognizer = ShapeRecognizer(angle_threshold=0.5, distance_threshold=3.0)

    print("Обработка points1:")
    print(f"Исходных точек: {len(points1)}")

    # Распознаем вершины
    vertices1 = recognizer.recognize_shape(points1)
    print(f"Найдено вершин: {len(vertices1)}")
    print("Вершины:", vertices1)

    # Создаем различные диаграммы для первого набора точек
    recognizer.plot_recognition_diagram(points1, vertices1, "recognition_result1.png")
    recognizer.plot_polar_diagram(points1, vertices1, "polar_diagram1.png")
    recognizer.plot_comparison_diagram(points1, vertices1, "comparison_diagram1.png")

    print("\nОбработка points2:")
    print(f"Исходных точек: {len(points2)}")

    # Распознаем вершины для второго набора
    recognizer2 = ShapeRecognizer(angle_threshold=0.5, distance_threshold=3.0)
    vertices2 = recognizer2.recognize_shape(points2)
    print(f"Найдено вершин: {len(vertices2)}")
    print("Вершины:", vertices2)

    # Создаем различные диаграммы для второго набора точек
    recognizer2.plot_recognition_diagram(points2, vertices2, "recognition_result2.png")
    recognizer2.plot_polar_diagram(points2, vertices2, "polar_diagram2.png")
    recognizer2.plot_comparison_diagram(points2, vertices2, "comparison_diagram2.png")


if __name__ == "__main__":
    test_with_your_data()