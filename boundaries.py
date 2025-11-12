import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import argparse


class BoundaryVectorizer:
    def __init__(self, json_file):
        """
        Инициализация векторaйзера с данными из JSON
        """
        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"Файл {json_file} не найден. Сначала запустите main.py для создания суперпикселей.")

        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.superpixels = self.data['superpixels']
        self.height = self.data['image_dimensions']['height']
        self.width = self.data['image_dimensions']['width']
        self.filename = json_file

        print(f"Загружено {len(self.superpixels)} суперпикселей из {json_file}")

    def reweighted_least_squares_full(self, points, max_iterations=20, tolerance=1e-4, weight_threshold=0.5):
        """
        RLS с общим уравнением прямой Ax + By + C = 0
        """
        if len(points) < 2:
            return None, []

        x = points[:, 0]
        y = points[:, 1]

        weights = np.ones(len(points))
        best_line = None
        best_weights = None

        for iteration in range(max_iterations):
            if np.sum(weights) < 1e-10:
                break

            weights_normalized = weights / np.sum(weights)

            # Вычисляем параметры прямой Ax + By + C = 0
            mean_x = np.average(x, weights=weights_normalized)
            mean_y = np.average(y, weights=weights_normalized)

            # Центрируем точки
            x_centered = x - mean_x
            y_centered = y - mean_y

            # Вычисляем матрицу ковариации
            cov_xx = np.average(x_centered * x_centered, weights=weights_normalized)
            cov_yy = np.average(y_centered * y_centered, weights=weights_normalized)
            cov_xy = np.average(x_centered * y_centered, weights=weights_normalized)

            # Собственный вектор, соответствующий наименьшей дисперсии
            # Направление с наименьшей дисперсией - нормаль к прямой
            if abs(cov_xy) < 1e-10:
                if cov_xx < cov_yy:
                    A, B = 1, 0  # Вертикальная прямая
                else:
                    A, B = 0, 1  # Горизонтальная прямая
            else:
                # Вычисляем собственный вектор для наименьшего собственного значения
                trace = cov_xx + cov_yy
                det = cov_xx * cov_yy - cov_xy * cov_xy
                lambda_min = (trace - np.sqrt(trace ** 2 - 4 * det)) / 2

                A = cov_xy
                B = lambda_min - cov_xx
                norm = np.sqrt(A ** 2 + B ** 2)
                if norm < 1e-10:
                    break
                A, B = A / norm, B / norm

            # Вычисляем C
            C = -(A * mean_x + B * mean_y)

            # Нормализуем уравнение (делаем A² + B² = 1)
            norm_abc = np.sqrt(A ** 2 + B ** 2 + C ** 2)
            A, B, C = A / norm_abc, B / norm_abc, C / norm_abc

            # Вычисляем расстояния до прямой
            distances = np.abs(A * x + B * y + C)

            # Обновляем веса по Tukey's biweight
            mad = np.median(np.abs(distances - np.median(distances)))
            if mad < 1e-10:
                break

            scaled_residuals = distances / (4.685 * mad)
            new_weights = np.zeros_like(weights)
            mask = scaled_residuals < 1
            new_weights[mask] = (1 - scaled_residuals[mask] ** 2) ** 2

            # Сохраняем лучшую прямую
            best_line = (A, B, C)
            best_weights = new_weights.copy()

            # Проверяем сходимость
            weight_change = np.max(np.abs(new_weights - weights))
            weights = new_weights

            if weight_change < tolerance:
                break

        # Находим ключевые точки (с высокими весами)
        if best_weights is not None and best_line is not None:
            key_point_indices = np.where(best_weights > weight_threshold)[0]
            key_points = points[key_point_indices]

            # Находим крайние точки среди ключевых
            if len(key_points) >= 2:
                A, B, C = best_line

                # Направляющий вектор прямой (перпендикулярно (A,B))
                dir_vector = np.array([-B, A])
                dir_vector = dir_vector / np.linalg.norm(dir_vector)

                # Проецируем точки на направляющий вектор
                projections = np.dot(key_points, dir_vector)

                # Находим точки с минимальной и максимальной проекцией
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)

                extreme_points = [key_points[min_idx], key_points[max_idx]]
                return best_line, extreme_points

        return None, []

    def find_shared_boundary_points(self, boundary_points1, boundary_points2, tolerance=2.0):
        """
        Находит общие точки между двумя границами
        """
        shared_points = []

        for point1 in boundary_points1:
            for point2 in boundary_points2:
                distance = np.sqrt((point1['x'] - point2['x']) ** 2 +
                                   (point1['y'] - point2['y']) ** 2)
                if distance < tolerance:
                    shared_points.append({
                        'point1': point1,
                        'point2': point2,
                        'distance': distance
                    })

        return shared_points

    def optimize_vertices_with_neighbors(self, vector_data):
        """
        Оптимизирует вершины с учетом соседних суперпикселей
        """
        from sklearn.cluster import DBSCAN

        # Собираем все вершины
        all_vertices = []
        vertex_info = []

        for sp in vector_data['vector_superpixels']:
            for i, vertex in enumerate(sp['vertices']):
                all_vertices.append([vertex['x'], vertex['y']])
                vertex_info.append({
                    'superpixel_id': sp['id'],
                    'vertex_index': i,
                    'vertex': vertex
                })

        if len(all_vertices) < 2:
            return vector_data

        # Кластеризуем близкие вершины
        points = np.array(all_vertices)
        clustering = DBSCAN(eps=3.0, min_samples=2).fit(points)
        labels = clustering.labels_

        # Оптимизируем вершины в кластерах
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue

            cluster_mask = labels == cluster_id
            cluster_points = points[cluster_mask]
            cluster_center = np.mean(cluster_points, axis=0)

            # Заменяем все вершины кластера на общий центр
            for i, info in enumerate(vertex_info):
                if labels[i] == cluster_id:
                    sp_id = info['superpixel_id']
                    vertex_idx = info['vertex_index']

                    # Находим суперпиксель и обновляем вершину
                    for sp in vector_data['vector_superpixels']:
                        if sp['id'] == sp_id and vertex_idx < len(sp['vertices']):
                            sp['vertices'][vertex_idx] = {
                                'x': float(cluster_center[0]),
                                'y': float(cluster_center[1])
                            }

        return vector_data


    def find_shared_vertices(self, superpixel_data, neighbor_superpixels):
        """
        Находит общие вершины с соседними суперпикселями
        """
        shared_vertices = []

        for neighbor in neighbor_superpixels:
            if neighbor['vertices'] and superpixel_data['vertices']:
                # Находим общие вершины (близкие точки)
                for vertex in superpixel_data['vertices']:
                    for n_vertex in neighbor['vertices']:
                        distance = np.sqrt((vertex['x'] - n_vertex['x']) ** 2 +
                                           (vertex['y'] - n_vertex['y']) ** 2)
                        if distance < 3.0:  # Пороговое расстояние
                            shared_vertices.append({
                                'vertex': vertex,
                                'neighbor_id': neighbor['id'],
                                'shared_with': n_vertex
                            })

        return shared_vertices

    def optimize_shared_vertices(self, vector_data):
        """
        Оптимизирует вершины, которые являются общими для нескольких суперпикселей
        """
        # Собираем все вершины и находим кластеры близких точек
        all_vertices = []
        for superpixel in vector_data['vector_superpixels']:
            for vertex in superpixel['vertices']:
                all_vertices.append({
                    'x': vertex['x'],
                    'y': vertex['y'],
                    'superpixel_id': superpixel['id'],
                    'vertex_index': superpixel['vertices'].index(vertex)
                })

        # Кластеризуем близкие вершины
        from sklearn.cluster import DBSCAN

        if len(all_vertices) < 2:
            return vector_data

        points = np.array([[v['x'], v['y']] for v in all_vertices])

        # DBSCAN для нахождения кластеров близких точек
        clustering = DBSCAN(eps=3.0, min_samples=2).fit(points)
        labels = clustering.labels_

        # Для каждого кластера находим среднюю точку
        unique_labels = set(labels)
        cluster_centers = {}

        for label in unique_labels:
            if label == -1:  # Шум
                continue
            cluster_points = points[labels == label]
            center = np.mean(cluster_points, axis=0)
            cluster_centers[label] = center

        # Заменяем вершины в кластерах на общие центры
        for i, vertex_info in enumerate(all_vertices):
            label = labels[i]
            if label in cluster_centers:
                # Обновляем вершину
                sp_id = vertex_info['superpixel_id']
                vertex_idx = vertex_info['vertex_index']

                for sp in vector_data['vector_superpixels']:
                    if sp['id'] == sp_id and vertex_idx < len(sp['vertices']):
                        sp['vertices'][vertex_idx] = {
                            'x': float(cluster_centers[label][0]),
                            'y': float(cluster_centers[label][1])
                        }

        return vector_data


    def try_vectorize_with_n_sides(self, points, n_sides):
        """
        Пытается векторизовать с заданным количеством сторон
        """
        if len(points) < n_sides:
            return None

        # Сегментируем точки на n_sides секторов
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

        # Сортируем точки по углу
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        # Разбиваем на сегменты
        segment_size = len(points) // n_sides
        segments = []

        for i in range(n_sides):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < n_sides - 1 else len(points)
            segments.append(sorted_points[start_idx:end_idx])

        # Для каждого сегмента находим ключевые точки с помощью RLS
        key_points = []
        used_points = set()  # Для отслеживания уже использованных точек

        for segment in segments:
            if len(segment) < 2:
                continue

            # Применяем RLS к сегменту
            line, extreme_points = self.reweighted_least_squares_full(
                segment)  # FIXED: changed weights to extreme_points
            if line is None:
                continue

            # Используем крайние точки, найденные RLS
            if extreme_points and len(extreme_points) >= 2:
                # Берем крайние точки, которые еще не использовались
                for point in extreme_points:
                    point_tuple = (point[0], point[1])

                    # Проверяем, не использовалась ли уже эта точка
                    if point_tuple not in used_points:
                        key_points.append(point)
                        used_points.add(point_tuple)

        # Если набрали достаточно уникальных точек, строим полигон
        if len(key_points) >= 3:
            return self.build_polygon_from_unique_points(key_points, n_sides)

        return None

    def build_polygon_from_unique_points(self, key_points, n_sides):
        """
        Строит полигон ТОЛЬКО из исходных точек границы
        """
        points_array = np.array(key_points)

        if len(points_array) < 3:
            return None

        # Упорядочиваем точки по углу
        ordered_points = self.order_points_by_angle(points_array)

        # Если точек больше чем нужно - равномерно выбираем n_sides точек
        if len(ordered_points) > n_sides:
            # Выбираем каждую k-ю точку для равномерного распределения
            step = len(ordered_points) / n_sides
            indices = [int(i * step) for i in range(n_sides)]
            ordered_points = [ordered_points[i] for i in indices]

        # Если точек меньше - оставляем как есть (не создаем новых!)
        return ordered_points


    def fallback_vectorization(self, points):
        """
        Упрощенный метод векторизации, когда основные методы не сработали
        """
        if len(points) < 3:
            return None

        # Просто берем выпуклую оболочку всех точек
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            return [{'x': float(p[0]), 'y': float(p[1])} for p in hull_points]
        except:
            # Если и это не сработало, берем равномерно распределенные точки
            return self.order_points_by_angle(points)

    def order_points_by_angle(self, points):
        """
        Упорядочивает точки по углу относительно центра
        """
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        return [{'x': float(p[0]), 'y': float(p[1])} for p in sorted_points]

    def vectorize_superpixel(self, superpixel_data, max_sides=6):
        """
        Улучшенная векторизация с гарантией уникальных точек и fallback на меньшее количество сторон
        """
        boundary_points = superpixel_data['boundary_points']

        if len(boundary_points) < 3:
            return None

        # Преобразуем в numpy array
        points = np.array([[p['x'], p['y']] for p in boundary_points])

        # Пытаемся векторизовать с разным количеством сторон
        for n_sides in range(max_sides, 2, -1):  # от 6 до 3
            vertices = self.try_vectorize_with_n_sides(points, n_sides)
            if vertices and len(vertices) >= 3:
                return vertices

        # Если ничего не сработало, используем упрощенный метод
        return self.fallback_vectorization(points)


    def order_points_by_angle(self, points):
        """
        Упорядочивает точки по углу относительно центра
        """
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        return [{'x': float(p[0]), 'y': float(p[1])} for p in sorted_points]

    def vectorize_all_superpixels(self, expected_sides=6):
        """
        Векторизует все суперпиксели с оптимизацией общих вершин
        """
        vector_data = {
            'image_dimensions': self.data['image_dimensions'],
            'vector_superpixels': [],
            'source_file': self.filename
        }

        successful_count = 0

        # Первый проход - базовая векторизация
        for superpixel in self.superpixels:
            vertices = self.vectorize_superpixel(superpixel, expected_sides)

            vector_superpixel = {
                'id': superpixel['id'],
                'center': superpixel['center'],
                'vertices': vertices if vertices else [],
                'area': superpixel['area'],
                'neighbors': superpixel.get('neighbors', [])
            }

            vector_data['vector_superpixels'].append(vector_superpixel)

            if vertices and len(vertices) >= 3:
                successful_count += 1

        print(f"Успешно векторизовано {successful_count}/{len(self.superpixels)} суперпикселей")

        # Второй проход - оптимизация общих вершин
        vector_data = self.optimize_vertices_with_neighbors(vector_data)

        return vector_data

    def visualize_vectorized_boundaries(self, vector_data, save_path=None):
        """
        Визуализирует векторные границы
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Первый subplot: векторные границы
        empty_image = np.ones((self.height, self.width, 3))
        ax1.imshow(empty_image, cmap='gray')

        # Рисуем векторные границы
        colors = plt.cm.Set3(np.linspace(0, 1, len(vector_data['vector_superpixels'])))

        for i, superpixel in enumerate(vector_data['vector_superpixels']):
            vertices = superpixel['vertices']

            if len(vertices) >= 3:
                # Создаем полигон
                polygon_vertices = [(v['x'], v['y']) for v in vertices]
                polygon = patches.Polygon(polygon_vertices,
                                          closed=True,
                                          alpha=0.7,
                                          edgecolor=colors[i % len(colors)],
                                          facecolor=colors[i % len(colors)],
                                          linewidth=2)
                ax1.add_patch(polygon)

                # Подписываем ID
                center = superpixel['center']
                ax1.text(center['x'], center['y'], str(superpixel['id']),
                         fontsize=8, ha='center', va='center', fontweight='bold')

        ax1.set_xlim(0, self.width)
        ax1.set_ylim(self.height, 0)
        ax1.set_aspect('equal')
        ax1.set_title('Векторизованные границы суперпикселей')
        ax1.axis('off')

        # Второй subplot: сравнение с исходными границами
        ax2.imshow(empty_image, cmap='gray')

        # Рисуем исходные граничные точки
        for i, superpixel in enumerate(self.superpixels):
            boundary_points = superpixel['boundary_points']
            if boundary_points:
                x_coords = [p['x'] for p in boundary_points]
                y_coords = [p['y'] for p in boundary_points]
                ax2.scatter(x_coords, y_coords, s=1, alpha=0.6, color=colors[i % len(colors)])

                # Центр
                center = superpixel['center']
                ax2.plot(center['x'], center['y'], 'ro', markersize=3)

        ax2.set_xlim(0, self.width)
        ax2.set_ylim(self.height, 0)
        ax2.set_aspect('equal')
        ax2.set_title('Исходные граничные точки')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Визуализация сохранена в {save_path}")

        plt.show()

    def save_vector_data(self, vector_data, filename=None):
        """
        Сохраняет векторные данные в JSON
        """
        if filename is None:
            base_name = os.path.splitext(self.filename)[0]
            filename = f"{base_name}_vectorized.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, indent=2, ensure_ascii=False)

        print(f"Векторные данные сохранены в {filename}")
        return filename


def find_superpixel_files():
    """Находит доступные файлы с суперпикселями"""
    files = []
    for file in os.listdir('.'):
        if file.startswith('superpixels_') and file.endswith('.json'):
            files.append(file)
    return files


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Векторизация границ суперпикселей')
    parser.add_argument('--input', '-i', help='Входной JSON файл с суперпикселями')
    parser.add_argument('--sides', '-s', type=int, default=6, help='Ожидаемое количество сторон (по умолчанию 6)')

    args = parser.parse_args()

    # Если файл не указан, показываем доступные
    if not args.input:
        available_files = find_superpixel_files()
        if not available_files:
            print("Не найдены файлы с суперпикселями.")
            print("Сначала запустите main.py для создания суперпикселей.")
            return

        print("Доступные файлы с суперпикселями:")
        for i, file in enumerate(available_files):
            print(f"{i + 1}. {file}")

        choice = input("Выберите файл (номер): ").strip()
        try:
            args.input = available_files[int(choice) - 1]
        except (ValueError, IndexError):
            print("Неверный выбор")
            return

    try:
        # Создаем векторaйзер
        vectorizer = BoundaryVectorizer(args.input)

        # Векторизуем суперпиксели
        print(f"Векторизация с ожидаемым количеством сторон: {args.sides}")
        vector_data = vectorizer.vectorize_all_superpixels(expected_sides=args.sides)

        # Сохраняем результаты
        output_file = vectorizer.save_vector_data(vector_data)

        # Визуализируем
        image_file = output_file.replace('.json', '.png')
        vectorizer.visualize_vectorized_boundaries(vector_data, image_file)

        print("\nГотово! Результаты:")
        print(f"- Векторные данные: {output_file}")
        print(f"- Визуализация: {image_file}")

    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()