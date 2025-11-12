import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


class SuperpixelPlotter:
    def __init__(self, json_file):
        """Инициализация с JSON файлом суперпикселей"""
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # Определяем тип файла и загружаем соответствующие данные
        if 'vector_superpixels' in self.data:
            # Это векторный файл
            self.superpixels = self.data['vector_superpixels']
            self.is_vectorized = True
            print(f"Загружен векторный файл: {len(self.superpixels)} суперпикселей")
        elif 'superpixels' in self.data:
            # Это исходный файл
            self.superpixels = self.data['superpixels']
            self.is_vectorized = False
            print(f"Загружен исходный файл: {len(self.superpixels)} суперпикселей")
        else:
            raise KeyError("Неизвестный формат JSON файла")

        self.height = self.data['image_dimensions']['height']
        self.width = self.data['image_dimensions']['width']
        self.filename = json_file

    def flip_y_coordinate(self, y):
        """Переворачивает координату Y"""
        return self.height - y

    def flip_points(self, points):
        """Переворачивает Y координаты у набора точек"""
        if isinstance(points, list) and points:
            if isinstance(points[0], dict) and 'x' in points[0] and 'y' in points[0]:
                return [{'x': p['x'], 'y': self.flip_y_coordinate(p['y'])} for p in points]
            elif isinstance(points[0], (list, tuple)) and len(points[0]) == 2:
                return [(p[0], self.flip_y_coordinate(p[1])) for p in points]
        return points

    def get_superpixel_by_id(self, sp_id):
        """Находит суперпиксель по ID"""
        for sp in self.superpixels:
            if sp['id'] == sp_id:
                return sp
        return None

    def plot_selected_superpixels(self, selected_ids, save_path=None):
        """
        Отрисовывает выбранные суперпиксели на одном изображении
        """
        fig, ax = plt.subplots(figsize=(15, 12))

        # Создаем пустое изображение
        empty_image = np.ones((self.height, self.width, 3))
        ax.imshow(empty_image, cmap='gray')

        # Цвета для разных суперпикселей
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_ids)))

        found_count = 0

        for i, sp_id in enumerate(selected_ids):
            superpixel = self.get_superpixel_by_id(sp_id)

            if superpixel is None:
                print(f"Суперпиксель с ID {sp_id} не найден")
                continue

            found_count += 1

            if self.is_vectorized:
                # Отрисовка векторных суперпикселей
                self._plot_vector_superpixel(ax, superpixel, colors[i], sp_id)
            else:
                # Отрисовка исходных суперпикселей
                self._plot_original_superpixel(ax, superpixel, colors[i], sp_id)

        if found_count == 0:
            print("Не найдено ни одного суперпикселя из указанных ID")
            plt.close()
            return

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')

        title_type = "Векторизованные" if self.is_vectorized else "Исходные"
        ax.set_title(f'{title_type} суперпиксели: {selected_ids}\n(всего найдено: {found_count}/{len(selected_ids)})')
        ax.legend(loc='upper right')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Изображение сохранено в {save_path}")

        plt.show()

    def _plot_vector_superpixel(self, ax, superpixel, color, sp_id):
        """Отрисовывает векторный суперпиксель"""
        vertices = superpixel.get('vertices', [])

        if vertices:
            # Создаем полигон из векторных вершин
            polygon_vertices = [(v['x'], v['y']) for v in vertices]
            polygon = patches.Polygon(polygon_vertices,
                                      closed=True,
                                      alpha=0.7,
                                      edgecolor=color,
                                      facecolor=color,
                                      linewidth=3,
                                      label=f'SP {sp_id} (векторный)')
            ax.add_patch(polygon)

            # Рисуем вершины
            x_coords = [v['x'] for v in vertices]
            y_coords = [v['y'] for v in vertices]
            ax.scatter(x_coords, y_coords, s=50, color=color, alpha=0.9, marker='s', edgecolors='black')

            # Рисуем центр
            center = superpixel['center']
            ax.plot(center['x'], center['y'], 'o', markersize=10,
                    markerfacecolor=color, markeredgecolor='black', markeredgewidth=2)

            # Подписываем ID
            ax.text(center['x'], center['y'], str(sp_id),
                    fontsize=12, ha='center', va='center',
                    fontweight='bold', color='white')
        else:
            print(f"Векторизованный суперпиксель {sp_id} не имеет вершин")

    def _plot_original_superpixel(self, ax, superpixel, color, sp_id):
        """Отрисовывает исходный суперпиксель"""
        boundary_points = superpixel.get('boundary_points', [])

        if boundary_points:
            # Переворачиваем точки для отображения
            flipped_points = self.flip_points(boundary_points)

            # Создаем полигон из граничных точек
            polygon_vertices = [(p['x'], p['y']) for p in flipped_points]
            polygon = patches.Polygon(polygon_vertices,
                                      closed=True,
                                      alpha=0.6,
                                      edgecolor=color,
                                      facecolor=color,
                                      linewidth=2,
                                      label=f'SP {sp_id}')
            ax.add_patch(polygon)

            # Рисуем граничные точки
            x_coords = [p['x'] for p in flipped_points]
            y_coords = [p['y'] for p in flipped_points]
            ax.scatter(x_coords, y_coords, s=8, color=color, alpha=0.7, edgecolors='black', linewidth=0.5)

            # Рисуем центр
            center_x = superpixel['center']['x']
            center_y = self.flip_y_coordinate(superpixel['center']['y'])
            ax.plot(center_x, center_y, 'o', markersize=8,
                    markerfacecolor=color, markeredgecolor='black', markeredgewidth=2)

            # Подписываем ID
            ax.text(center_x, center_y, str(sp_id),
                    fontsize=10, ha='center', va='center',
                    fontweight='bold', color='white')
        else:
            print(f"Суперпиксель {sp_id} не имеет граничных точек")

    def plot_comparison(self, original_json, vectorized_json, selected_ids, save_path=None):
        """
        Сравнивает исходные и векторные границы для выбранных суперпикселей
        """
        # Загружаем оба файла
        with open(original_json, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        with open(vectorized_json, 'r', encoding='utf-8') as f:
            vectorized_data = json.load(f)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Создаем пустые изображения
        empty_image = np.ones((self.height, self.width, 3))

        # Первый subplot: исходные границы
        ax1.imshow(empty_image, cmap='gray')
        ax1.set_title('Исходные границы')

        # Второй subplot: векторные границы
        ax2.imshow(empty_image, cmap='gray')
        ax2.set_title('Векторизованные границы')

        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_ids)))

        for i, sp_id in enumerate(selected_ids):
            # Ищем в исходных данных
            original_sp = None
            for sp in original_data['superpixels']:
                if sp['id'] == sp_id:
                    original_sp = sp
                    break

            # Ищем в векторных данных
            vector_sp = None
            for sp in vectorized_data['vector_superpixels']:
                if sp['id'] == sp_id:
                    vector_sp = sp
                    break

            if original_sp and original_sp.get('boundary_points'):
                # Рисуем исходные границы
                flipped_points = self.flip_points(original_sp['boundary_points'])
                polygon_vertices = [(p['x'], p['y']) for p in flipped_points]
                polygon1 = patches.Polygon(polygon_vertices,
                                           closed=True,
                                           alpha=0.5,
                                           edgecolor=colors[i],
                                           facecolor=colors[i],
                                           linewidth=2)
                ax1.add_patch(polygon1)

                # Центр
                center_x = original_sp['center']['x']
                center_y = self.flip_y_coordinate(original_sp['center']['y'])
                ax1.plot(center_x, center_y, 'o', markersize=6,
                         markerfacecolor=colors[i], markeredgecolor='black')
                ax1.text(center_x, center_y, str(sp_id), fontsize=10, ha='center', va='center')

            if vector_sp and vector_sp.get('vertices'):
                # Рисуем векторные границы
                vertices = vector_sp['vertices']
                polygon_vertices = [(v['x'], v['y']) for v in vertices]
                polygon2 = patches.Polygon(polygon_vertices,
                                           closed=True,
                                           alpha=0.7,
                                           edgecolor=colors[i],
                                           facecolor=colors[i],
                                           linewidth=3)
                ax2.add_patch(polygon2)

                # Центр и вершины
                center = vector_sp['center']
                ax2.plot(center['x'], center['y'], 'o', markersize=6,
                         markerfacecolor=colors[i], markeredgecolor='black')
                ax2.text(center['x'], center['y'], str(sp_id), fontsize=10, ha='center', va='center')

                # Вершины
                x_coords = [v['x'] for v in vertices]
                y_coords = [v['y'] for v in vertices]
                ax2.scatter(x_coords, y_coords, s=30, color=colors[i], alpha=0.8, marker='s')

        for ax in [ax1, ax2]:
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.set_aspect('equal')
            ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Сравнение сохранено в {save_path}")

        plt.show()


def find_json_files():
    """Находит все JSON файлы в текущей директории"""
    return [f for f in os.listdir('.') if f.endswith('.json')]


def main():
    """Основная функция для запуска"""
    json_files = find_json_files()

    if not json_files:
        print("Не найдены JSON файлы в текущей директории")
        return

    print("Доступные JSON файлы:")
    for i, file in enumerate(json_files):
        print(f"{i + 1}. {file}")

    try:
        choice = int(input("Выберите файл (номер): ").strip())
        selected_file = json_files[choice - 1]
    except (ValueError, IndexError):
        print("Неверный выбор")
        return

    # Запрос ID суперпикселей
    ids_input = input("Введите ID суперпикселей через запятую: ").strip()
    try:
        selected_ids = [int(id_str.strip()) for id_str in ids_input.split(',')]
    except ValueError:
        print("Неверный формат ID. Используйте числа, разделенные запятыми.")
        return

    try:
        # Создаем плоттер и отрисовываем
        plotter = SuperpixelPlotter(selected_file)

        # Создаем имя файла для сохранения
        base_name = os.path.splitext(selected_file)[0]
        ids_str = '_'.join(map(str, selected_ids))
        save_path = f'{base_name}_selected_{ids_str}.png'

        plotter.plot_selected_superpixels(selected_ids, save_path)

        # Если есть оба типа файлов, предлагаем сравнение
        if 'vectorized' in selected_file:
            # Ищем соответствующий исходный файл
            original_files = [f for f in json_files if 'vectorized' not in f and 'superpixels' in f]
            if original_files:
                print(f"\nНайден исходный файл: {original_files[0]}")
                if input("Создать сравнение с исходными границами? (y/n): ").lower() == 'y':
                    comparison_path = f'comparison_{ids_str}.png'
                    plotter.plot_comparison(original_files[0], selected_file, selected_ids, comparison_path)

    except KeyError as e:
        print(f"Ошибка формата JSON файла: {e}")
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()