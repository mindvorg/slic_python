import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

def voronoi_segmentation(imgcol, u, v, p=2.0, num_superpixels=100, compactness=10,
                         max_iterations=10, random_init=True, poisson_radius_factor=0.5):
    """
    Сегментация изображения на основе диаграммы Вороного.
    Параметры p, compactness, max_iterations оставлены для совместимости интерфейса
    с оригинальной функцией slic_modif, но не используются.

    Parameters
    ----------
    imgcol : np.ndarray (H, W, 3)
        Цветное RGB-изображение (может быть предварительно перевёрнуто по вертикали).
    u, v : np.ndarray (H, W)
        Компоненты градиента (не используются в разбиении, но сохранены для унификации).
    num_superpixels : int
        Желаемое количество суперпикселей.
    random_init : bool
        Если True – центры инициализируются с помощью Poisson disk sampling.
        Если False – равномерная сетка с шагом, соответствующим num_superpixels.
    poisson_radius_factor : float
        Коэффициент минимального расстояния между центрами для Poisson disk (0..1).
        Чем меньше, тем плотнее могут располагаться центры.

    Returns
    -------
    labels : np.ndarray (H, W) int32
        Карта меток суперпикселей.
    centers : np.ndarray (N, 2)
        Координаты центров суперпикселей (y, x).
    """
    height, width = imgcol.shape[:2]

    # ----- генерация центров (точек Вороного) -----
    if random_init:
        # Оценка шага для желаемого количества суперпикселей
        step = int(np.sqrt((height * width) / num_superpixels))
        min_distance = step * poisson_radius_factor
        radius = min_distance / max(height, width)   # нормализованный радиус

        engine = qmc.PoissonDisk(d=2, radius=radius, hypersphere='volume',
                                 ncandidates=30, seed=42)
        points_norm = engine.random(num_superpixels)   # может вернуть меньше точек

        # Масштабируем в координаты изображения
        seeds_y = np.clip((points_norm[:, 0] * (height - 1)).astype(int), 0, height-1)
        seeds_x = np.clip((points_norm[:, 1] * (width - 1)).astype(int), 0, width-1)
    else:
        # Равномерная сетка
        step = int(np.sqrt((height * width) / num_superpixels))
        yv, xv = np.mgrid[step//2 : height : step, step//2 : width : step]
        seeds_y = yv.ravel()
        seeds_x = xv.ravel()

    seeds = np.column_stack((seeds_y, seeds_x))
    n_seeds = seeds.shape[0]
    if n_seeds == 0:
        # fallback: один центр в середине
        seeds = np.array([[height//2, width//2]], dtype=int)
        n_seeds = 1

    # ----- построение диаграммы Вороного -----
    # Создаём координаты всех пикселей
    yy, xx = np.mgrid[0:height, 0:width]
    coords = np.column_stack((yy.ravel(), xx.ravel()))

    # k-d дерево для быстрого поиска ближайшего центра
    tree = cKDTree(seeds)
    _, indices = tree.query(coords, k=1)  # indices – номер центра для каждого пикселя

    labels = indices.reshape(height, width).astype(np.int32)

    # Центры возвращаем как (N,2) для совместимости с последующей обработкой
    centers = seeds.copy()

    return labels, centers