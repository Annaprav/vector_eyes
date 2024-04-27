import cv2
import numpy as np

def read_specific_pixels(image, center, height):
    """
    Читает пиксели вокруг центра заданной области изображения.

    Args:
        image: Массив numpy, представляющий изображение.
        center: Координаты центра области (горизонтальная, вертикальная).
        height: Высота/ширина квадратной области вокруг центра.

    Returns:
        black_colour: Список координат черных пикселей в области.
    """
    # Разбиваем координаты центра на горизонтальную и вертикальную части
    start_horizontal, start_vertical = center

    black_colour = []

    # Проходимся по области вокруг центра
    for j in range(-int(height*1.5), 0):
        for i in range(start_vertical - height, start_vertical + int(height*1.5)):
            black_lower_range = [80, 50, 50]
            pixel = start_horizontal + j
            color = image[int(pixel), i]
            # Проверяем, является ли цвет пикселя черным
            if all(color <= black_lower_range):
                black_colour.append([int(pixel), i])
    return black_colour


def get_eye_coordinates(image, face_coordinates):
    """
    Получает координаты областей левого и правого глаз, а также черные пиксели вокруг зрачков.

    Args:
        image: Массив numpy, представляющий изображение.
        face_coordinates: Координаты ключевых точек лица.

    Returns:
        left_eye: Область левого глаза.
        right_eye: Область правого глаза.
        left_black_pixels: Список черных пикселей вокруг левого зрачка.
        right_black_pixels: Список черных пикселей вокруг правого зрачка.
    """
    # Вырезаем области глаз (по бровям)
    # Границы по верхней точке брови и нижней точки глаза, по бокам область ограничена крайними точками глаз (с учетом этого берутся индексы)
    left_eye = image[face_coordinates[19][1]:face_coordinates[42][1], face_coordinates[36][0]:face_coordinates[39][0]]
    right_eye = image[face_coordinates[24][1]:face_coordinates[47][1], face_coordinates[42][0]:face_coordinates[45][0]]

    # Вычисляем координаты центров глаз
    eye_left_coordinate = [int(face_coordinates[37][0] + (face_coordinates[38][0] - face_coordinates[37][0]) / 2),
                           int(face_coordinates[38][1] + (face_coordinates[40][1] - face_coordinates[38][1]) / 2)]
    eye_right_coordinate = [int(face_coordinates[43][0] + (face_coordinates[44][0] - face_coordinates[43][0]) / 2),
                            int(face_coordinates[43][1] + (face_coordinates[47][1] - face_coordinates[43][1]) / 2)]

    # Читаем черные пиксели вокруг центра глаза
    left_black_pixels = read_specific_pixels(image, eye_left_coordinate,
                                              int((face_coordinates[38][0] - face_coordinates[37][0]) / 2))
    right_black_pixels = read_specific_pixels(image, eye_right_coordinate,
                                               int((face_coordinates[44][0] - face_coordinates[43][0]) / 2))

    return left_eye, right_eye, left_black_pixels, right_black_pixels


def get_pupil_point(image, black_coordinates, eye_top_point_x, eye_bottom_point_y):
    """
    Находит координаты зрачка на изображении.

    Args:
        image: Массив numpy, представляющий изображение.
        black_coordinates: Список черных пикселей вокруг зрачка.
        eye_top_point_x: Координата x верхней точки глаза.
        eye_bottom_point_y: Координата y нижней точки глаза.

    Returns:
        pupil_point: Координаты зрачка на изображении.
    """
    pupil_point = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=250, param2=10, minRadius=1, maxRadius=-1)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:1]:
            pupil_point = [int(eye_top_point_x[0]) + i[0], int(eye_bottom_point_y[1]) + i[1]]
    else:
        a = 0
        for j, k in black_coordinates:
            if a == int(len(black_coordinates) / 2):
                pupil_point = [k, j]
            a += 1
    return pupil_point
