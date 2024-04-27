import cv2      
from pupil import read_specific_pixels, get_eye_coordinates,get_pupil_point

def draw_vector(image,landmarks,landmarks_points):
        
        # Находим области изображения глаз и черные пиксели на них
        left_eye, right_eye, eye_left_black_pixels, eye_right_black_pixels = get_eye_coordinates(image, landmarks_points)

        #Находим точки зрачка глаз
        left_eye_coord, eye_brow_coord = landmarks_points[36], landmarks_points[19]
        left_pupil_point = get_pupil_point(left_eye, eye_left_black_pixels, left_eye_coord, eye_brow_coord)

        right_eye_coord = landmarks_points[42]
        right_pupil_point = get_pupil_point(right_eye, eye_right_black_pixels, right_eye_coord, eye_brow_coord)

        # Находим по 6 точек глаз
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
        
        # Определяем координаты центра глаза
        left_eye_center = (sum([p[0] for p in left_eye_points]) // 6, sum([p[1] for p in left_eye_points]) // 6)
        right_eye_center = (sum([p[0] for p in right_eye_points]) // 6, sum([p[1] for p in right_eye_points]) // 6)

        # Определяем направление векторов (вектор - от центра до зрачка)
        left_vector = (left_pupil_point[0] - left_eye_center[0], left_pupil_point[1] - left_eye_center[1])
        right_vector = (right_pupil_point[0] - right_eye_center[0], right_pupil_point[1] - right_eye_center[1])

        # Находим вектор = сложения векторов (вектор взгяялда, биссектриса)
        bisector_vector = left_vector + right_vector

        # Изменяем длину вектора взгляда
        length = 10
        left_pupil_point_extended = (left_pupil_point[0] + bisector_vector[0] * length, left_pupil_point[1] + bisector_vector[1] * length)
        right_pupil_point_extended = (right_pupil_point[0] + bisector_vector[0] * length, right_pupil_point[1] + bisector_vector[1] * length)

        # Рисуем прямые, параллельные биссектрисе и проходящие через точки зрачков
        cv2.arrowedLine(image, left_pupil_point, tuple(map(int, left_pupil_point_extended)), (0, 255, 255), 2)
        cv2.arrowedLine(image, right_pupil_point, tuple(map(int, right_pupil_point_extended)), (0, 255, 255), 2)

        return image