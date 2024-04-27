import os
import cv2
import argparse
import dlib
import numpy as np

from vector import draw_vector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--img', required=True, help='Path for source image')
    parser.add_argument('--out', required=True, help='Path for  output images')
    args = parser.parse_args()


    # Загрузка изображения
    image = cv2.imread(args.img)

    # Конвертация изображения в оттенки серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Детекция лиц и точек ориентиров лица
    frontal_face_detector = dlib.get_frontal_face_detector()
    frontal_face_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = frontal_face_detector(image_gray)

    if faces is None:
        print('Нет лиц!!!')
        exit(-1)

    for face in faces:
        landmarks = frontal_face_predictor(image_gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
        #Отримовка векторов взгялда
        image = draw_vector(image,landmarks,landmarks_points)

 

    cv2.imwrite(args.out, image)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
        
    cv2.destroyAllWindows()
