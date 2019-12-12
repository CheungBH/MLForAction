from config import config
import cv2
from src.processor import ImageProcessor
from angle_calculator import AngleCalculator


video_path = config.webcam_num

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
joint_ls = {"Left arm": [5, 7, 9], "Right arm": [6, 8, 10], "Left shoulder": [7, 9, 11], "Right shoulder": [8, 10, 12]}
AC = AngleCalculator()


def run(video_path):
    frm_cnt = 0
    cap = cv2.VideoCapture(video_path)

    size = (cap.get(3), cap.get(4))
    IP = ImageProcessor(size)

    while True:
        frm_cnt += 1
        ret, frame = cap.read()

        if ret:
            key_point, img, img_black = IP.process_img(frame)
            if len(img) > 0 and len(key_point) > 0:
                cv2.imshow("result", img)
                cv2.waitKey(2)


            else:
                cv2.imshow("result", img)
                cv2.waitKey(2)



        else:
            break


if __name__ == "__main__":
    run(video_path)
