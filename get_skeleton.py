from config import config
import cv2
import os
from src.processor import ImageProcessor
from angle_calculator import AngleCalculator


video_path = config.webcam_num

body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
joint_ls = {"Left arm": [5, 7, 9], "Right arm": [6, 8, 10], "Left shoulder": [7, 9, 11], "Right shoulder": [8, 10, 12]}
AC = AngleCalculator()


def run_video(video_path):
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


def run_folder(image_folder):
    os.makedirs(image_folder+"_ske", exist_ok=True)
    pic_num = 0
    img_ls = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    des_ls = [os.path.join(image_folder+"_ske", img_name) for img_name in os.listdir(image_folder)]
    IP = ImageProcessor()

    for idx, img in enumerate(img_ls):
        image = cv2.imread(img)
        key_point, img, img_black = IP.process_img(image)
        cv2.imwrite(des_ls[idx], img_black)
        pic_num += 1
        print("Finish processing pic {}".format(pic_num))


if __name__ == "__main__":
    run_folder("Video/drown/normal_im")
