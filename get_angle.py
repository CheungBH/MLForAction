import cv2
from detect import ImageProcessor
import csv
import os
from utils.utils import Utils

IP = ImageProcessor()
body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
joint_ls = {"Left arm": [5, 7, 9], "Right arm": [6, 8, 10], "Left shoulder": [7, 5, 11], "Right shoulder": [8, 6, 12],
            "Left hip": [5, 11, 13], "Right hip": [6, 12, 14], "Left leg": [11, 13, 15], "Right leg": [12, 14, 16]}


class AngleCSVWriter:
    def __init__(self, label, action, csvfile):
        self.csv_file = csvfile
        self.label = label
        self.action = action

    def run_img(self, inp, ispath=True):
        frame = cv2.imread(inp) if ispath else inp
        waitkey = 1000 if ispath else 2
        key_point, img, _ = IP.process_single_person(frame)
        if len(img) > 0 and len(key_point) > 0:
            angle_ls = [Utils.get_angle(key_point[joint_ls[key][1]], key_point[joint_ls[key][0]], key_point[joint_ls[key][2]])
                        for key in list(joint_ls.keys())]
            angle_ls += [self.label, self.action]
            self.csv_file.writerow(angle_ls)
            cv2.imshow("result", img)
            cv2.waitKey(waitkey)
        else:
            cv2.imshow("result", img)
            cv2.waitKey(waitkey)

    def run_video(self, video):
        frm_cnt = 0
        cap = cv2.VideoCapture(video)
        while True:
            frm_cnt += 1
            ret, frame = cap.read()
            if ret:
                self.run_img(frame, ispath=False)
            else:
                break

    def run_img_folder(self, folder):
        imgs = [os.path.join(folder, filename) for filename in os.listdir(folder)]
        for img in imgs:
            self.run_img(img)

    def run_video_folder(self, folder):
        videos = [os.path.join(folder, filename) for filename in os.listdir(folder)]
        for video in videos:
            self.run_video(video)


if __name__ == "__main__":
    # if run image folders
    f = open('yoga.csv', 'w', encoding='utf-8', newline='\n')
    dir_src = "Video/yoga"
    dir_ls = [os.path.join(dir_src, sub_dir) for sub_dir in os.listdir(dir_src)]
    label_dict = ["boat", "cobra", "chair", "camel", "triangle"]   # 一个文件夹对应一个label
    assert len(dir_ls) == len(label_dict), "The length of label and directory are not equal!"
    for idx, dir in enumerate(dir_ls):
        AngleCSVWriter(idx, label_dict[idx], csv.writer(f)).run_img_folder(dir)
    f.close()


    # if run image
    # f = open('yoga.csv', 'w', encoding='utf-8', newline='\n')
    # label, idx = "boat", 0 #image label and idx
    # img_path = "Video/yoga_img.jpg"
    # AngleCSVWriter(idx, label, csv.writer(f)).run_img(img_path)
    # f.close()

    # if run video
    # f = open('yoga.csv', 'w', encoding='utf-8', newline='\n')
    # label, idx = "boat", 0 #image label and idx
    # video_path = "Video/yoga_video.mp4"
    # AngleCSVWriter(idx, label, csv.writer(f)).run_video(video_path)
    # f.close()
