import cv2
from src.processor import ImageProcessor
from utils.utils import Utils

IP = ImageProcessor()
body_parts = ["Nose", "Left eye", "Right eye", "Left ear", "Right ear", "Left shoulder", "Right shoulder", "Left elbow",
              "Right elbow", "Left wrist", "Right wrist", "Left hip", "Right hip", "Left knee", "Right knee",
              "Left ankle", "Right ankle"]
joint_ls = {"Left arm": [5, 7, 9], "Right arm": [6, 8, 10], "Left shoulder": [7, 5, 11], "Right shoulder": [8, 6, 12],
            "Left hip": [5, 11, 13], "Right hip": [6, 12, 14], "Left leg": [11, 13, 15], "Right leg": [12, 14, 16]}


class MLTester:
    def __init__(self):
        pass

    def run_img(self, inp, ispath=True):
        frame = cv2.imread(inp) if ispath else inp
        # waitkey = 100 if ispath else 2
        key_point, img, img_black = IP.process_img(frame)
        if len(img) > 0 and len(key_point) > 0:
            angle_ls = [Utils.get_angle(key_point[joint_ls[key][1]], key_point[joint_ls[key][0]], key_point[joint_ls[key][2]])
                        for key in list(joint_ls.keys())]


            ### predict the action

            cv2.imshow("result", img)
            cv2.waitKey(2)
        else:
            cv2.imshow("result", img)
            cv2.waitKey(2)

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


if __name__ == "__main__":
    path = ""
    if "avi" in path or "mp4" in path:
        MLTester().run_video(path)
    elif "jpg" in path or "png" in path or "jpeg" in path:
        MLTester().run_img(path)
    else:
        raise ValueError("Wrong input file format")
