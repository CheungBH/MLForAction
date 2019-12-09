from src.utils.utils import Utils


class Locator(object):
    def __init__(self, base=(320, 0)):
        self.base_location = base
        self.main_human_idx = 0

    def check_by_coord(self, humans):
        human_len = [len(humans[num].body_parts) for num in range(len(humans))]
        self.main_human_idx = humans[human_len.index(min(human_len))]

    def check_by_base_location(self, humans):
        distance = [Utils.cal_dis(humans[num], self.base_location) for num in range(len(humans))]
        self.main_human_idx = distance.index(min(distance))

    def locate_user(self, kps):
        try:
            humans = [kps[i]["keypoints"].tolist()[0] for i in range(len(kps))]
            self.check_by_base_location(humans)
        except KeyError:
            humans = [kps[i]["keypoints"] for i in range(len(kps))]
            self.check_by_coord(humans)
        return kps[self.main_human_idx]

    @staticmethod
    def detect_user(coord1, coord2, height):
        if coord1 - coord2 > 0.4 * height and coord2 > 0 and coord1 < height:
            return True
        else:
            return False
