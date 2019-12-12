from src.utils.utils import Utils


class AngleCalculator(object):
    def __init__(self):
        self.cal = Utils.get_angle

    def cal_angles(self, *coords):
        angles = []
        for coord in coords:
            angle = self.cal(coord[0], coord[1], coord[2])
            angles.append(angle)
        return angles


if __name__ == '__main__':
    AC = AngleCalculator()
    res = AC.cal_angles([(0, 0), (0, 1), (1, 0)], [(0, 1), (0, 0), (1, 0)])
    print(res)