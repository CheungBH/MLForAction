

class Locator(object):
    def __init__(self, nums):
        self.chosen_idx = nums

    def locate_user(self, id2ske, id2bbox):
        if self.chosen_idx == "all":
            return id2ske, id2bbox
        else:
            if type(self.chosen_idx) == list:
                chosen_ske, chosen_box = {}, {}
                for key in id2ske.keys():
                    if key in self.chosen_idx:
                        chosen_ske[key] = id2ske[key]
                        chosen_box[key] = id2bbox[key]
                return chosen_ske, chosen_box
            else:
                raise TypeError("Your input should be a list of numbers of a string 'all'")