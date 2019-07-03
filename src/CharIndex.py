import pickle
import json
import os

class CharIndex():
    def __init__(self, file_addr):
        f = open(file_addr)
        formulas_str = f.readlines()
        chars = {"آ","ب"}
        for formula in formulas_str:
            for char in formula:
                if char not in chars:
                    chars.add(char)
        self.char_dict = {k: i for i, k in enumerate(chars)}
        self.char_list = [None] * len(chars)
        for i, k in enumerate(self.char_dict):
            self.char_list[i] = k

    def save(self, dir_addr):
        handler1 = open(os.path.join(dir_addr, "char_dict.json"), "w")
        json.dump(self.char_dict, handler1)
        handler2 = open(os.path.join(dir_addr, "char_list.json"), "w")
        json.dump(self.char_list, handler2)



#a = CharIndex("./Dataset/formulas/train_formulas.txt")
#a.save(".")