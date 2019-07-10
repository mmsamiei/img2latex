import pickle
import json
import os

class CharIndex():
    def __init__(self, file_addr):
        f = open(file_addr)
        formulas_str = f.readlines()
        chars = {"آ","پ","خ"}
        for formula in formulas_str:
            for char in formula:
                if char not in chars:
                    chars.add(char)
        self.char_dict = {k: i for i, k in enumerate(chars)}
        self.char_list = [None] * len(chars)
        for i, k in enumerate(self.char_dict):
            self.char_list[i] = k
        print(self.char_dict['خ'])
        print(self.char_dict['آ'])
        print(self.char_dict['پ'])

    def save(self, dir_addr):
        handler1 = open(os.path.join(dir_addr, "char_dict.json"), "w")
        json.dump(self.char_dict, handler1, ensure_ascii=False)
        handler2 = open(os.path.join(dir_addr, "char_list.json"), "w")
        json.dump(self.char_list, handler2, ensure_ascii=False)

    def load(self, dir_addr):
        handler1 = open(os.path.join(dir_addr, "char_dict.json"), "r")
        self.char_dict = json.load(self.char_dict, handler1, ensure_ascii=False)
        handler2 = open(os.path.join(dir_addr, "char_list.json"), "r")
        self.char_list = json.load(self.char_list, handler2, ensure_ascii=False)

    def translate_to_char(self, tensor):
        ## tensor = [max_len]
        str = ""
        for i,v in enumerate(tensor):
            str = str + self.char_list[v]
            if v == self.char_dict['پ']:
                break
        return str[1:-1]


a = CharIndex("./Dataset/formulas/train_formulas.txt")
a.save(".")