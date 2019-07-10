import pickle
import json
import os

class TokenIndex():

    def __init__(self, file_addr=None):
        self.token_dict = dict()
        self.token_list = []
        if file_addr is not None:
            f = open(file_addr)
            formulas_str = f.readlines()
            tokens = {"<start>","<end>","<pad>"}
            for formula in formulas_str:
                for token in formula.strip().split():
                    if token not in tokens:
                        tokens.add(token)
            self.token_dict = {k: i for i, k in enumerate(tokens)}
            self.token_list = [None] * len(tokens)
            for i, k in enumerate(self.token_dict):
                self.token_list[i] = k
            print(self.token_dict['<start>'])
            print(self.token_dict['<end>'])
            print(self.token_dict['<pad>'])

    def save(self, dir_addr):
        handler1 = open(os.path.join(dir_addr, "token_dict.json"), "w")
        json.dump(self.token_dict, handler1, ensure_ascii=False)
        handler2 = open(os.path.join(dir_addr, "token_list.json"), "w")
        json.dump(self.token_list, handler2, ensure_ascii=False)

    def load(self, dir_addr):
        handler1 = open(os.path.join(dir_addr, "token_dict.json"), "r")
        self.token_dict = json.load(handler1)
        handler2 = open(os.path.join(dir_addr, "token_list.json"), "r")
        self.token_list = json.load(handler2)

    def translate_to_token(self, tensor):
        ## tensor = [max_len]
        str = ""
        for i,v in enumerate(tensor):
            str = str + self.token_list[int(v.item())]
            if int(v.item()) == self.token_dict['<end>']:
                break
        return str[1:-1]


if __name__ == "__main__":
    a = TokenIndex("./Dataset/formulas/train_formulas.txt")
    a.save(".")