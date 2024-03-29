from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import numpy as np
from skimage import io, transform
import json
from torch.utils.data.sampler import SubsetRandomSampler

class Img2LatexDataset(Dataset):

    def __init__(self, img_dir, formula_file=None, transform=None):
        self.img_dir = img_dir
        self.formula_file = formula_file
        self.transform = transform
        f = open(formula_file)
        self.formulas = f.readlines()

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        img_addr_idx = os.path.join(self.img_dir, str(idx)+".png")
        image = io.imread(img_addr_idx)
        formula = self.formulas[idx]
        sample = {'image': image, 'formula':formula}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, formula = sample['image'], sample['formula']
        h,w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'formula': formula}


class ToTensor(object):

    def __init__(self, str_file_addr, index_file_addr, char_or_token="char"):
        self.char_or_token = char_or_token  ### string -  char or token
        f = open(str_file_addr)
        formulas = f.readlines()
        if char_or_token == "char":
            lens = [len(formula) for formula in formulas]
        elif char_or_token =="token":
            lens = [len(formula.strip().split()) for formula in formulas]
        self.max_len = max(lens) + 2
        with open(index_file_addr) as handler:
            self.char_dict = json.load(handler)

    def __call__(self, sample):
        if(self.char_or_token == "char"):
            image, formula = sample['image'], sample['formula']
            formula = "آ" + formula + "پ"
            formula = formula + "خ" * (self.max_len - len(formula))
            formula_tensor = torch.LongTensor((self.max_len))
            for i, c in enumerate(formula):
                formula_tensor[i] = self.char_dict[c]
            return {'src':torch.from_numpy(image).unsqueeze(0), 'trg':formula_tensor}
        elif(self.char_or_token == "token"):
            image, formula = sample['image'], sample['formula']
            formula = " <start> " + formula + " <end> "
            formula = formula + " <pad> " * (self.max_len - len(formula.strip().split()))
            formula_tensor = torch.LongTensor((self.max_len))
            for i, c in enumerate(formula.strip().split()):
                if c not in self.char_dict:
                    formula_tensor[i] = self.char_dict['<UNK>']
                    continue
                formula_tensor[i] = self.char_dict[c]
            return {'src': torch.from_numpy(image).unsqueeze(0).float(), 'trg': formula_tensor}

if __name__ == "__main__":
    print("hi")
    transformed_dataset = Img2LatexDataset(".././Dataset/images/images_train",".././Dataset/formulas/train_formulas.txt",
                                            transform=transforms.Compose([
                                                Rescale((200, 30)),
                                                ToTensor(".././Dataset/formulas/train_formulas.txt", "../token_dict.json", "token")
                                            ]))
    sampler = SubsetRandomSampler(range(0,1000))
    dataloader = DataLoader(transformed_dataset, batch_size=64, drop_last=True, num_workers=1, sampler=sampler)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        images = sample_batched['src']
        formulas = sample_batched['trg']
        print(images.shape)
        print(formulas.shape)
        print(formulas[0])