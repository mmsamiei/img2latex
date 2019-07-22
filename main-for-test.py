from utils.CharIndex import *
from utils.Img2LatexDataset import *
from Model.model import *
from trainer.trainer import *
import json
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import os, os.path
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

chardict_file_addr = "./char_dict.json"
tokendict_file_addr = "./token_dict.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-base', dest='base_model')
    parser.add_argument('-addr', dest='file_addr')
    parsed_args = parser.parse_args()


    hidden_size = 512
    emb_size = 30
    with open(tokendict_file_addr) as handler:
        token_dict = json.load(handler)
    vocab_size = len(token_dict.keys())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = CNNEncoder()
    rnn_encoder = RNNEncoder(512, hidden_size)
    decoder = RNNDecoder(hidden_size, emb_size, vocab_size)
    img2seq = Img2seq(encoder, rnn_encoder, decoder, dev).to(dev)

    state_dict = torch.load(parsed_args.base_model)
    img2seq.load_state_dict(state_dict)

    char_index = CharIndex()
    char_index.load('.')
    token_index = TokenIndex()
    token_index.load('.')

    image = io.imread(parsed_args.file_addr)
    image = torch.from_numpy(image).float().to(dev).unsqueeze(0).unsqueeze(0)
    result = img2seq.greedy_inference(image, token_dict['<start>'], 120)
    str = token_index.translate_to_token(result[0])
    print(str)
