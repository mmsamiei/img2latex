# from src.Img2LatexDataset import *
# from src.CharIndex import *
# from src.TokenIndex import *
from utils.TokenIndex import *
from utils.CharIndex import *
from utils.Img2LatexDataset import *
from Model.model import *
from trainer.trainer import *
import json
from torch.utils.data.sampler import SubsetRandomSampler
import argparse
import matplotlib.pyplot as plt

chardict_file_addr = "./char_dict.json"
tokendict_file_addr = "./token_dict.json"

if __name__ == "__main__":
    print("salam")
    parser = argparse.ArgumentParser()
    parser.add_argument('-base', dest='base_model')
    parsed_args = parser.parse_args()

    batch_size = 64
    transformed_dataset = Img2LatexDataset("./Dataset/images/images_train", "./Dataset/formulas/train_formulas.txt",
                                           transform=transforms.Compose([
                                               Rescale((200, 30)),
                                               ToTensor("./Dataset/formulas/train_formulas.txt", "token_dict.json",
                                                        "token")
                                           ]))

    validation_transformed_dataset = Img2LatexDataset("./Dataset/images/images_validation",
                                                      "./Dataset/formulas/validation_formulas.txt",
                                                      transform=transforms.Compose([
                                                          Rescale((200, 30)),
                                                          ToTensor("./Dataset/formulas/validation_formulas.txt",
                                                                   "token_dict.json", "token")
                                                      ]))

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, drop_last=True,
                            sampler=SubsetRandomSampler(range(1, 100)))
    validation_dataloader = DataLoader(validation_transformed_dataset, batch_size=batch_size, drop_last=True,
                                       sampler=SubsetRandomSampler(range(1, 100)))
    hidden_size = 128
    encoder_zip_size = 64
    emb_size = 20
    with open(tokendict_file_addr) as handler:
        token_dict = json.load(handler)
    vocab_size = len(token_dict.keys())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = CNNEncoder(hidden_size, encoder_zip_size).double()
    decoder = RNNDecoder(hidden_size, emb_size, vocab_size).double()
    img2seq = Img2seq(encoder, decoder, dev).double().to(dev)

    char_index = CharIndex()
    char_index.load('.')
    token_index = TokenIndex()
    token_index.load('.')

    trainer = Trainer(img2seq, dataloader, validation_dataloader, token_dict['<pad>'], dev)

    if parsed_args.base_model is None:
        trainer.init_weights()
    else:
        state_dict = torch.load(parsed_args.base_model)
        img2seq.load_state_dict(state_dict)

    print("number of model parameters! : ", trainer.count_parameters())

    train = True
    if train:
        #trainer.pretrain(3)
        train_loss, valid_loss = trainer.train(50)
        print("train loss is : \n {}".format(train_loss))
        print("valid loss is : \n {}".format(valid_loss))

        fig, ax = plt.subplots()
        ax.plot(train_loss)
        ax.plot(valid_loss)
        plt.show()
        fig.savefig("test.png")

    validation_dataloader_generation = DataLoader(validation_transformed_dataset, batch_size=64, drop_last=False)

    str_list = []
    for i_batch, sample_batched in enumerate(validation_dataloader_generation):
        images = sample_batched['src'].to(dev)
        result = img2seq.greedy_inference(images, token_dict['<start>'], 70)
        for i, tensor in enumerate(result):
            str = token_index.translate_to_token(tensor)
            str_list.append(str)

    with open('predicted.txt', 'w') as f:
        for item in str_list:
            f.write("%s\n" % item)
