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

    batch_size = 128
    transformed_dataset = Img2LatexDataset("./Dataset/images/images_train", "./Dataset/formulas/train_formulas.txt",
                                           transform=transforms.Compose([
                                               Rescale((60, 400)),
                                               ToTensor("./Dataset/formulas/train_formulas.txt", "token_dict.json",
                                                        "token")
                                           ]))

    validation_transformed_dataset = Img2LatexDataset("./Dataset/images/images_validation",
                                                      "./Dataset/formulas/validation_formulas.txt",
                                                      transform=transforms.Compose([
                                                          Rescale((60, 400)),
                                                          ToTensor("./Dataset/formulas/validation_formulas.txt",
                                                                   "token_dict.json", "token")
                                                      ]))

    test_transformed_dataset = Img2LatexDataset("./Dataset/images/images_test",
                                                      "./Dataset/formulas/train_formulas.txt",
                                                      transform=transforms.Compose([
                                                          Rescale((60, 400)),
                                                          ToTensor("./Dataset/formulas/train_formulas.txt",
                                                                   "token_dict.json", "token")
                                                      ]))

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(range(0, 10000)))
    validation_dataloader = DataLoader(validation_transformed_dataset, batch_size=batch_size, drop_last=True)
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

    train = False
    if train:
        #trainer.pretrain_encoders(5)
        #trainer.pretrain_cnn(5)
        #trainer.pretrain_encoders(1)
        #trainer.pretrain_decoder(5)
        train_loss, valid_loss = trainer.train(60)
        print("train loss is : \n {}".format(train_loss))
        print("valid loss is : \n {}".format(valid_loss))

        fig, ax = plt.subplots()
        ax.plot(train_loss)
        ax.plot(valid_loss)
        plt.show()
        fig.savefig("test.png")

    validation_dataloader_generation = DataLoader(test_transformed_dataset, batch_size=64, drop_last=False)

    str_list = []
    for i_batch, sample_batched in enumerate(validation_dataloader_generation):
        images = sample_batched['src'].to(dev)
        result = img2seq.greedy_inference(images, token_dict['<start>'], 120)
        for i, tensor in enumerate(result):
            str = token_index.translate_to_token(tensor)
            str_list.append(str)

    with open('predicted.txt', 'w') as f:
        for item in str_list:
            f.write("%s\n" % item)
