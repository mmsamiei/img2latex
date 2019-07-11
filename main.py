#from src.Img2LatexDataset import *
#from src.CharIndex import *
#from src.TokenIndex import *
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


if __name__ =="__main__":
    print("salam")
    parser = argparse.ArgumentParser()
    parser.add_argument('-base', dest='base_model')
    parsed_args = parser.parse_args()


    batch_size = 64
    transformed_dataset = Img2LatexDataset("./Dataset/images/images_train","./Dataset/formulas/train_formulas.txt",
                                                transform=transforms.Compose([
                                                    Rescale((200, 30)),
                                                    ToTensor("./Dataset/formulas/train_formulas.txt", "token_dict.json", "token")
                                                ]))

    validation_transformed_dataset = Img2LatexDataset("./Dataset/images/images_validation", "./Dataset/formulas/validation_formulas.txt",
                                           transform=transforms.Compose([
                                               Rescale((200, 30)),
                                               ToTensor("./Dataset/formulas/validation_formulas.txt", "token_dict.json", "token")
                                           ]))

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(range(1,10000)))
    validation_dataloader = DataLoader(validation_transformed_dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(range(1,1000)))
    hidden_size = 256
    emb_size = 20
    with open(tokendict_file_addr) as handler:
        token_dict = json.load(handler)
    vocab_size = len(token_dict.keys())
    dev = torch.device("cpu")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device('cuda')
    encoder = CNNEncoder(hidden_size).double()
    decoder = RNNDecoder(hidden_size, emb_size, vocab_size).double()
    ## TODO double!!!
    img2seq = Img2seq(encoder, decoder, dev).to(dev)

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

    trainer.pretrain(20)
    print("number of model parameters! : ", trainer.count_parameters())
    train_loss, valid_loss = trainer.train(100)

    print("train loss is : \n {}".format(train_loss))
    print("valid loss is : \n {}".format(valid_loss))

    fig, ax = plt.subplots()
    ax.plot(train_loss)
    ax.plot(valid_loss)
    plt.show()
    fig.savefig("test.png")

    for i_batch, sample_batched in enumerate(dataloader):
         images = sample_batched['src'].to(dev)
         result = img2seq.greedy_inference(images, token_dict['<start>'], 50)
         str = token_index.translate_to_token(result[0])
         str = token_index.translate_to_token(result[1])
         print(str)
    # for i_batch, sample_batched in enumerate(dataloader):
    #      print(i_batch)
    #      images = sample_batched['src']
    #      formulas = sample_batched['trg']
    #      formulas = formulas.permute(1,0)
    #      print(images.shape)
    #      print(formulas.shape)
    #      img2seq(images, formulas)
