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
chardict_file_addr = "./char_dict.json"
tokendict_file_addr = "./token_dict.json"


if __name__ =="__main__":
    print("hi")
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

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(range(1,128)))
    validation_dataloader = DataLoader(validation_transformed_dataset, batch_size=batch_size, drop_last=True, sampler=SubsetRandomSampler(range(1,100)))
    hidden_size = 256
    emb_size = 10
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
    trainer.init_weights()
    print("number of model parameters! : ", trainer.count_parameters())
    trainer.train(9)


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

