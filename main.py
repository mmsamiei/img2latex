from src.Img2LatexDataset import *
from Model.model import *
from trainer.trainer import *
import json
chardict_file_addr = "./char_dict.json"


if __name__ =="__main__":
    print("hi")
    batch_size = 8
    transformed_dataset = Img2LatexDataset("./Dataset/images/images_train","./Dataset/formulas/train_formulas.txt",
                                                transform=transforms.Compose([
                                                    Rescale((200, 30)),
                                                    ToTensor("./Dataset/formulas/train_formulas.txt", "char_dict.json")
                                                ]))

    validation_transformed_dataset = Img2LatexDataset("./Dataset/images/images_validation", "./Dataset/formulas/validation_formulas.txt",
                                           transform=transforms.Compose([
                                               Rescale((200, 30)),
                                               ToTensor("./Dataset/formulas/validation_formulas.txt", "char_dict.json")
                                           ]))

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, drop_last=True)
    validation_dataloader = DataLoader(validation_transformed_dataset, batch_size=batch_size, drop_last=True)
    hidden_size = 300
    emb_size = 20
    with open(chardict_file_addr) as handler:
        char_dict = json.load(handler)
    vocab_size = len(char_dict.keys())
    dev = torch.device("cpu")

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dev = torch.device('cuda')
    encoder = CNNEncoder().double()
    decoder = RNNDecoder(hidden_size, emb_size, vocab_size).double()
    ## TODO double!!!
    img2seq = Img2seq(encoder, decoder, dev).to(dev)

    trainer = Trainer(img2seq, dataloader, validation_dataloader, char_dict['خ'], dev)
    trainer.init_weights()
    print("number of model parameters! : ", trainer.count_parameters())
    trainer.train(50)

    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch)
    #     images = sample_batched['src']
    #     formulas = sample_batched['trg']
    #     formulas = formulas.permute(1,0)
    #     print(images.shape)
    #     print(formulas.shape)
    #     img2seq(images, formulas)

