# بسم الله الرحمن الرحیم


from torch import nn
import torch
import torch.nn.functional as F
import random
import json



class CNNEncoder(nn.Module):
    def __init__(self, output_size=300, zip_size = 20):
        super(CNNEncoder, self).__init__()
        self.zip_size = zip_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.dropout = nn.Dropout(p=0.5)
        self.output_size = output_size
        self.fc = nn.Linear(5120, self.output_size)
        self.fc_zip = nn.Linear(5120, zip_size)

    def forward(self, x):
        temp = x
        temp = F.relu(self.conv1(temp))
        temp = self.dropout2d(temp)
        temp = F.relu(self.conv2(temp))
        temp = self.dropout2d(temp)
        temp = self.pool1(temp)
        temp = F.relu(self.conv3(temp))
        temp = self.dropout2d(temp)
        temp = F.relu(self.conv4(temp))
        temp = self.dropout2d(temp)
        temp = self.pool2(temp)
        temp = temp.view(-1, 5120)
        temp = self.dropout(temp)
        zip = self.fc_zip(temp)
        temp = F.relu(self.fc(temp))
        return temp, zip

class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, emb_size, vocab_size):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
    # layers
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        # input = [batch_size]
        # hidden = [1, batch_size, hid_dim]
        temp = input.unsqueeze(0) # temp = [1, batch_size]
        temp = self.embedding(temp)  # temp = [1, batch_size, emb_dim]
        output, hidden = self.gru(temp, hidden)  # output = [1, batch_size, hid_dim]
        prediction = self.fc(output.squeeze(0)) # prediction = p[batch_size, hid_dim]
        return prediction, hidden

class Img2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Img2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_rate = 0.99):
        # src = [batch_size, 1, 200, 30]
        # trg  = [trg_sent_len, batch_size]
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_dim = self.decoder.vocab_size
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_dim).to(self.device)
        hidden, encoder_zip = self.encoder(src)
        hidden = hidden.unsqueeze(0)
        # hidden = [1, batch_size, hid_dim]
        input = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            ###
            hidden[0,:,:self.encoder.zip_size] = encoder_zip
            ###
            outputs[t] = output
            teacher_forcing = random.random() < teacher_forcing_rate
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_forcing else top1)
        return outputs

    def greedy_inference(self, src, start_token_index, max_len):
        # src = [batch_size, 1, 200, 30]
        batch_size = src.shape[0]
        max_len = max_len
        trg_vocab_dim = self.decoder.vocab_size
        # tensor to store decoder outputs
        results = torch.zeros(batch_size, max_len)
        hidden, encoder_zip = self.encoder(src)
        hidden = hidden.unsqueeze(0)
        # hidden = [1, batch_size, hid_dim]
        input = torch.LongTensor(batch_size).fill_(start_token_index).to(self.device)
        results[:, 0] = input
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            ###
            hidden[0, :, :self.encoder.zip_size] = encoder_zip
            ###
            top1 = output.max(1)[1]
            input = top1
            results[:, t] = top1
        return results





if __name__ =="__main__":
    hidden_size = 300
    emb_size = 20
    vocab_size = 50
    batch_size = 64
    dev = torch.device("cpu")

    encoder = CNNEncoder(zip_size=40)
    decoder = RNNDecoder(hidden_size, emb_size, vocab_size)
    img2seq = Img2seq(encoder, decoder, dev)

    src = torch.rand((batch_size,1,200,30))
    trg = torch.LongTensor(40, batch_size).random_(0,vocab_size)
    res = img2seq(src, trg)
    print(res.shape)


    print("**** test inference part! ***")
    handler1 = open("token_dict.json")
    token_dict = json.load(handler1)
    result = img2seq.greedy_inference(src, token_dict['<start>'], 40)
    print(result.shape)
    print(result)
