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
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.output_size = output_size
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
        temp = temp.view(temp.shape[0], temp.shape[1], -1)
        return temp

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
    # layers
        self.gru = nn.GRU(input_size, hidden_size, dropout=0.5)

    def forward(self, x, hidden):
        ### x = (seq_len, batch_size, input_size)
        #hidden = torch.zeros((1, x.shape[1], self.hidden_size)).to(self.device) ### (1, batch_size, hidden_size)
        _, hidden = self.gru(x, hidden)
        return hidden

class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, emb_size, vocab_size):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
    # layers
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, dropout=0.5)
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
    def __init__(self, encoder, rnn_encoder,  decoder, device):
        super(Img2seq, self).__init__()
        self.encoder = encoder
        self.rnn_encoder = rnn_encoder
        self.decoder = decoder
        self.device = device
        # middle layer
        self.middle_gru = nn.GRU(128, self.decoder.hidden_size, dropout=0.5)

    def forward(self, src, trg, teacher_forcing_rate = 0.99):
        # src = [batch_size, 1, 200, 30]
        # trg  = [trg_sent_len, batch_size]
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_dim = self.decoder.vocab_size
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_dim).to(self.device)
        encoder_result = self.encoder(src)  ### encoder_result = (batch_size, 128, 20)
        encoder_result = encoder_result.permute(2, 0, 1)  ### encoder_result = (20, batch_size, 128)

        hidden = torch.zeros((1, batch_size, self.rnn_encoder.hidden_size)).double().to(self.device)  ### (1, batch_size, hidden_size)
        hidden = self.rnn_encoder(encoder_result, hidden)

        # hidden = [1, batch_size, hid_dim]
        input = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
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

        encoder_result = self.encoder(src)  ### hidden = (batch_size, 128, 20)
        encoder_result = encoder_result.permute(2, 0, 1)
        ## TODO
        hidden = torch.zeros((1, batch_size, self.decoder.hidden_size)).double().to(self.device)
        _, hidden = self.middle_gru(encoder_result, hidden)

        # hidden = [1, batch_size, hid_dim]
        input = torch.LongTensor(batch_size).fill_(start_token_index).to(self.device)
        results[:, 0] = input
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
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

    encoder = CNNEncoder(zip_size=40).double()
    rnn_encoder = RNNEncoder(128, hidden_size).double()
    decoder = RNNDecoder(hidden_size, emb_size, vocab_size).double()
    img2seq = Img2seq(encoder, rnn_encoder, decoder, dev).double()

    src = torch.rand((batch_size,1,400,60)).double()
    trg = torch.LongTensor(40, batch_size).random_(0,vocab_size)
    res = img2seq(src, trg)
    print(res.shape)


    print("**** test inference part! ***")
    # handler1 = open("token_dict.json")
    # token_dict = json.load(handler1)
    # result = img2seq.greedy_inference(src, token_dict['<start>'], 40)
    # print(result.shape)
    # print(result)
