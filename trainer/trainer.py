from torch import nn
import torch
from torch import optim
import time
import os
import datetime
from utils.TokenIndex import *


class Trainer:
    def __init__(self, model, dataloader, validation_dataloader, PAD_IDX, dev):
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.dev = dev

    def init_weights(self):
        for name, param in self.model.named_parameters():
            nn.init.uniform_(param.data, -0.09, 0.09)

    def count_parameters(self):
        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        return sum(params)

    def train_one_epoch(self, clip=1000):
        self.model.train() #This will turn on dropout (and batch normalization)
        epoch_loss = 0
        for i, sample_batched in enumerate(self.dataloader):
            src, trg = sample_batched['src'], sample_batched['trg']
            src = src.to(self.dev)
            trg = trg.to(self.dev)
            # src = [batch_size, 1, 200, 30]
            # trg = [batch_Size, seq_len]
            self.optimizer.zero_grad()
            trg = trg.permute(1,0)
            # trg = [trg sent len, batch size]
            output = self.model(src, trg)
            # output = [trg sent len, batch size, output dim]
            trg = trg[1:].contiguous().view(-1)
            # trg = [(trg sent len - 1) * batch size]
            output = output[1:].view(-1, output.shape[-1])
            # output = [(trg sent len - 1) * batch size, output dim]
            loss = self.criterion(output, trg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss/len(self.dataloader)

    def train(self, N_epoch, save_period = 1):
        epoch_losses = []
        valid_losses = []
        directory_name = datetime.datetime.now().strftime('%Y-%m-%d--%H:%M:%S')
        os.makedirs('./saved_models/{}'.format(directory_name))

        for i_epoch in range(N_epoch):
            start_time = time.time()
            epoch_loss = self.train_one_epoch()
            epoch_losses.append(epoch_loss)
            valid_losses.append(self.evaluate())
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print("epoch {}, time elapse is {} mins {} seconds".format(i_epoch, epoch_mins, epoch_secs))
            print("train loss: {} \t valid loss: {}".format(epoch_loss, valid_losses[-1]))
            if (i_epoch + 1) % save_period == 0:
                temp_path = os.path.join('.', 'saved_models')
                temp_path = os.path.join(temp_path, directory_name)
                temp_path = os.path.join(temp_path, 'model-{}.pt'.format(i_epoch))
                torch.save(self.model.state_dict(), temp_path)
                print("model saved in: {}".format(temp_path))
            ## TODO
            self.inference_one_sample()
        return epoch_losses, valid_losses

    def pretrain(self, N_epoch):
        self.model.train()
        pretrain_optimizer = optim.Adam(self.model.decoder.parameters())
        for i_epoch in range(N_epoch):
            epoch_loss = 0
            for i, sample_batched in enumerate(self.dataloader):
                src, trg = sample_batched['src'], sample_batched['trg']
                src = src.to(self.dev)
                trg = trg.to(self.dev)
                # src = [batch_size, 1, 200, 30]
                # trg = [batch_Size, seq_len]
                pretrain_optimizer.zero_grad()
                trg = trg.permute(1, 0)
                # trg = [trg sent len, batch size]
                output = self.model(src, trg)
                # output = [trg sent len, batch size, output dim]
                trg = trg[1:].contiguous().view(-1)
                # trg = [(trg sent len - 1) * batch size]
                output = output[1:].view(-1, output.shape[-1])
                # output = [(trg sent len - 1) * batch size, output dim]
                loss = self.criterion(output, trg)
                loss.backward()
                pretrain_optimizer.step()
                epoch_loss += loss
            epoch_loss = epoch_loss / len(self.dataloader)
            print("pretrain epoch {} finished loss is {}".format(i_epoch, epoch_loss))


    def evaluate(self):
        self.model.eval() #This will turn off dropout (and batch normalization)
        epoch_loss = 0
        with torch.no_grad():
            for i, sample_batched in enumerate(self.validation_dataloader):
                src, trg = sample_batched['src'], sample_batched['trg']
                src = src.to(self.dev)
                trg = trg.to(self.dev)
                # src = [batch_size, 1, 200, 30]
                # trg = [batch_Size, seq_len]
                trg = trg.permute(1, 0)
                # trg = [trg sent len, batch size]
                output = self.model(src, trg, 0)
                trg = trg[1:].contiguous().view(-1)
                output = output[1:].view(-1, output.shape[-1])
                loss = self.criterion(output, trg)
                epoch_loss += loss.item()
        return  epoch_loss/len(self.validation_dataloader)

    def inference_one_sample(self):
        self.model.eval()
        token_index = TokenIndex()
        token_index.load('.')
        with torch.no_grad():
            sample_batched = next(iter(self.dataloader))
            images = sample_batched['src'].to(self.dev)
            ##TODO
            result = self.model.greedy_inference(images, 18, 50)
            str = token_index.translate_to_token(result[0])
            str = token_index.translate_to_token(result[1])
            print("generated sample: {}".format(str))
        return



    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs