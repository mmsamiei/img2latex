from torch import nn
import torch
from torch import optim
import time
class Trainer:
    def __init__(self, model, dataloader, validation_dataloader, PAD_IDX):
        self.model = model
        self.dataloader = dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    def init_weights(self):
        for name, param in self.model.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def count_parameters(self):
        params = [p.numel() for p in self.model.parameters() if p.requires_grad]
        return sum(params)

    def train_one_epoch(self, clip=10):
        self.model.train() #This will turn on dropout (and batch normalization)
        epoch_loss = 0
        for i, sample_batched in enumerate(self.dataloader):
            src, trg = sample_batched['src'], sample_batched['trg']
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
        return epoch_loss

    def train(self, N_epoch):
        epoch_losses = []
        valid_losses = []
        for i_epoch in range(N_epoch):
            start_time = time.time()
            epoch_loss = self.train_one_epoch()
            epoch_losses.append(epoch_loss)
            valid_losses.append(self.evaluate())
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            print("epoch {}, time elapse is {} seconds".format(i_epoch, epoch_secs))
            ## TODO

        print(epoch_losses)

    def evaluate(self):
        self.model.eval() #This will turn off dropout (and batch normalization)
        epoch_loss = 0
        with torch.no_grad():
            for i, sample_batched in enumerate(self.validation_dataloader):
                src, trg = sample_batched['src'], sample_batched['trg']
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

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

