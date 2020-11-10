import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.frame = dataframe

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        item = self.frame.iloc[idx]
        return torch.LongTensor(item['word_list']), torch.tensor(item['Label'])

class AttractivenessRNN(nn.Module):
    def __init__(self, pretrained_weights, embedding_dim, hidden_dim, n_layers, padding_idx=437247, drop_prob=0.5):
        super(AttractivenessRNN, self).__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weights, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, int(np.floor(hidden_dim/2)))
        self.fc2 = nn.Linear(int(np.floor(hidden_dim/2)), 1)

    def forward(self, x, hidden):
        # Perform a forward pass of our model on some input and hidden state.
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.fc2(out)
        # print(out.size())

        out = out.view(batch_size, -1)
        out = out[:, -1]
        # print(out.size())

        # return last sigmoid output and hidden state
        return out, hidden

    def init_hidden(self, batch_size, train_on_gpu):
        # Initializes hidden state
        # (create two new tensors with sizes n_layers x batch_size x hidden_dim, initialized to zero, for hidden state and cell state of LSTM)
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
