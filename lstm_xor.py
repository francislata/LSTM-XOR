import torch
import torch.nn as nn

class LSTMXOR(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMXOR, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.hidden_size = hidden_size
    
    def forward(self, x):
        hidden_states = torch.zeros([1, x.size(0), self.hidden_size]), torch.zeros([1, x.size(0), self.hidden_size])

        x = x.unsqueeze(2)
        y, hidden_states = self.lstm(x, hidden_states)
        y = y.contiguous().view(y.size(0), -1)
        y = self.fc(y)
        y = y.squeeze()

        return y
