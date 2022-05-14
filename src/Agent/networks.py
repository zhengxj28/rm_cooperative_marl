import torch
from torch import nn, optim
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, state_size, embedding_size, one_hot=True):
        super(QNet, self).__init__()
        self.one_hot = one_hot
        self.embedding = nn.Linear(input_dim*state_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if self.one_hot:
            ones = torch.eye(state_size)
            x = torch.stack([ones.index_select(0, x[i]) for i in range(x.shape[0])])
            x = x.view(x.shape[0], -1)
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    x = [[1, 2, 3], [4, 5, 6]]
    x = torch.tensor(x, dtype=torch.int64)
    
    input_dim = 3
    output_dim = 8
    hidden_dim = 128
    embedding_size = 32
    state_size = 10

    q_net = QNet(input_dim, hidden_dim, output_dim, state_size, embedding_size)
    output = q_net(x)
    print(output)
