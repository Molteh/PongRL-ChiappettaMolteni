import torch
import torch.nn.functional as F


class PongNN(torch.nn.Module):

    # Our batch shape for input x is (100, 100)

    def __init__(self):
        super(PongNN, self).__init__()

        # 10000 input features, 100 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(100 * 100, 100)

        # 100 input features, 1 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(100, 1)

    def forward(self, x):

        # Reshape data to input to the input layer of the neural net
        # Size changes from (100, 100) to (1, 10000)
        # Recall that the -1 infers this dimension from the other given dimension
        #x = x.view(-1, 100 * 100)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 100000) to (1, 100)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 100) to (1, 1)
        x = self.fc2(x)

        return x
