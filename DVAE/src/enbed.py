import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(12,64),
            nn.ReLU(),
            nn.Linear(64,256)
        )
    
    def forward(self,x):
        x = self.layers(x)
        return x

def sigmoid(z):
    z = z.detach().numpy()
    result = 1 / (1+np.exp(-z))
    return torch.tensor(result)


mlp = MLP()
y = [ 0,  9,  0,  7,  3,  5,  7, 10,  9,  1, 11,  0, 10,  2,  0, 10,  3,  0, 5, 10]
y = torch.tensor(y)
act_lable = F.one_hot(y,num_classes=12)
act_lable = act_lable.float()
a = mlp(act_lable)
a = sigmoid(a)

x= [ 0,  9,  0,  7,  3,  5,  7, 10,  9,  1, 11,  0, 10,  2,  0, 10,  3,  0, 5, 10]
x = torch.tensor(x)
label_embed = nn.Embedding(12,256)
b = label_embed(x)
b = sigmoid(b)

# sigma_label = label_embed(label2)[None]