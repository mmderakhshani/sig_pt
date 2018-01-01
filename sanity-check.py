import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as V
import numpy as np
import pdb
from arch import Net as Model
import torch.nn.functional as F

model_url = './signetf_lambda0.95.pkl'
num_class = 115

# Model and Optimizer definition
model = Model(model_url, num_class)
print(model)
model.eval()

input_np = np.load('./data/input.npy')
train = np.load('./data/np1.npy')
test = np.load('./data/np2.npy')

inp = V(torch.from_numpy(input_np).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0))
output = model(inp)
output_np = output.data.numpy()

pdb.set_trace()
print("Sum of Absolute Error: {}".format(np.abs(output_np - test).sum()))