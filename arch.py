import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pickle
import pdb

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self, model_url):
        super(Net, self).__init__()
        self.model_url = model_url
        ext, cls = self.build_architecture()
        self.extract = nn.ModuleList(ext)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.classifier = nn.ModuleList(cls)


    def forward(self, input):

        out = input
        for layer in self.extract:
            out = layer(out)
        out = self.pool(out)
        out = out.view(out.size(0),-1)
        for layer in self.classifier:
            out = layer(out)
        return out
    
    def build_architecture(self):
        with open(self.model_url, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        parameters = data['params']
        extractor = []
        classifier = []
        BN_counter = 1
        BN_flag = False
        classifier_flag = False
        conv3 = False
        for i, p in enumerate(parameters):
            ndim = p.ndim
            print(p.shape)
            if ndim == 4:
                out_channel, in_channel, F, _ = p.shape
                if F == 11:
                    extractor += [nn.Conv2d(in_channel, out_channel, F, stride=4, padding=0, bias=False)]
                    # extractor[-1].weight.data = torch.from_numpy(self.rot90(p))
                    extractor[-1].weight.data = torch.from_numpy(np.array(p[:, :, ::-1, ::-1]))
                elif F==5:
                    extractor += [nn.Conv2d(in_channel, out_channel, F, stride=1, padding=2, bias=False)]
                    # extractor[-1].weight.data = torch.from_numpy(self.rot90(p))
                    extractor[-1].weight.data = torch.from_numpy(np.array(p[:, :, ::-1, ::-1]))
                elif F==3:
                    extractor += [nn.Conv2d(in_channel, out_channel, F, stride=1, padding=1, bias=False)]
                    # extractor[-1].weight.data = torch.from_numpy(self.rot90(p))
                    extractor[-1].weight.data = torch.from_numpy(np.array(p[:, :, ::-1, ::-1]))
                    conv3 = True
                # extractor += [nn.ReLU()]
            elif ndim == 2:
                in_channel, out_channel = p.shape
                classifier += [nn.Linear(in_channel, out_channel, bias=False)]
                classifier[-1].weight.data = torch.from_numpy(np.array(p.T))
                # classifier += [nn.ReLU()]
                classifier_flag = True
            elif ndim == 1:
                if BN_counter == 4:
                    BN_flag = True
                    BN_counter = 1
                else:
                    BN_counter = BN_counter + 1
                if BN_flag == True:
                    in_channel = p.shape
                    if not classifier_flag:
                        extractor += [nn.BatchNorm2d(in_channel)]
                        extractor[-1].weight.data = torch.from_numpy(parameters[i-2])
                        extractor[-1].bias.data = torch.from_numpy(parameters[i-3])
                        extractor[-1].running_mean = torch.from_numpy(parameters[i-1])
                        # extractor[-1].running_var = torch.from_numpy(parameters[i])
                        extractor[-1].running_var = torch.from_numpy((1./(parameters[i]**2)) - 1e-4)
                        extractor += [nn.ReLU()]
                        if not conv3:
                            extractor += [nn.MaxPool2d(3,stride=2)]
                    else:
                        classifier += [nn.BatchNorm1d(in_channel)]
                        classifier[-1].weight.data = torch.from_numpy(parameters[i-2])
                        classifier[-1].bias.data = torch.from_numpy(parameters[i-3])
                        classifier[-1].running_mean = torch.from_numpy(parameters[i-1])
                        # extractor[-1].running_var = torch.from_numpy(parameters[i])
                        classifier[-1].running_var = torch.from_numpy((1./(parameters[i]**2)) - 1e-4)
                        classifier += [nn.ReLU()]
                    BN_flag = False
        return extractor, classifier

    def rot90(self, W):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = np.rot90(W[i, j], 2)
        return W
