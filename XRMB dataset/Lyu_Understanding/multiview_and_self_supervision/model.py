import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# Gradient Reversal Layer, https://github.com/hanzhaoml/MDAN.git
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(self, inputs):
        return inputs

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input


# Multi-Layer Perceptrons
class MLP(nn.Module):
    def __init__(self, input_d, structure, output_d, dropprob=0.0):
        super(MLP, self).__init__()
        self.net = nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropprob)
        struc = [input_d] + structure + [output_d]

        for i in range(len(struc)-1):
            self.net.append(nn.Linear(struc[i], struc[i+1]))

    def forward(self, x):
        for i in range(len(self.net)-1):
            x = F.relu(self.net[i](x))
            x = self.dropout(x)

        # For the last layer
        y = self.net[-1](x)

        return y


# MiniMaxCCA model for independence regularization
class MMDCCA(nn.Module):
    def __init__(self, view1_dim, view2_dim, phi_size, tau_size, latent_dim=1):
        super(MMDCCA, self).__init__()
        self.phi = MLP(view1_dim, phi_size, latent_dim)
        self.tau = MLP(view2_dim, tau_size, latent_dim)
        # gradient reversal layer
        self.grl1 = GradientReversalLayer()
        self.grl2 = GradientReversalLayer()

    def forward(self, x1, x2):
        y1 = self.phi(self.grl1.apply(x1))
        y2 = self.tau(self.grl2.apply(x2))

        return y1, y2


# Multiview Deterministic AutoEncoder
class DAE(nn.Module):
    def __init__(self, z_dim=10, c_dim=2, i_shape=[50]*3, p=0):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_view = 3
        self.i_shape = i_shape
        self.mid_shape = 512
        self.p = 0
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(self.num_view):
            self.encoder.append(enc_model(i_shape[i], z_dim, c_dim[i], self.mid_shape, self.p))
            self.decoder.append(dec_model(i_shape[i], z_dim, c_dim[i], self.mid_shape, self.p))

    def encode(self, x):
        shared = []
        private = []
        for i in range(self.num_view):
            tmp = self.encoder[i](x[i])
            shared.append(tmp[0])
            private.append(tmp[1])

        return shared, private

    def decode(self, s, p):
        recons = []
        for i in range(self.num_view):
            tmp = self.decoder[i](s[i], p[i])
            recons.append(tmp)

        return recons

    def forward(self, x):
        shared, private = self.encode(x)
        recons = self.decode(shared, private)

        return shared, private, recons


class enc_model(nn.Module):
    def __init__(self, i_shape, z_dim, c_dim, mid_shape, p):
        super(enc_model, self).__init__()
        self.i_shape = i_shape
        self.mid_shape = mid_shape
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.act = nn.ReLU()
        self.drp = nn.Dropout(p=p)
        self.one = nn.Linear(self.i_shape, self.mid_shape)
        self.sec = nn.Linear(self.mid_shape, self.mid_shape)
        self.thr = nn.Linear(self.mid_shape, self.mid_shape)
        self.S = nn.Linear(self.mid_shape, self.z_dim)
        self.P = nn.Linear(self.mid_shape, self.c_dim)

    def forward(self, x):
        tmp = self.sec(self.drp(self.act(self.one(x))))
        tmp = self.thr(self.drp(self.act(tmp)))
        shared = self.S(tmp)
        private = self.P(tmp)
        return shared, private


class dec_model(nn.Module):
    def __init__(self, i_shape, z_dim, c_dim, mid_shape, p):
        super(dec_model, self).__init__()
        self.i_shape = i_shape
        self.mid_shape = mid_shape
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.act = nn.ReLU()
        self.drp = nn.Dropout(p=p)
        self.de1 = nn.Linear(self.z_dim+self.c_dim, self.mid_shape)
        self.de2 = nn.Linear(self.mid_shape, self.mid_shape)
        self.de3 = nn.Linear(self.mid_shape, self.mid_shape)
        self.de4 = nn.Linear(self.mid_shape, self.i_shape)

    def forward(self, s, p):
        y = torch.cat((s, p), 1)
        y = self.drp(self.act(self.de1(y)))
        y = self.de3(self.drp(self.act(self.de2(y))))
        y = self.de4(self.drp(self.act(y)))
        return y
