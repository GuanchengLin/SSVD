import dgl
import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn
import pandas as pd
from typing import Union
import torch.nn.functional as f
import numpy as np
from torch.autograd import Function

from dgl.nn.pytorch.conv.gatedgraphconv import GatedGraphConv

class ReVeal(nn.Module):
    def __init__(self, input_channels=120, hidden_channels=200, max_edge_types=2, num_layers=8):
        super(Devign, self).__init__()
        self.inp_dim = input_channels
        self.out_dim = hidden_channels
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_layers
        self.ggnn = GatedGraphConv(in_feats=input_channels, out_feats=hidden_channels,
                                   n_steps=num_layers, n_etypes=max_edge_types)
        self.conv_l1 = torch.nn.Conv1d(hidden_channels, hidden_channels, 3)
        self.maxpool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2 = torch.nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.maxpool2 = torch.nn.MaxPool1d(2, stride=2)

        self.concat_dim = input_channels + hidden_channels
        self.conv_l1_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = torch.nn.MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = torch.nn.Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = torch.nn.MaxPool1d(2, stride=2)

        self.mlp_z = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=self.concat_dim, out_features=256)
        )
        self.mlp_y = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=hidden_channels, out_features=256)
        )
        self.dropout = nn.Dropout(p=0.3)
        self.out_head = MetricLearningModel(256, 128)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_types: torch.Tensor):
        edge_list = edge_index.cpu().tolist()
        outputs = self.ggnn(dgl.graph((edge_list[0], edge_list[1]), num_nodes=x.shape[0],
                                      device=torch.device('cuda') if torch.cuda.is_available() else torch.device(
                                          'cpu')), x, edge_types)
        c_i = torch.cat((x, outputs), axis=1)
        z_node_dim = c_i.shape[1]

        c_i = c_i.view(-1, 400, z_node_dim)
        h_i = outputs
        h_i_node_dim = h_i.shape[1]
        h_i = h_i.view(-1, 400, h_i_node_dim)

        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        before_avg = torch.mul(self.mlp_y(Y_2), self.mlp_z(Z_2))
        avg = before_avg.mean(dim=1)
        result, embedding, pseudo_out, worst_out = self.out_head(self.dropout(avg))
        output = {'logits': result, 'pseudo_out': pseudo_out, 'worst_out': worst_out}
        output['hidden_state'] = embedding
        
        return output


class MetricLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.2, aplha=0.5):
        super(MetricLearningModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.internal_dim = int(hidden_dim / 2)
        self.dropout_p = dropout_p
        self.alpha = aplha
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p)
        )
        self.feature = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.internal_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(in_features=self.internal_dim, out_features=self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_p),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=2),
        )

        self.pseudo_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.hidden_dim * 2, 2)
        )

        self.worst_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, 2)
        )

        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=False)

    def extract_feature(self, x):
        out = self.layer1(x)
        out = self.feature(out)
        return out

    def forward(self, x):
        h_a = self.extract_feature(x)
        y_a = self.classifier(h_a)
        pseudo_out = self.pseudo_head(h_a)

        f_adv = self.grl_layer(h_a)

        worst_out = self.worst_head(f_adv)

        return y_a, h_a, pseudo_out, worst_out


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha=1.0, lo=0.0, hi=1., max_iters=1000., auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = np.float32(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff=1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.coeff, None

if __name__ == '__main__':
    from dataset import GraphDataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = GraphDataset('ssl_data/Devign/0.1/train.pkl', mode='train')
    loaderMy = DataLoader(dataset, batch_size=4, pin_memory=True,
                          num_workers=4, collate_fn=collate_fn)
    model = ReVeal()
    for i, data in enumerate(loaderMy):
        out = model(data['input_ids'], data['edge_index'], data['edge_type'])
        print(out)
        break
