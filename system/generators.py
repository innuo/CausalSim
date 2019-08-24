import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd

class ForwardGenerator(nn.Module):
    def __init__(self, variable_dict, causal_graph, options):
        super(ForwardGenerator, self).__init__()
        self.variable_dict = variable_dict
        self.causal_graph = causal_graph
        self.num_vars = len(self.variable_dict)
        
        mlist = []
        for i, v in enumerate(self.causal_graph.topo_sorted):
            mlist.append(MechanismNetwork(variable_dict[v]['full_parent_dim'], variable_dict[v]['dim'], options['hidden_dims'],
                                            variable_dict[v]['type'] == 'categorical'))
        self.module_list = nn.ModuleList(mlist)


    def forward(self, z, do_df=pd.DataFrame()):
        do_vars = list(do_df)
        x = Variable(torch.zeros(z.shape[0], self.num_vars))
        x_one_hot_dict = dict()

        for i, v in enumerate(self.causal_graph.topo_sorted):
            if v in do_vars:
                x[:,self.variable_dict[v]['id']] = do_df[v].values
            else:
                inp = z[:,self.variable_dict[v]['latent_ids']]
                for p in self.causal_graph.parents[v]:
                    parent_col = x[:,self.variable_dict[p]['id']]
                    parent_dim = self.variable_dict[p]['dim']
                    parent_col_tx = to_one_hot(parent_col, parent_dim) if self.variable_dict[p]['type'] == 'categorical' else parent_col.unsqueeze(1)
                    inp = torch.cat((inp, parent_col_tx), 1)

                pred, pred_one_hot = self.module_list[i](inp)
                x[:,self.variable_dict[v]['id']] = pred.squeeze()
                x_one_hot_dict[v] = pred_one_hot
        
        return x, x_one_hot_dict
    pass

def to_one_hot(y, n_dims):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

      
class LatentGenerator(nn.Module):
    def __init__(self, num_latents, x_dim, x_one_hot_dim, variable_dict, options):
        super(LatentGenerator, self).__init__()
        self.x_dim = x_dim
        self.x_one_hot_dim = x_one_hot_dim
        self.model = MechanismNetwork(x_dim + x_one_hot_dim, num_latents, options['hidden_dims'])
        self.variable_dict = variable_dict

    def forward(self, x):
        x_missing = x != x
        x[x_missing] = 0
        x_cat = x_missing.type(torch.FloatTensor)
        
        for v in self.variable_dict.keys():
            col = x[:,self.variable_dict[v]['id']]
            dim = self.variable_dict[v]['dim']
            col_tx = to_one_hot(col, dim) if self.variable_dict[v]['type'] == 'categorical' else col.unsqueeze(1)
            #col_tx = col_tx + torch.randn(col_tx.shape) * 0.1
            x_cat = torch.cat((x_cat, col_tx.type(torch.FloatTensor)), 1) 

        z,_ = self.model(x_cat)
        means = torch.mean(z, dim=0)
        stds = torch.std(z, dim=0) 
        z = z - means[None, :]
        z = z/stds[None, :]

        return z

class MechanismNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, categorical_output=False):
        super(MechanismNetwork, self).__init__()
        self.num_hidden_layers = len(hidden_dims)
        self.categorical_output = categorical_output
        lin_layers = []
        nonlin_layers = []
        bn_layers = []

        for i, h in enumerate(hidden_dims):
            if i ==0:
                inp = input_dim
            else:
                inp = hidden_dims[i-1]
            lin_layers.append(nn.Linear(inp, h))
            nonlin_layers.append(nn.Tanh())
            bn_layers.append(nn.BatchNorm1d(h))

        self.lin_layers = nn.ModuleList(lin_layers)
        self.nonlin_layers = nn.ModuleList(nonlin_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        for i in range(self.num_hidden_layers):
            x = self.bn_layers[i](self.nonlin_layers[i](self.lin_layers[i](x)))
 
        pred = self.output_layer(x)   
        if self.categorical_output:
            _, y = torch.max(pred, 1) #TODO: does this work?
            return y, pred
        else:
            return pred, None
