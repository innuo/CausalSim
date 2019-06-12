import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd

class ForwardGenerator(nn.Module):
    def __init__(self, variable_dict, causal_graph, options):
        self.variable_dict = variable_dict
        self.causal_graph = causal_graph
        self.num_vars = len(self.variable_dict)
        mdict = {}
        for v in variable_dict.keys():
            mdict[v] = MechanismNetwork(variable_dict[v]['full_parent_dim'], 1, options['hidden_dims'],
                                            variable_dict[v]['type'] == 'categorical')
        self.module_dict = nn.ModuleDict(mdict)

    def forward(self, z, do_df=pd.DataFrame()):
        do_vars = list(do_df)
        x = Variable(torch.zeros(z.shape[0], self.num_vars))
        for v in self.causal_graph.topo_sorted:
            if v in do_vars:
                x[:,self.variable_dict[v]['id']] = do_df[v].values
            else:
                inp = z[self.variable_dict[v]['latent_ids']]
                for p in self.causal_graph.parents[v]:
                    parent_col = x[:,self.variable_dict[p]['id']]
                    parent_dim = self.variable_dict[p]['dim']
                    parent_col_tx = to_one_hot(parent_col, parent_dim) if self.variable_dict[v]['type'] == 'categorical' else col
                    inp = torch.cat((inp, parent_col_tx), 1)

                x[:self.variable_dict['id']] = self.module_dict[v](inp)
        
        return(x)
    pass

def to_one_hot(y, n_dims):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

      
class LatentGenerator(nn.Module):
    def __init__(self, num_latents, x_dim, options):
        self.model = MechanismNetwork(2 * x_dim, num_latents, options['hidden_dims'])

    def forward(self, x):
        x_missing = x != x
        x[x_missing] = 0
        x_cat = torch.cat((x, x_missing), 1)
        z = self.model(x_cat)
        return z

class MechanismNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, categorical_output=False):
        super(MechanismNetwork, self).__init__()
        self.num_hidden_layers = len(hidden_dims)
        lin_layers = []
        nonlin_layers = []

        for i, h in enumerate(hidden_dims):
            if i ==0:
                inp = input_dim
            else:
                inp = hidden_dims[i-1]
            lin_layers.append(nn.Linear(inp, h))
            nonlin_layers.append(nn.Tanh())

        self.lin_layers = nn.ModuleList(lin_layers)
        self.nonlin_layers = nn.ModuleList(nonlin_layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        if categorical_output:
            self.transform_layer = nn.Sigmoid()
        else:
            self.transform_layer = nn.Linear(output_dim, output_dim) #change to Identity

    def forward(self, x):
        for i in range(self.num_hidden_layers):
            x = self.nonlin_layers(self.lin_layers[i](x))
        y = self.transform_layer(self.output_layer(x))        
        return y
