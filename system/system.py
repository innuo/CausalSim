import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from generators import ForwardGenerator, LatentGenerator

class SystemModel():
    def __init__(self, variable_dict, causal_graph, 
                         conditional_model=True, 
                         additional_latent_dict = None):
        ### variable_dict {variable_name: dict{'type', 'dim', 'trasform', 'inverse_transform'})
        self.variable_dict = variable_dict
        self.categorical_col_ids = [variable_dict[v]['id'] for v in variable_dict.keys() if variable_dict[v]['type'] == 'categorical']
        self.numeric_col_ids = [variable_dict[v]['id'] for v in variable_dict.keys() if variable_dict[v]['type'] == 'numeric']

        self.causal_graph = causal_graph
        self.num_latents = 0
        for v in self.variable_dict.keys():
            additional_latents = additional_latent_dict[v] if additional_latent_dict is not None else []
            self.variable_dict[v]['latents'] = ['z_%s'%v] + additional_latents
            self.variable_dict[v]['latent_ids'] = range(self.num_latents, self.num_latents+len(self.variable_dict[v]['latents']))
            self.num_latents = self.num_latents + len(self.variable_dict[v]['latents'])
            
            #self.variable_dict[v]['parent_ids'] = [self.variable_dict[k]['id'] for k in causal_graph.parents[v]]
            parent_dim = sum([self.variable_dict[k]['dim'] for k in causal_graph.parents[v]])
            self.variable_dict[v]['full_parent_dim'] = parent_dim + len(self.variable_dict[v]['latents'])
   
        self.x_one_hot_dim = sum([self.variable_dict[k]['dim'] for k in self.variable_dict.keys()])
  
        self.forward_generator = ForwardGenerator(self.variable_dict, 
                                    self.causal_graph, {'hidden_dims':[10, 10]})
        self.latent_generator = LatentGenerator(self.num_latents, 
                            len(self.variable_dict), self.x_one_hot_dim, self.variable_dict, {'hidden_dims':[10, 10]})
           
        self.is_trained = False

    def update_structure(self, structure):
        self.structure = structure
        self.is_trained = False
    
    def learn_generators(self, dataset, options):
        dataloader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=True)
        forward_optim = optim.Adam(self.forward_generator.parameters(), lr=options['forward_lr'])
        latent_optim  = optim.Adam(self.latent_generator.parameters(), lr=options['latent_lr'])

        forward_scheduler = StepLR(forward_optim, step_size=5, gamma=0.5)
        latent_scheduler  = StepLR(latent_optim, step_size=5, gamma=0.5)

        mse_loss = nn.MSELoss()
        num_x = 0
        for epoch in range(options['num_epochs']):
            for x_dict in dataloader:
                x_df = pd.DataFrame.from_dict(x_dict)
                forward_optim.zero_grad()
                latent_optim.zero_grad()
                num_x += x_df.shape[0]

                x = torch.tensor(x_df.values)
                x_missing = x != x    

                z     = self.latent_generator(x)
                x_gen = self.forward_generator(z, do_df=pd.DataFrame()) #TODO: conditioned
                z_prior = torch.randn(z.shape)
                
                z_dist_loss  = mmd_loss(z, z_prior)
                #x_dist_loss  = mmd_loss(x, x_gen)
                xmse         = mse_loss(x.type(torch.FloatTensor), x_gen)

                total_loss = xmse + options['z_dist_scalar'] * z_dist_loss #+ options['x_dist_scalar'] * x_dist_loss
                total_loss.backward()

                forward_optim.step()
                latent_optim.step()
                if num_x % 500 == 0:
                    print ("Epoch = %2d, num_x = %5d, x_loss = %.5f, z_loss = %.5f"%
                           (epoch, num_x, xmse, z_dist_loss))

            forward_scheduler.step()
            latent_scheduler.step()

        self.is_trained = True
        pass
    
    def sample(self, num_samples, do_df=pd.DataFrame()): #TODO: conditioned
        pass

def square_dist_mat(x, y):
    xs = x.pow(2).sum(1, keepdim=True)
    ys = y.pow(2).sum(1, keepdim=True)
    return torch.clamp(xs + ys.t() - 2.0 * x @ y.t(), min=0.0)

def mmd_loss(x, y, d = 1):
    dists_x = square_dist_mat(x, x)
    dists_y = square_dist_mat(y, y)
    dists_xy = square_dist_mat(x, y)

    dist = 0
    for scale in [0.1, 1.0, 10.]:
        c = d * scale
        res  = c/(c + dists_x) + c/(c+ dists_y) - 2 * c/(c + dists_xy)
        dist += res.mean()
    
    return dist

        
 
if __name__ == '__main__':
    from datahandler import DataSet
    from structure import CausalStructure
    import os

    df = pd.read_csv('../data/5d.csv')
    r = DataSet([df])
    cs = CausalStructure(r.variable_names)
    cs.learn_structure(r)
    sm = SystemModel(r.variable_dict, cs)

    options = {'batch_size':10,
               'num_epochs':20,
               'forward_lr': 0.001,
               'latent_lr':0.01,
               'z_dist_scalar': 10.0
                }

    sm.learn_generators(r, options)

