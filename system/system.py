import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from generators import ForwardGenerator, LatentGenerator
import matplotlib.pyplot as plt
import seaborn as sb

class SystemModel():
    def __init__(self, variable_dict, causal_graph, 
                         conditional_model=True, 
                         additional_latent_dict = None):
        ### variable_dict {variable_name: dict{'type', 'dim', 'trasform', 'inverse_transform'})
        self.variable_dict = variable_dict
        self.categorical_col_ids = [variable_dict[v]['id'] for v in variable_dict.keys() if variable_dict[v]['type'] == 'categorical']
        self.numeric_col_ids = [variable_dict[v]['id'] for v in variable_dict.keys() if variable_dict[v]['type'] == 'numeric']

        self.causal_graph = causal_graph

        self.categorical_cols = [v for v in variable_dict.keys()
                              if variable_dict[v]['type'] == "categorical"]
        
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
                                    self.causal_graph, {'hidden_dims':[10, 10, 10]})
        self.latent_generator = LatentGenerator(self.num_latents, 
                            len(self.variable_dict), self.x_one_hot_dim, self.variable_dict, {'hidden_dims':[10, 10, 10]})
        
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
        ce_loss  = nn.CrossEntropyLoss()

        num_batches = 0
        for epoch in range(options['num_epochs']):
            for x in dataloader:
                #x_df = pd.DataFrame.from_dict(x_dict)
                forward_optim.zero_grad()
                latent_optim.zero_grad()
                num_batches += 1

                #x = torch.tensor(x_df.values)
                x_non_missing = x == x   
                x_missing = x != x 

                z     = self.latent_generator(x)
                x_gen, x_gen_one_hot_dict = self.forward_generator(z, do_df=pd.DataFrame()) #TODO: conditioned
                z_prior = torch.randn(z.shape)
                
                z_dist_loss  = mmd_loss(z, z_prior)
                #x_dist_loss  = mmd_loss(x, x_gen)
 
                total_loss = options['z_dist_scalar'] * z_dist_loss #+ options['x_dist_scalar'] * x_dist_loss
                            
                for v in self.variable_dict.keys():
                    variable_id = self.variable_dict[v]['id']
                    inds = torch.nonzero(x_non_missing[:,variable_id]).squeeze()

                    if self.variable_dict[v]['type'] == 'category':
                        target = x[:,variable_id].type(torch.LongTensor)
                        total_loss += ce_loss(x_gen_one_hot_dict[v][inds], target[inds])
                    else:
                        target = x[:,variable_id].type(torch.FloatTensor)
                        total_loss += mse_loss(x_gen[inds, variable_id], target[inds])

                total_loss.backward()
                forward_optim.step()
                latent_optim.step()

                if (num_batches * options['batch_size']) % 500  == 0:
                    print ("Epoch = %2d, num_b = %5d, total_loss = %.5f, z_loss = %.5f"%
                           (epoch, num_batches, total_loss, z_dist_loss))

            forward_scheduler.step()
            latent_scheduler.step()

            if epoch % 2 == 1:
               
                x = x.detach().numpy()
                x[x_missing] = np.nan

                x_gen = x_gen.detach().numpy()
                x_gen_p, _ = self.forward_generator(z_prior)
                x_gen_p = x_gen_p.detach().numpy()

                x_df = pd.DataFrame(x)
                x_df['type'] = 'orig'
                x_gen_df = pd.DataFrame(x_gen)
                x_gen_df['type'] = 'decoded'
                x_gen_p_df = pd.DataFrame(x_gen_p)
                x_gen_p_df['type'] = 'gen'

                x_df = x_df.append(x_gen_df, ignore_index=True)
                x_df = x_df.append(x_gen_p_df, ignore_index=True)

                sb.pairplot(pd.DataFrame(x_df), hue = 'type', 
                                 markers=["o", "s", "+"], 
                                 diag_kind = "hist", dropna=True)
                
                plt.show()
                

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

    ################
    #Example 1
    #df = pd.read_csv('data/5dmissing.csv')
    #df = pd.read_csv('data/5d.csv')
    #r = DataSet([df])
    ###############

    ##############
    #Example 2
    dd = pd.read_csv('data/toy-DSD.csv', 
        usecols=['Product','Channel','Sales'])

    ds = pd.read_csv('data/toy-Survey.csv',
        usecols=['Product','Shopper','VolumeBought'])

    dp = pd.read_csv('data/toy-Product.csv',
        usecols = ['Product','Channel','Shopper'])
    
    dt = pd.read_csv('data/toy-TLOG.csv',
        usecols = ['Product','Price','VolumeBought','Sales'])

    r = DataSet([dd, ds, dp, dt])
    #############

    cs = CausalStructure(r.variable_names)

    cs.learn_structure(r)
    cs.plot()
    plt.show()

    sm = SystemModel(r.variable_dict, cs)

    options = {'batch_size':200,
               'num_epochs':20,
               'forward_lr': 0.01,
               'latent_lr':0.03,
               'z_dist_scalar': 10.0
                }

    sm.learn_generators(r, options)

