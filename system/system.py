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
                                    self.causal_graph, {'hidden_dims':[20, 20]})
        self.latent_generator = LatentGenerator(self.num_latents, 
                            len(self.variable_dict), self.x_one_hot_dim, self.variable_dict, {'hidden_dims':[20, 20]})
        
        self.is_trained = False

    def update_structure(self, structure):
        self.structure = structure
        self.is_trained = False
    
    def learn_generators(self, dataset, options):
        dataloader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=True)
        forward_optim = optim.Adam(self.forward_generator.parameters(), lr=options['forward_lr'])
        latent_optim  = optim.Adam(self.latent_generator.parameters(), lr=options['latent_lr'])

        forward_scheduler = StepLR(forward_optim, step_size=200, gamma=0.5)
        latent_scheduler  = StepLR(latent_optim, step_size=200, gamma=0.5)

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

                z_mean, z_std     = self.latent_generator(x)

                #z = torch.randn(z_mean.shape) * z_std + z_mean
                z = z_mean

                z_prior = torch.randn(z.shape)

                ind = torch.rand (z.shape).argsort (dim = 0)
                z_s = torch.zeros (z.shape).scatter_ (0, ind, z)

                x_gen, x_gen_one_hot_dict = self.forward_generator(z, do_df=pd.DataFrame()) #TODO: conditioned
                z_dist_loss  = mmd_loss(z, z_prior) #+ mmd_loss(z, z_s) 
 
                total_loss = options['z_dist_scalar'] * z_dist_loss #+ options['x_dist_scalar'] * x_dist_loss

                x_loss = torch.tensor(0.0)
                for v in self.variable_dict.keys():
                    variable_id = self.variable_dict[v]['id']
                    inds = torch.nonzero(x_non_missing[:,variable_id]).squeeze()

                    if self.variable_dict[v]['type'] == 'categorical':
                        target = x[:,variable_id].type(torch.LongTensor)
                        x_loss += ce_loss(x_gen_one_hot_dict[v][inds], target[inds])
                    else:
                        target = x[:,variable_id].type(torch.FloatTensor)
                        x_loss += mse_loss(x_gen[inds, variable_id], target[inds])

                total_loss += x_loss
                total_loss.backward()
                forward_optim.step()
                latent_optim.step()

                if (num_batches * options['batch_size']) % 5000  == 0:
                    print ("Epoch = %2d, num_b = %5d, total_loss = %.5f, x_loss = %.5f, z_loss = %.5f"%
                           (epoch, num_batches, total_loss, x_loss, z_dist_loss))

            forward_scheduler.step()
            latent_scheduler.step()

            if epoch % 200 == 1 and options['plot']:
                tmp = DataLoader(dataset, batch_size=500, shuffle=True)
                x = iter(tmp).next()
                z, _ = self.latent_generator(x)

                x = x.detach().numpy()
                x[x_missing] = np.nan


                z_prior = torch.randn(z.shape)

                x_gen, x_gen_one_hot_dict = self.forward_generator(z, do_df=pd.DataFrame())
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
                                 markers=["o", "+", "s"],
                                 diag_kind = "hist", dropna=True)


                z_df = pd.DataFrame(z.detach().numpy())
                z_s_df = pd.DataFrame(z_prior.detach().numpy())

                z_df['type'] = "Z"
                z_s_df['type'] = 'Zs'
                z_s_df = z_s_df.append(z_df, ignore_index=True)

                sb.pairplot(pd.DataFrame(z_s_df), hue = 'type', 
                                 markers=["o", "s"], 
                                 diag_kind = "hist", dropna=True)
                
                plt.show()
                

        self.is_trained = True
        data_loader = DataLoader(dataset, batch_size=dataset.__len__(),
                             num_workers=1, shuffle=False)

        all_x = next(iter(data_loader))

        self.z_training = self.latent_generator(all_x)
        pass
    
    def fill(self, dataset):
        dataset_size = dataset.__len__()
        data_loader = DataLoader(dataset, batch_size=dataset_size,
                             num_workers=1, shuffle=False)
        #for x in data_loader:
        #    all_x = x
        all_x = next(iter(data_loader))

        non_missing = all_x == all_x
        z_mean,_ = self.latent_generator(all_x)
        x_gen, _ = self.forward_generator(z_mean)
        x_gen[non_missing] = all_x[non_missing]

        x_df = self.__make_data_frame(x_gen)
        
        return x_df


    def sample(self, num_samples, do_df=pd.DataFrame()): #TODO: conditioned
     
        z_p = torch.randn((num_samples, self.num_latents))
        x_gen, _ = self.forward_generator(z_p)
        x_df = pd.DataFrame(x_gen)
        x_df = self.__make_data_frame(x_gen)
        return x_df


    def __make_data_frame(self, x_gen):
        x_gen = x_gen.detach().numpy()
        x_df = pd.DataFrame()
        for v in self.variable_dict.keys():
            
            id = self.variable_dict[v]['id']
            #print("%s, %d"%(v, id))
            #print(dataset.variable_dict[v]['inverse_transform'])
            if self.variable_dict[v]['type'] == 'categorical':
                col = np.int16(x_gen[:, id])
            else:
                col = x_gen[:, id]
            x_df[v] = self.variable_dict[v]['inverse_transform'](col)

        return x_df

def square_dist_mat(x, y):
    xs = x.pow(2).sum(1, keepdim=True)
    ys = y.pow(2).sum(1, keepdim=True)
    return torch.clamp(xs + ys.t() - 2.0 * x @ y.t(), min=0.0)

def mmd_loss(x, y, d = 1):
    dists_x = square_dist_mat(x, x)
    dists_y = square_dist_mat(y, y)
    dists_xy = square_dist_mat(x, y)

    nx = x.shape[0]
    ny = y.shape[0]

    res = 0
    for scale in [0.1, 1.0, 10, 100]:
        c = d * scale
        res_x  = c/(c + dists_x) 
        res  += (res_x.sum() - torch.diag(res_x).sum()) * 1.0/(nx * (nx-1))
        res_y  = c/(c + dists_y) 
        res  += (res_y.sum() - torch.diag(res_y).sum()) * 1.0/(ny * (ny-1))
        
        res_xy  =  c/(c + dists_xy)
        res -=  2 * res_xy.mean()
    
    return res


from datahandler import DataSet
from structure import CausalStructure
import networkx as nx
def old_main():

    import os

    ################
    # Example 1
    # df = pd.read_csv('data/5dmissing.csv')
    df = pd.read_csv('data/5d.csv')
    r = DataSet([df])
    ###############

    ##############
    # Example 2
    """
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
    """

    cs = CausalStructure(r.variable_names)

    cs.learn_structure(r)
    cs.plot()
    plt.show()

    """
    ### Replace structure EXAMPlE 2##
    variable_names = list(r.df)
    
    dag = nx.empty_graph(len(variable_names), create_using=nx.DiGraph())
    dag = nx.relabel_nodes(dag, dict(zip(range(len(variable_names)), variable_names)))

    edges = [('Product', 'Price'), 
            ('Channel', 'Price'),
            ('Product', 'VolumeBought'),
            ('Shopper', 'VolumeBought'),
            ('Price', 'Sales'),
            ('VolumeBought', 'Sales'),
            ('Price', 'VolumeBought')
            ]
    dag.add_edges_from(edges)
    cs.update_structure(dag, merge_type="replace")
    print("\n\nREPLACING LEARNED GRAPH WITH TRUE GRAPH")
    cs.plot()
    plt.show()
    ##################
    """

    sm = SystemModel(r.variable_dict, cs)

    options = {'batch_size': 500,
               'num_epochs': 20,
               'forward_lr': 0.01,
               'latent_lr': 0.03,
               'z_dist_scalar': 100.0,
               'plot': False
               }

    sm.learn_generators(r, options)

    filled_df = sm.fill(r)
    filled_df.to_csv('data/toy-result.csv', index=None, header=True)

    sample_df = sm.sample(1000)
    sample_df.to_csv('data/toy-simulated.csv', index=None, header=True)

def abc_test():
    df = pd.read_csv('data/abc.csv')
    r = DataSet([df])

    edges = [('A', 'B'),
             ('A', 'C'),
             ('B', 'C')
             ]
    dag = nx.empty_graph(3, create_using=nx.DiGraph())
    dag = nx.relabel_nodes(dag, dict(zip(range(3), ['A', 'B' , 'C'])))
    dag.add_edges_from(edges)
    cs = CausalStructure(r.variable_names, dag=dag)
    cs.plot()
    plt.show()

    sm = SystemModel(r.variable_dict, cs)

    options = {'batch_size': 50,
               'num_epochs': 1010,
               'forward_lr': 0.01,
               'latent_lr': 0.01,
               'z_dist_scalar': 100.0,
               'plot': True
               }

    sm.learn_generators(r, options)

if __name__ == '__main__':
    abc_test()