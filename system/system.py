from sklearn import preprocessing
import ForwardGenerator
import LatentGenerator

class SystemModel():
    def __init__(self, variable_dict, causal_graph, 
                         conditional_model=True, 
                         additional_latent_dict = None):
        ### variable_dict {variable_name: dict{'type', 'dim', 'transform', 'inverse_transform'})
        self.variable_dict = variable_dict
        self.causal_graph = causal_graph
        self.num_latents = 0
        for v in self.variable_dict.keys():
            
            self.variable_dict[v]['latents'] = ['z_%s'%v] + additional_latent_dict[v]
            self.variable_dict[v]['latent_ids'] = range(self.num_latents, self.num_latents+len(self.variable_dict[v]['latents']))
            self.num_latents = self.num_latents + len(self.variable_dict[v]['latents'])
            
            #self.variable_dict[v]['parent_ids'] = [self.variable_dict[k]['id'] for k in causal_graph.parents[v]]
            parent_dim = sum([self.variable_dict[k]['dim'] for k in causal_graph.parents[v]])
            self.variable_dict[v].full_parent_dim = parent_dim + len(self.variable_dict[v]['latents'])
   
        self.x_one_hot_dim = sum([self.variable_dict[k]['dim'] for k in self.variable_dict.keys()])
  
        self.forward_generator = ForwardGenerator(self.variable_dict, self.causal_graph, {'hidden_dims':[10, 10]})
        self.latent_generator = LatentGenerator(self.num_latents, self.x_one_hot_dim, {'hidden_dims':[10, 10]})
           
        self.is_trained = False

    def update_structure(self, structure):
        self.structure = structure
        self.is_trained = False
    
    def learn_samplers(self):
        pass
    
    def sample(self, num_samples):
        pass


