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
    
    def learn_generators(self, dataset, options):
        dataloader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=True)
        forward_optim = optim.Adam(self.latent_generator.parameters(), lr=args.lr)
        latent_optim  = optim.Adam(self.forward_generator.parameters(), lr=args.lr)

        forward_scheduler = StepLR(enc_optim, step_size=5, gamma=0.5)
        latent_scheduler  = StepLR(dec_optim, step_size=5, gamma=0.5)

        mse_loss = nn.MSELoss()
        for epoch in range(options['num_epochs'])
            for x_df in dataloader:
                forward_optim.zero_grad()
                latent_optim.zero_grad()

                x = torch.tensor(x_df.values)
                z     = self.latent_generator()
                x_gen = self.forward_generator(z, do_df=pd.DataFrame()) #TODO: conditioned
                z_prior = torchm.randn(z.shape)
                
                z_dist_loss  = mmd_loss(z, z_prior)
                x_dist_loss  = mmd_loss(x, x_gen)
                xmse         = mse_loss(x, x_gen)

                total_loss = xmse + options['x_dist_scalar'] * x_dist_loss + options['z_dist_scalar'] * z_dist_loss
                total_loss.backward()

                forward_optim.step()
                latent_optim.step()

        pass
    
    def sample(self, num_samples, do_df=pd.DataFrame()): #TODO: conditioned
        pass

