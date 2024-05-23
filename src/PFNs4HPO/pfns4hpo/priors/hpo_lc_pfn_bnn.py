import os
import torch
import math
import numpy as np
import random
from scipy.stats import norm
from pfns4hpo.priors.utils import Batch
from pfns4hpo.priors import hebo_prior
from pfns4hpo.utils import default_device
from pfns4hpo import encoders

def pow3(x, a, c, alpha, *args):
    return c - a * torch.pow(x+1, -alpha)


class DatasetPrior:

    output_sorted = None

    def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_normal_(m.weight)

    def __init__(self, num_features, num_outputs):
        self.num_features = num_features
        self.num_outputs = num_outputs
        
        self.num_inputs = 2*(num_features+2)
        num_hidden = 100
        N = 1000
        
        self.model = torch.nn.Sequential(
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            torch.nn.Linear(self.num_inputs, num_hidden),
            torch.nn.ELU(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.ELU(),
            torch.nn.Linear(num_hidden, self.num_outputs)
        )

        if DatasetPrior.output_sorted is None:
            # generate samples to approximate the CDF of the BNN output distribution
            output = torch.zeros((N, num_outputs))
            input = torch.from_numpy(np.random.uniform(size=(N,self.num_inputs))).to(torch.float32)
            with torch.no_grad():
                for i in range(N):
                    self.model.apply(DatasetPrior.init_weights)
                    output[i] = self.model(input[i])
            
            DatasetPrior.output_sorted = np.sort(torch.flatten(output).numpy())

        # fix the parameters of the BNN
        self.model.apply(DatasetPrior.init_weights)

        # fix other dataset specific
        self.input_features = np.random.uniform(size=(self.num_inputs,))
        p_alloc = np.random.dirichlet(tuple([1 for _ in range(self.num_features)] + [1, 1]))
        self.alloc = np.random.choice(self.num_features+2, size=(self.num_inputs,), p=p_alloc)
        #print(self.alloc)

    
    def input_for_config(self, config):
        input_noise = np.random.uniform(size=(self.num_inputs,))
        input = torch.zeros((self.num_inputs,))
        for j in range(self.num_inputs):
            if self.alloc[j] < self.num_features:
                input[j] = config[self.alloc[j]]
            elif self.alloc[j] == self.num_features:
                input[j] = self.input_features[j]
            else:
                input[j] = input_noise[j]
        return input
        
    def output_for_config(self, config):
        input = self.input_for_config(config)
        return self.model(input)

    def uniform(self, bnn_output, a=0.0, b=1.0):
        indices = np.searchsorted(DatasetPrior.output_sorted, bnn_output, side='left')
        return (b-a) * indices / len(DatasetPrior.output_sorted) + a
    
    def normal(self, bnn_output, loc=0, scale=1):
        eps = 0.5 / len(DatasetPrior.output_sorted) # to avoid infinite samples
        u = self.uniform(bnn_output, a=eps, b=1-eps)
        return norm.ppf(u, loc=loc, scale=scale)


def curve_prior(c_minus_a, dataset, config):
    output = dataset.output_for_config(config).numpy()
    c = dataset.uniform(output[0], a=c_minus_a, b=1.0)
    a = c - c_minus_a
    alpha = np.exp(dataset.normal(output[1], scale=2))
    sigma = np.exp(dataset.normal(output[2], loc=-5, scale=1))

    def foo(x_):
        y_ = pow3(x_, a, c, alpha.item())
        noise = np.random.normal(size=x_.shape, scale=sigma)
        return y_ + noise
            
    return foo

# function producing batches for PFN training
@torch.no_grad()
def get_batch(
    batch_size,
    seq_len,
    num_features,
    single_eval_pos,
    device=default_device,
    hyperparameters=None,
    **kwargs,
):
    if hyperparameters.get("load_path", False):
        if not hasattr(get_batch, "seq_counter"):
            get_batch.seq_counter = 0
            get_batch.loaded_chunk_id = None
            get_batch.loaded_chunk_x = None
            get_batch.loaded_chunk_y = None
            
        path = hyperparameters["load_path"]
        chunk_size = hyperparameters["chunk_size"]
        n_chunks = hyperparameters["n_chunks"]
        
        chunk_id = get_batch.seq_counter // chunk_size
        chunk_id = chunk_id % n_chunks  # cycle through the data
        if chunk_id != get_batch.loaded_chunk_id:
            # we need to load the next chunk
            get_batch.loaded_chunk_x = np.load(os.path.join(path, f"chunk_{chunk_id}_x.npy"))
            get_batch.loaded_chunk_y = np.load(os.path.join(path, f"chunk_{chunk_id}_y.npy"))
            get_batch.loaded_chunk_id = chunk_id
            x = get_batch.loaded_chunk_x[:,:batch_size]
            y = get_batch.loaded_chunk_y[:,:batch_size]
        else:
            offset = get_batch.seq_counter % chunk_size
            if offset+batch_size <= chunk_size:
                # we have all the data needed in memory
                x = get_batch.loaded_chunk_x[:,offset:offset+batch_size]
                y = get_batch.loaded_chunk_y[:,offset:offset+batch_size]
            else:
                # we have part of the data needed in memory, eagerly load next chunk already
                next_chunk_id = (chunk_id+1) % n_chunks
                next_chunk_x = np.load(os.path.join(path, f"chunk_{next_chunk_id}_x.npy"))
                next_chunk_y = np.load(os.path.join(path, f"chunk_{next_chunk_id}_y.npy"))
                # load rest
                x = np.concatenate((get_batch.loaded_chunk_x[:,offset:],
                                    next_chunk_x[:,:batch_size-chunk_size+offset]), axis=1)
                y = np.concatenate((get_batch.loaded_chunk_y[:,offset:],
                                    next_chunk_y[:,:batch_size-chunk_size+offset]), axis=1)
                get_batch.loaded_chunk_x = next_chunk_x
                get_batch.loaded_chunk_y = next_chunk_y
                get_batch.loaded_chunk_id = next_chunk_id
        assert(len(x[0]) == batch_size)
        assert(len(y[0]) == batch_size)
        get_batch.seq_counter += batch_size
        
        x = torch.from_numpy(x).to(device).float()
        y = torch.from_numpy(y).to(device).float()
    
        return Batch(x=x, y=y, target_y=y)
    else:
        # assert num_features == 2
        assert num_features >= 2

        num_params = np.random.randint(1, num_features - 2)
        
        x_ = torch.arange(1, seq_len + 1)
    
        x = []
        y = []
    
        for i in range(batch_size):
            epoch = torch.zeros(seq_len)
            id_curve = torch.zeros(seq_len)
            curve_val = torch.zeros(seq_len)
            config = torch.zeros(seq_len, num_params)
    
            # sample a collection of curves
            dataset = DatasetPrior(num_params, 3)
            curves = []
            curve_configs = []
            max_epochs = int(np.rint(np.exp(np.random.uniform(0, np.log(seq_len+1)))))
            c_minus_a = np.random.uniform() # epoch 0 performance (performance of a randomly initialized model, before training)
            for i in range(seq_len):
                c = np.random.uniform(size=(num_params,))  # random configurations
                curve_configs.append(c)
                curves.append(curve_prior(c_minus_a,dataset,c))
            
            # determine an ordering for unrolling curves
            log10_seq_len = np.log(seq_len)/np.log(10)
            p_new = 10**np.random.uniform(-log10_seq_len, 0) # probability of starting a new curve
            greediness = 10**np.random.uniform(0, log10_seq_len)  # Select prob 'best' / 'worst' possible (1 = URS)
            print(f"{p_new} {greediness} {max_epochs} {num_params}")
            ordering = hyperparameters.get("ordering", "SoftGreedy")
            ids = np.arange(seq_len)
            np.random.shuffle(ids)  # randomize the order of IDs to avoid learning weird biases
            
            cutoff = torch.zeros(seq_len).type(torch.int64)
            query_cache = {}
            for i in range(seq_len):
                if ordering == "URS":
                    candidates = [j for j, c in enumerate(cutoff) if c < max_epochs]
                    selected = np.random.choice(candidates)
                elif ordering == "BFS":
                    selected = i % seq_len
                elif ordering == "DFS":
                    selected = i // max_epochs
                elif ordering == "SoftGreedy":
                    u = np.random.uniform()
                    if u < p_new:
                        new_candidates = [j for j, c in enumerate(cutoff) if c == 0 and len(query_cache.get(j,set())) < max_epochs]
                    if u < p_new and len(new_candidates) > 0:
                        selected = np.random.choice(new_candidates)
                    else:
                        candidates = [j for j, c in enumerate(cutoff) if c+len(query_cache.get(j,set())) < max_epochs and c > 0]
                        #print(candidates)
                        if len(candidates) == 0:
                            if u >= p_new:
                                new_candidates = [j for j, c in enumerate(cutoff) if c == 0 and len(query_cache.get(j,set())) < max_epochs]
                            selected = np.random.choice(new_candidates)
                        else:
                            # use softmax selection based on performance and selected cutoff
                            # select a cutoff to compare on (based on current / last)
                            selected_cutoff = int(np.rint(np.exp(np.random.uniform(0, np.log(max_epochs+1)))-1))
                            values = []
                            for j in candidates:
                                v = torch.FloatTensor([min(selected_cutoff, cutoff[j]-1)])
                                values.append(curves[j](v))
                            sm_values = np.power(greediness, np.asarray(values))
                            sm_values = sm_values / np.sum(sm_values)
                            if np.isnan(np.sum(sm_values)):
                                print(values)
                                print(greediness)
                                print(selected_cutoff)
                                print(cutoff[j])
                                for j in candidates:
                                    print(curves[:cutoff[j]])
                            selected = np.random.choice(candidates, p=sm_values.flatten())
                else:
                    raise NotImplementedError

                id_curve[i] = ids[selected]
                config[i] = torch.from_numpy(curve_configs[selected])
                if i < single_eval_pos:
                    # we actually unroll this curve
                    curve_val[i] = curves[selected](cutoff[selected])
                    cutoff[selected] += 1
                    epoch[i] = cutoff[selected]
                else:
                    # we query a future point on the selected curve
                    query_cache[selected] = query_cache.get(selected,set())
                    candidate_query_points = [c+1 for c in range(cutoff[selected],max_epochs) if c+1 not in query_cache[selected]]
                    #print(f"{selected}: {candidate_query_points}")
                    # we use a log-uniform distribution to bias sampling to shorter horizons
                    selected_query_idx = int(np.rint(np.exp(np.random.uniform(0, np.log(len(candidate_query_points))))-1))
                    v = candidate_query_points[selected_query_idx]
                    selected_query_point = torch.FloatTensor([v])
                    curve_val[i] = curves[selected](selected_query_point)
                    epoch[i] = selected_query_point
                    query_cache[selected].add(v)
                    #print(query_cache)
            x.append(torch.cat([torch.stack([id_curve, epoch], dim=1), config], dim=1))
            y.append(curve_val)

            # now determine an 
    
        x = torch.stack(x, dim=1).to(device).float()
        y = torch.stack(y, dim=1).to(device).float()
    
        return Batch(x=x, y=y, target_y=y)


class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        seq_len = 1000
        self.normalizer = torch.nn.Sequential(
            encoders.Normalize(0.0, seq_len),
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
        )
        self.epoch_enc = torch.nn.Linear(1, out_dim, bias=False)
        self.idcurve_enc = torch.nn.Embedding(2*seq_len+1, out_dim)
        self.configuration_enc = encoders.get_variable_num_features_encoder(encoders.Linear)(in_dim-2, out_dim)

    def forward(self, *x, **kwargs):
        x = torch.cat(x, dim=-1)
        out = self.epoch_enc(self.normalizer(x[..., :1])) \
            + self.idcurve_enc(x[..., 1:2].int()).squeeze(2) \
            + self.configuration_enc(x[..., 2:])
        return out


def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)