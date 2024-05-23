import time
import functools
import random
import math
import traceback
import warnings
import random, math
import numpy as np
import torch
from torch import nn
import gpytorch
import botorch
from gpytorch.models import ExactGP
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors.torch_priors import GammaPrior, UniformPrior, LogNormalPrior
from gpytorch.means import ZeroMean
from botorch.models.transforms.input import *
from gpytorch.constraints import GreaterThan
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP

import torch.nn.utils.rnn as rnn_utils

from . import utils
from ..utils import default_device, to_tensor
from .prior import Batch
from .utils import get_batch_to_dataloader
from pfns4hpo import encoders

def constraint_based_on_distribution_support(prior: torch.distributions.Distribution, device):

    if hasattr(prior.support, 'upper_bound'):
        return gpytorch.constraints.Interval(to_tensor(prior.support.lower_bound,device=device),
                                             to_tensor(prior.support.upper_bound,device=device))
    else:
        return gpytorch.constraints.GreaterThan(to_tensor(prior.support.lower_bound,device=device))
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
    elif type(m) == torch.nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class FeatureExtractor(nn.Module):
    """
    The feature extractor that is part of the deep kernel.
    """
    def __init__(self, configuration):
        super(FeatureExtractor, self).__init__()

        self.configuration = configuration

        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()
        # adding one to the dimensionality of the initial input features
        # for the concatenation with the budget.
        initial_features = configuration['nr_initial_features'] + 1
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])
        
        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i + 1}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i + 1}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )
            print(":", i)


        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] +
                configuration['cnn_nr_channels'],  # accounting for the learning curve features
                configuration[f'layer{self.nr_layers}_units']
            ),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(configuration['cnn_kernel_size'],), out_channels=4),
            nn.AdaptiveMaxPool1d(1),
        )
        self.init()

    def forward(self, x, budgets, learning_curves):

        # add an extra dimensionality for the budget
        # making it nr_rows x 1.
        budgets_ = budgets / 50
        # concatenate budgets with examples
        
        x = torch.cat((x, budgets_), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                # getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(
                        x
                    )
                # )
            ) 

        # add an extra dimensionality for the learning curve
        # making it nr_rows x 1 x lc_values.
        learning_curves = torch.unsqueeze(learning_curves, 1)
        lc_features = self.cnn(learning_curves)
        # revert the output from the cnn into nr_rows x nr_kernels.
        lc_features = torch.squeeze(lc_features, 2)

        # put learning curve features into the last layer along with the higher level features.
        x = torch.cat((x, lc_features), dim=1)
        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))

        return x
    
    def init(self):
        self.apply(init_weights)

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        device = train_x.device
        self.mean_module = ConstantMean()
        lengthscale_prior = GammaPrior(torch.tensor(1.0, device=device), torch.tensor(1.5212245992840594, device=device))
        outputscale_prior = UniformPrior(0.05, 0.2)
        
        covar_module = gpytorch.kernels.RBFKernel(lengthscale_prior=lengthscale_prior, lengthscale_constraint=constraint_based_on_distribution_support(lengthscale_prior, device=device))
        self.covar_module = gpytorch.kernels.ScaleKernel(covar_module, outputscale_prior=outputscale_prior, outputscale_constraint=constraint_based_on_distribution_support(outputscale_prior, device=device))
        self.feature_extractor = feature_extractor

    def forward(self, config, budgets, learning_curves):
        # Use the feature extractor to get the features
        # assuming that the first 10 columns are the configuration
        # the next 1 is the budget
        # and the rest is the learning curve.
        extracted_features = self.feature_extractor(config, budgets, learning_curves)
        mean_x = self.mean_module(extracted_features)
        covar_x = self.covar_module(extracted_features)
        return MultivariateNormal(mean_x, covar_x)


def get_model(x, hyperparameters: dict, device=default_device):
    
    # likelihood definition
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    likelihood.register_prior("noise_prior",
                              LogNormalPrior(torch.tensor(hyperparameters.get('hebo_noise_logmean',-4.63), device=device),
                                             torch.tensor(hyperparameters.get('hebo_noise_std', 0.25), device=device)
                                             ),
                              "noise")
   
    # model definition
    configuration = {
        'nr_layers': 2,
        'nr_initial_features': hyperparameters["num_features"],
        'layer1_units': 64,
        'layer2_units': 128,
        'cnn_nr_channels': 4,
        'cnn_kernel_size': 3
    }
    
    feature_extractor = FeatureExtractor(configuration).to(device)
    model = GPModel(x, torch.rand(x.size(0)), likelihood, feature_extractor)
    model.feature_extractor.init()
    
    model.to(device)
    likelihood.to(device)
    model = model.pyro_sample_from_prior()
    return model, model.likelihood

def generate_variable_length_sequences(budgets, device=default_device):
    """
    Generate a batch of variable-length sequences based on the provided budgets.

    Args:
    budgets (torch.Tensor): A 1D tensor of budget values.

    Returns:
    torch.Tensor: A padded tensor of sequences.
    """
    
    sequences = []
    p = -0.8
    for budget in budgets:
        if budget > 0:
            budget = random.choices(range(budget, 0, -1), [1 / math.pow(((budget - 0) - i), p) for i in range(budget - 0)])[0]
        
        if budget == 0:
            sequence = torch.zeros(size=(3,))
        elif 0 < budget <= 3:
            sequence = torch.zeros(size=(3,))
            sequence[:budget] = torch.rand(size=(budget,))
        else:
            sequence = torch.rand(size=(budget,))
        sequences.append(sequence)
    
    sequences.append(torch.empty(size=(50,)))  
    padded_sequences = rnn_utils.pad_sequence(sequences, batch_first=True)[:-1]
    return padded_sequences.to(device)

def generate_budgets_with_total_sum(target_sum, ratio_of_ones, max_budget=50, device='cpu'):
    """
    Generate a tensor of budget values whose sum equals the target sum, with a specified ratio of 1-value budgets.

    Args:
    target_sum (int): The total sum of the budgets.
    ratio_of_ones (float): The ratio of the budgets that should be 1.
    max_budget (int): The maximum value for an individual budget.
    device (str): The device on which the tensor will be created.

    Returns:
    torch.Tensor: A tensor of budget values.
    """
    num_ones = int(target_sum * ratio_of_ones)
    budgets = [1] * num_ones
    current_sum = sum(budgets)

    # Generate budgets until the sum is close to the target
    while current_sum < target_sum:
        next_budget = torch.randint(2, max_budget + 1, (1,)).item()
        
        # Adjust the last budget if the sum exceeds the target
        if current_sum + next_budget > target_sum:
            next_budget = target_sum - current_sum

        budgets.append(next_budget)
        current_sum += next_budget

    random.shuffle(budgets)
    return torch.tensor(budgets, device=device)

def create_formatted_tensor(configurations, budgets, learning_curves, predicted_values, device='cpu'):
    """
    Create a formatted tensor for learning curves and return y values for each configuration at each epoch.

    Args:
    configurations (torch.Tensor): Tensor of shape N x d containing configurations.
    budgets (torch.Tensor): Tensor of shape N containing budget values.
    learning_curves (torch.Tensor): Tensor of shape N x 50 containing learning curves.
    device (str): The device on which the tensor will be created.

    Returns:
    torch.Tensor: Formatted tensor of shape N x (1+1+d).
    torch.Tensor: Tensor of y values corresponding to each configuration at each epoch.
    """
    formatted_rows = []
    y_values = []
    __ = 0

    for id_curve in range(learning_curves.shape[0]):
        budget = budgets[id_curve].item()
        configuration = configurations[id_curve].to(device)

        for epoch in range(1, budget + 1):
            row = torch.cat([torch.tensor([id_curve + 1], dtype=configuration.dtype, device=device),  # id_curve (1-indexed)
                             torch.tensor([epoch], dtype=configuration.dtype, device=device),       # epoch
                             configuration])  # configuration
            formatted_rows.append(row)

            # Get the corresponding y value from the learning curve
            if epoch < budget:
                y_value = learning_curves[id_curve, epoch - 1].to(device)
            else:
                y_value = predicted_values[__]
                __ += 1
            y_values.append(y_value)

    return torch.stack(formatted_rows), torch.tensor(y_values, device=device)

def rearrange_tensor(formatted_tensor, y_values):
    """
    Rearrange the formatted tensor and y values by moving the row with the highest epoch
    (greater than 1) for each id_curve to the end of the tensor.

    Args:
    formatted_tensor (torch.Tensor): The formatted tensor of shape N x (1+1+d).
    y_values (torch.Tensor): The tensor of y values.

    Returns:
    torch.Tensor: Rearranged formatted tensor.
    torch.Tensor: Corresponding rearranged y values.
    """
    # Extract id_curves
    id_curves = formatted_tensor[:, 0].int()

    # Initialize lists for the new order of rows and rows to move
    new_order = []
    rows_to_move = []
    rows_to_move_y = []

    # Iterate through each id_curve and rearrange
    for id_curve in torch.unique(id_curves):
        # Filter rows for the current id_curve
        id_curve_mask = id_curves == id_curve
        id_curve_rows = formatted_tensor[id_curve_mask]
        id_curve_y_values = y_values[id_curve_mask]

        # Find the row with the highest epoch (greater than 1)
        max_epoch_row_index = torch.argmax(id_curve_rows[:, 1])
        if id_curve_rows[max_epoch_row_index, 1] > 1:
            rows_to_move.append(id_curve_rows[max_epoch_row_index].unsqueeze(0))  # Unsqueeze to make it 2D
            rows_to_move_y.append(id_curve_y_values[max_epoch_row_index].unsqueeze(0))  # Unsqueeze to make it 2D
            # Remove the row from the original tensor
            id_curve_rows = torch.cat([id_curve_rows[:max_epoch_row_index], id_curve_rows[max_epoch_row_index + 1:]])
            id_curve_y_values = torch.cat([id_curve_y_values[:max_epoch_row_index], id_curve_y_values[max_epoch_row_index + 1:]])

        new_order.append((id_curve_rows, id_curve_y_values))

    # Concatenate the rows to form the rearranged tensor and y_values
    rearranged_formatted_tensor = torch.cat([rows for rows, _ in new_order] + rows_to_move)
    rearranged_y_values = torch.cat([y_values for _, y_values in new_order] + rows_to_move_y)

    return rearranged_formatted_tensor, rearranged_y_values

@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, device=default_device, hyperparameters={},
              batch_size_per_gp_sample=None, single_eval_pos=None,
              fix_to_range=None, equidistant_x=False, verbose=False, **kwargs):
    '''
    This function is very similar to the equivalent in .fast_gp. The only difference is that this function operates over
    a mixture of GP priors.
    :param batch_size:
    :param seq_len:
    :param num_features:
    :param device:
    :param hyperparameters:
    :param for_regression:
    :return:
    '''
    hyperparameters["num_features"] = num_features - 51
    fix_to_range = [1.5, -1.5]

    X, Y = [], []

    for _ in range(batch_size):

        with gpytorch.settings.fast_computations(*hyperparameters.get('fast_computations',(True,True,True))):

            budgets = torch.randint(1, 51, (seq_len,), device=device)
            budgets[torch.rand_like(budgets.float()) < 0.2] = 0
            # generate_budgets_with_total_sum(seq_len, np.random.uniform(), device=device)
            num_candidates = budgets.size(0)
            learning_curves = generate_variable_length_sequences(budgets, device=device)
            x = torch.rand(num_candidates, num_features, device=device)
            
            unused_feature_likelihood = hyperparameters.get('unused_feature_likelihood', True)
            
            if unused_feature_likelihood:
                r = torch.rand(x.size(1), device=device)
                unused_feature_mask = r < unused_feature_likelihood
                if unused_feature_mask.all():
                    unused_feature_mask[r.argmin()] = False
                configurations = x[...,~unused_feature_mask]
                hyperparameters["num_features"] = configurations.size(1)
            else:
                configurations = x
 
            get_model_and_likelihood = lambda: get_model(configurations, hyperparameters)
            model, likelihood = get_model_and_likelihood()

            if verbose: print(list(model.named_parameters()),
                            (list(model.input_transform.named_parameters()), model.input_transform.concentration1, model.input_transform.concentration0)
                                if model.input_transform is not None else None,
                            )

            successful_sample = 0
            while successful_sample < 1:
                with gpytorch.settings.prior_mode(True):
                    model.eval()
                    # print(x.device, device, f'{model.covar_module.base_kernel.lengthscale=}, {model.covar_module.base_kernel.lengthscale.device=}')
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            predictions = model(configurations, budgets.unsqueeze(-1), learning_curves)
                            predictions = likelihood(predictions).sample()

                    except (RuntimeError, ValueError) as e:
                        successful_sample -= 1
                        model, likelihood = get_model_and_likelihood()
                        if successful_sample < -100:
                            # print(f'Could not sample from model after {successful_sample} attempts. {e}')
                            raise e
                        continue

                    smaller_mask = predictions < fix_to_range[0]
                    larger_mask = predictions >= fix_to_range[1]
                    in_range_mask = ~ (smaller_mask & larger_mask).all(0)
                    
                    if in_range_mask.item():
                        successful_sample -= 1
                        model, likelihood = get_model_and_likelihood()
                        continue

                    successful_sample = True

                    feat = torch.cat([budgets.unsqueeze(-1), learning_curves, configurations], dim=-1)
                    # feat, targ = create_formatted_tensor(configurations, budgets, learning_curves, predictions, device=device)
                    # feat, targ = rearrange_tensor(feat, targ)
                    X.append(feat)
                    Y.append(predictions)
    
    x = torch.stack(X, dim=1).to(device).float()
    y = torch.stack(Y, dim=1).to(device).float()

    return Batch(x=x, y=y, target_y=y)

class MultiCurvesEncoder(torch.nn.Module):
    def __init__(self, num_features, emsize):
        super().__init__()
        self.budget_enc = torch.nn.Sequential(
            encoders.Normalize(1.0, 51.0),
            encoders.Normalize(0.5, math.sqrt(1 / 12)),
            torch.nn.Linear(1, emsize, bias=False)
        )
        self.lc_encoder = encoders.get_normalized_uniform_encoder(encoders.Linear)(50, emsize) # 50 is the number of epochs
        self.configuration_enc = encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear))(num_features - 51, emsize)

    def forward(self, x):
        # TODO
        # divided by the number of non zero values
        # to make the learning curve values in the same range as the other features.
        # lc = x[..., 1:51] * 50/lc[lc==0].size(-1)
        out = self.budget_enc(x[..., :1]) \
            + self.lc_encoder(x[..., 1:51]) \
            + self.configuration_enc(x[..., 51:])
        return out

def get_encoder():
    return lambda num_features, emsize: MultiCurvesEncoder(num_features, emsize)