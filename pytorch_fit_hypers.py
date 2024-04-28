import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import MarginalLogLikelihood
import pickle
import numpy as np
from botorch import fit_gpytorch_model
from botorch.models.gp_regression import SingleTaskGP
import torch

DATA_TO_LOAD = "/tirl/experiments/mpc_frozencartpole_2024-04-23/21-47-25/seed_0/test_data.p"
data = pickle.load(open(DATA_TO_LOAD, "rb"))
train_inputs = torch.tensor(np.array(data.x))
train_outputs = torch.tensor(np.array(data.y))
for model_ix in range(train_outputs.shape[1]):
    train_targets = train_outputs[:, model_ix]
    kernel = ScaleKernel(RBFKernel(ard_num_dims=5))
    model = SingleTaskGP(train_X=train_inputs,
                    train_Y=train_targets.view(-1,1),
                    covar_module=kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    
    print(model_ix, model.covar_module.outputscale, model.covar_module.base_kernel.lengthscale)