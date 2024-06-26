# %%
# Setup
# Imports

import torch as t
import matplotlib.pyplot as plt
import importlib
import numpy as np
import einops

from circuits.utils import (
    othello_hf_dataset_to_generator,
    get_model,
    get_aes
)

from circuits.eval_sae_as_classifier import construct_othello_dataset


# device = 'cuda:0'
device = 'cpu'
repo_dir = '/share/u/can/chess-gpt-circuits'
tracer_kwargs = {'scan': False, 'validate': False}

# Import model
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = get_model(model_name, device)
aes = get_aes(node_type='mlp_neuron', repo_dir=repo_dir)
ae0 = aes[0]

# Load data
context_length = 59
activation_dim = 512  # output dimension of the layer
dict_size = 2048
batch_size = 10
n_batches = 5

dataset_name = "taufeeque/othellogpt"
# data = othello_hf_dataset_to_generator(
#     dataset_name, context_length=context_length, split="train", streaming=True
# )

data = construct_othello_dataset(
    custom_functions=[],
    n_inputs=batch_size*n_batches,
    split="train",
    device=device
    )



# %%
# Cache Neuron Activations
layers = [0, 1]
neuron_acts = {}

for batch_idx in range(n_batches):
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    data_batch = data['encoded_inputs'][batch_start:batch_end]
    data_batch = t.tensor(data_batch, device=device)

    with t.no_grad(), model.trace(data_batch, **tracer_kwargs):
        for layer in layers:
            neuron_acts[layer] = model.blocks[layer].mlp.hook_post.output.save() # could reduce to a smaller datatype eg b float16

print(neuron_acts[0].shape)
# %%
# Regression tensor of shape
'''
b: batch_size, number of games
L: sequence length, number of moves per game
C: cell/square the most recent move was played on (J for just played) ++ all cells/squares that are occupied, all moves in the token string prior the most recent move
'''

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Model training
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Predictions
# y_pred = model.predict(X_test)

# # Evaluation
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
                     

'''
othello utils
get data batch
collect activations batch

'''
# %%
