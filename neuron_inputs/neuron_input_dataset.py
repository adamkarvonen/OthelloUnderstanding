# %%
# Setup
# Imports

import torch as t
import matplotlib.pyplot as plt
import importlib
import numpy as np
import einops
from collections import defaultdict

from circuits.utils import othello_hf_dataset_to_generator, get_model, get_aes
import circuits.othello_utils as othello_utils

from circuits.eval_sae_as_classifier import construct_othello_dataset


device = "cuda:0"
# device = "cpu"
repo_dir = "/share/u/can/chess-gpt-circuits"
repo_dir = "/home/adam/OthelloUnderstanding/"
tracer_kwargs = {"scan": False, "validate": False}

# Import model
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
model = get_model(model_name, device)
aes = get_aes(node_type="mlp_neuron", repo_dir=repo_dir)
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
    custom_functions=[othello_utils.games_batch_to_classifier_input_BLC],
    n_inputs=batch_size * n_batches,
    split="train",
    device=device,
)

t.set_grad_enabled(False)

# %%
# Cache Neuron Activations
layers = [0, 1]
neuron_acts = defaultdict(list)

for batch_idx in range(n_batches):
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    data_batch = data["encoded_inputs"][batch_start:batch_end]
    data_batch = t.tensor(data_batch, device=device)

    with t.no_grad(), model.trace(data_batch, **tracer_kwargs):
        for layer in layers:
            neuron_acts[layer].append(
                model.blocks[layer].mlp.hook_post.output.save()
            )  # could reduce to a smaller datatype eg b float16

# %%'

for layer in neuron_acts:
    neuron_acts[layer] = t.stack(neuron_acts[layer])
    neuron_acts[layer] = einops.rearrange(neuron_acts[layer], "n b l c -> (n b) l c")

print(neuron_acts[0].shape)

mlp_acts_BLD = neuron_acts[1]

games_BLC = data[othello_utils.games_batch_to_classifier_input_BLC.__name__]


import numpy as np
from sklearn.model_selection import train_test_split

X = einops.rearrange(games_BLC, "b l c -> (b l) c").cpu().numpy()
y = einops.rearrange(mlp_acts_BLD, "b l d -> (b l) d").cpu().numpy()

print(X.shape, y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Function to train and evaluate a model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# Linear Regression
linear_reg = LinearRegression()
linear_mse, linear_r2 = train_and_evaluate(linear_reg, X_train, X_test, y_train, y_test)
print(f"Linear Regression - MSE: {linear_mse}, R2: {linear_r2}")

# # Decision Tree
# dt_reg = MultiOutputRegressor(DecisionTreeRegressor(random_state=42))
# dt_mse, dt_r2 = train_and_evaluate(dt_reg, X_train, X_test, y_train, y_test)
# print(f"Decision Tree - MSE: {dt_mse}, R2: {dt_r2}")

# # Random Forest
# rf_reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
# rf_mse, rf_r2 = train_and_evaluate(rf_reg, X_train, X_test, y_train, y_test)
# print(f"Random Forest - MSE: {rf_mse}, R2: {rf_r2}")

# Regression tensor of shape
"""
b: batch_size, number of games
L: sequence length, number of moves per game
C: cell/square the most recent move was played on (J for just played) ++ all cells/squares that are occupied, all moves in the token string prior the most recent move
"""

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


"""
othello utils
get data batch
collect activations batch

"""
# %%
