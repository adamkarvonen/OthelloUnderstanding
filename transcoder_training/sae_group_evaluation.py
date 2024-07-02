# %%
# Imports 

# %autoreload 2

import os
import torch as t
from collections import defaultdict
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pandas as pd
import gc
import json

from circuits.utils import (
    othello_hf_dataset_to_generator,
    get_model,
)
from circuits.dictionary_learning.buffer import NNsightActivationBuffer
from circuits.dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew, IdentityDict
from circuits.dictionary_learning.evaluation import evaluate
from circuits.utils import get_single_ae


# %%
transcoder = False
device='cuda:0'
repo_dir = '/share/u/can/OthelloUnderstanding'

node_type = "sae_feature"
# node_type = "mlp_neuron"

# ae_group_name = 'mlp_transcoder_all_layers_panneal_0628'
ae_group_name = 'mlp_out_sweep_all_layers_panneal_0628'
ae_type = 'p_anneal'

# %%
# model and data parameters
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
dataset_name = "taufeeque/othellogpt"
context_length = 59
activation_dim = 512  # output dimension of the layer

buffer_size = int(3e4 / 4)
llm_batch_size = 128 # 256 for A100 GPU, 64 for 1080ti
sae_batch_size = 8192
num_tokens = 200_000_000    

# Initialize model, data and activation buffer
model = get_model(model_name, device)
data = othello_hf_dataset_to_generator(
    dataset_name, context_length=context_length, split="train", streaming=True
)

# %%
# scan for all trainers in an ae_group
ae_group_dir = os.path.join(repo_dir, 'autoencoders', ae_group_name)
list_subfolders_with_paths = lambda directory: [f.path for f in os.scandir(directory) if f.is_dir()]
ae_group_trainers_per_layer = {}
for layer_name in list_subfolders_with_paths(ae_group_dir):
    layer_idx = int(layer_name.split('_')[-1])
    ae_group_trainers_per_layer[layer_idx] = list_subfolders_with_paths(layer_name)
ae_group_trainers_per_layer

# %%

evaluations = defaultdict(list)

for layer_idx in ae_group_trainers_per_layer:
    print(f"Layer {layer_idx}")
    t.cuda.empty_cache()
    gc.collect()

    if transcoder:
        io = "in_and_out"
        submodule = model.blocks[layer_idx].mlp
    else:
        io = "out"
        # submodule = model.blocks[layer].hook_resid_post # resid_post
        submodule = model.blocks[layer_idx].hook_mlp_out # resid_post
        # submodule = model.blocks[layer].mlp.hook_post # resid_pre

    activation_buffer = NNsightActivationBuffer(
        data,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )
    

    for trainer_name in tqdm(ae_group_trainers_per_layer[layer_idx]):
        evaluations['path'].append(trainer_name[len(repo_dir)+1:])
        evaluations['layer_idx'].append(layer_idx)
        evaluations['trainer_idx'].append(trainer_name.split('iner')[-1]) # TODO change convention to have an underscore before the trainer idx

        cfg_path = os.path.join(trainer_name, 'config.json')
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

        evaluations['cfg_sparsity_penalty'].append(cfg['trainer']['sparsity_penalty'])
        evaluations['cfg_learning_rate'].append(cfg['trainer']['lr'])
        evaluations['cfg_random_seed'].append(cfg['trainer']['seed'])

        ae = get_single_ae(trainer_name, ae_type=ae_type, device=device)
        eval_results = evaluate(
            ae,
            activation_buffer,
            max_len=context_length,
            batch_size=sae_batch_size,
            io=io,
            tracer_args={'scan': False, 'validate':False},
            device=device,
        )
        for metric in eval_results:
            evaluations[metric].append(eval_results[metric])

# %%
df_eval_ae_group = pd.DataFrame(evaluations)
df_eval_ae_group.to_csv(os.path.join(ae_group_dir, 'evaluations.csv'))
# %%
