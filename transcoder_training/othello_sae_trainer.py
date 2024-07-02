# Simplified version of train sae with multiple hyperparameters in parallel

import argparse
import torch as t
import gc
import itertools

from circuits.utils import (
    othello_hf_dataset_to_generator,
    get_model,
)

from circuits.dictionary_learning.buffer import NNsightActivationBuffer
from circuits.chess_utils import encode_string
from circuits.dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    IdentityDict,
)
from circuits.dictionary_learning.training import trainSAE

# These imports are required for the current hacky way we are loading SAE classes
from circuits.dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew
from circuits.dictionary_learning.trainers.gated_anneal import GatedAnnealTrainer
from circuits.dictionary_learning.trainers.gdm import GatedSAETrainer
from circuits.dictionary_learning.trainers.p_anneal import PAnnealTrainer
from circuits.dictionary_learning.trainers.p_anneal_new import PAnnealTrainerNew
from circuits.dictionary_learning.trainers.standard import StandardTrainer
from circuits.dictionary_learning.trainers.p_anneal_new import PAnnealTrainerNew
from circuits.dictionary_learning.trainers.standard_new import StandardTrainerNew
from circuits.dictionary_learning.trainers.transcoder_p_anneal import TranscoderPAnnealTrainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True,
                        help="where to store sweep")
    parser.add_argument("--transcoder", action="store_true",  help="train transcoder if flag is set, SAE otherwise")
    parser.add_argument("--no_wandb_logging", action="store_true",  help="omit wandb logging")
    parser.add_argument("--dry_run", action="store_true",  help="dry run sweep")
    args = parser.parse_args()
    return args


def run_sae_training(
        layer : int,
        save_dir : str,
        device : str,
        dry_run : bool = False,
        transcoder : bool = False,
        no_wandb_logging : bool = False,
):
   
    # model and data parameters
    model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
    dataset_name = "taufeeque/othellogpt"
    context_length = 59
    activation_dim = 512  # output dimension of the layer

    buffer_size = int(3e4 / 4)
    llm_batch_size = 128 # 256 for A100 GPU, 64 for 1080ti
    sae_batch_size = 8192
    num_tokens = 200_000_000

    # sae training parameters
    # random_seeds = t.arange(10).tolist()
    random_seeds = [0]
    initial_sparsity_penalties = t.linspace(.008, .1, 7).tolist()
    learning_rates = [0.003, 0.0003, 0.0003]


    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    save_steps = None
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None
    p_start = 1
    p_end = 0.2
    anneal_end = None  # steps - int(steps/10)
    expansion_factor = 8
    sparsity_queue_length = 10
    anneal_start = 1000
    n_sparsity_updates = 10

    log_steps = 5 # Log the training on wandb
    if no_wandb_logging:
        log_steps = None


    # Initialize model, data and activation buffer
    model = get_model(model_name, device)
    data = othello_hf_dataset_to_generator(
        dataset_name, context_length=context_length, split="train", streaming=True
    )
    if transcoder:
        trainer_class = TranscoderPAnnealTrainer
        io = "in_and_out"
        submodule = model.blocks[layer].mlp
    else:
        trainer_class = PAnnealTrainer
        io = "out"
        # submodule = model.blocks[layer].hook_resid_post 
        # submodule = model.blocks[layer].hook_mlp_out
        submodule = model.blocks[layer].hook_attn_out
        # submodule = model.blocks[layer].mlp.hook_post # latent activations after non-linearity

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

    # create the list of configs
    trainer_configs = []
    for seed, initial_sparsity_penalty, learning_rate in itertools.product(random_seeds, initial_sparsity_penalties, learning_rates):
        trainer_configs.append(
            {
                "trainer": trainer_class,
                "dict_class": AutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": expansion_factor * activation_dim,
                "lr": learning_rate,
                "sparsity_function": "Lp^p",
                "initial_sparsity_penalty": initial_sparsity_penalty,
                "p_start": p_start,
                "p_end": p_end,
                "anneal_start": int(anneal_start),
                "anneal_end": anneal_end,
                "sparsity_queue_length": sparsity_queue_length,
                "n_sparsity_updates": n_sparsity_updates,
                "warmup_steps": warmup_steps,
                "resample_steps": resample_steps,
                "steps": steps,
                "seed": seed,
                "wandb_name": f"PAnnealTrainer-othello-mlp-layer-{layer}",
                "layer" : layer,
                "lm_name" : model_name,
                "device": device,
            }
        )


    print(f"len trainer configs: {len(trainer_configs)}")
    save_dir = f'{save_dir}/layer_{layer}'

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            transcoder=transcoder,
        )

if __name__ == "__main__":
    args = get_args()
    for layer in range(8):
        run_sae_training(
            layer=layer,
            save_dir=args.save_dir,
            device="cuda:0",
            dry_run=args.dry_run,
            no_wandb_logging=args.no_wandb_logging,
            transcoder=args.transcoder,
        )
    t.cuda.empty_cache()
    gc.collect()