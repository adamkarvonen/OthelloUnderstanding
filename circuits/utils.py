from ast import main
from dataclasses import dataclass
from huggingface_hub import hf_hub_download
import torch
from nnsight import NNsight
import json
from typing import Any
from datasets import load_dataset
from einops import rearrange
from jaxtyping import Int, Float, jaxtyped
from torch import Tensor
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from transformer_lens import HookedTransformer
from transformer_lens import utils as tl_utils
from enum import Enum
from typing import Optional, Union
import pandas as pd

from circuits.dictionary_learning.buffer import NNsightActivationBuffer
from circuits.chess_utils import encode_string
from circuits.dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    IdentityDict,
)

# These imports are required for the current hacky way we are loading SAE classes
from circuits.dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew
from circuits.dictionary_learning.trainers.top_k import AutoEncoderTopK
from circuits.dictionary_learning.trainers.gated_anneal import GatedAnnealTrainer
from circuits.dictionary_learning.trainers.gdm import GatedSAETrainer
from circuits.dictionary_learning.trainers.p_anneal import PAnnealTrainer
from circuits.dictionary_learning.trainers.standard import StandardTrainer
# from circuits.dictionary_learning.trainers.p_anneal_new import PAnnealTrainerNew
# from circuits.dictionary_learning.trainers.standard_new import StandardTrainerNew


@dataclass
class AutoEncoderBundle:
    ae: AutoEncoder
    buffer: Optional[NNsightActivationBuffer]
    model: NNsight
    activation_dim: int
    dictionary_size: int
    context_length: int
    submodule: Any


class SubmoduleType(Enum):
    resid_post = "resid_post"
    mlp_act = "mlp_act"


def get_model(model_name: str, device: torch.device) -> NNsight:
    if model_name == "Baidicoot/Othello-GPT-Transformer-Lens":
        tf_model = HookedTransformer.from_pretrained("Baidicoot/Othello-GPT-Transformer-Lens")
        model = NNsight(tf_model).to(device)
        return model
    
    if model_name == "mntss/Othello-GPT":
        tf_model = HookedTransformer.from_pretrained_no_processing("Baidicoot/Othello-GPT-Transformer-Lens")
        state_dict = tl_utils.download_file_from_hf("mntss/Othello-GPT", "tl_model.pth")
        tf_model.load_state_dict(state_dict)
        model = NNsight(tf_model).to(device)
        return model

    if (
        model_name == "adamkarvonen/RandomWeights8LayerOthelloGPT2"
        or model_name == "adamkarvonen/RandomWeights8LayerChessGPT2"
    ):
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    if model_name == "adamkarvonen/8LayerChessGPT2":
        # Old method of loading model from nanogpt weights
        # model = convert_nanogpt_model(
        #     f"{model_path}lichess_8layers_ckpt_no_optimizer.pt", torch.device(device)
        # )
        # tokenizer = NanogptTokenizer(meta_path=f"{model_path}meta.pkl")
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        model = NNsight(model).to(device)
        return model

    raise ValueError("Model not found.")


def get_mlp_activations_submodule(model_name: str, layer: int, model: NNsight) -> Any:
    if model_name in ["Baidicoot/Othello-GPT-Transformer-Lens", "mntss/Othello-GPT"]:
        return model.blocks[layer].mlp.hook_post
    if model_name in [
        "adamkarvonen/8LayerChessGPT2",
        "adamkarvonen/RandomWeights8LayerOthelloGPT2",
        "adamkarvonen/RandomWeights8LayerChessGPT2",
    ]:
        return model.transformer.h[layer].mlp.act
    raise ValueError("Model not found.")


def get_resid_post_submodule(model_name: str, layer: int, model: NNsight) -> Any:
    if model_name in ["Baidicoot/Othello-GPT-Transformer-Lens", "mntss/Othello-GPT"]:
        return model.blocks[layer].hook_resid_post
    if model_name in [
        "adamkarvonen/8LayerChessGPT2",
        "adamkarvonen/RandomWeights8LayerOthelloGPT2",
        "adamkarvonen/RandomWeights8LayerChessGPT2",
    ]:
        return model.transformer.h[layer]  # residual stream after the layer
    raise ValueError("Model not found.")


def get_submodule(
    model_name: str, layer: int, model: NNsight, submodule_type: Optional[SubmoduleType] = None
) -> Any:
    if submodule_type is None or submodule_type == SubmoduleType.resid_post:
        return get_resid_post_submodule(model_name, layer, model)
    elif submodule_type == SubmoduleType.mlp_act:
        return get_mlp_activations_submodule(model_name, layer, model)
    else:
        raise ValueError("submodule_type not recognized")


def concatenate_csv_files(file_list: list[str], output_file: str):
    # Load and concatenate the CSV files
    dataframes = [pd.read_csv(file) for file in file_list]
    concatenated_df = pd.concat(dataframes)

    # Save the concatenated data frame to a new CSV file
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated data saved to {output_file}")


def get_identity_autoencoder(config: dict) -> IdentityDict:
    """The identity autoencoder just returns activations as is. We can use this to run the full pipeline
    on GPT activations rather than autoencoder activations.
    To use this, run `autoencoders/gen_identity_ae_folders.ipynb` to generate the identity autoencoder folders.
    Then, in full_pipeline.ipynb, set:
    autoencoder_group_paths = ["../autoencoders/chess_mlp_acts_identity_aes/"]
    csv_output_path = "../autoencoders/chess_mlp_acts_identity_aes/results.csv"
    """
    ae = IdentityDict()
    ae.activation_dim = config["trainer"]["activation_dim"]
    ae.dict_size = config["trainer"]["dict_size"]
    return ae


def get_ae_bundle(
    autoencoder_path: str,
    device: torch.device,
    data: Any,  # iter of list of ints
    batch_size: int,
    n_ctxs: int = 512,
    include_buffer: bool = True,
) -> AutoEncoderBundle:
    autoencoder_model_path = f"{autoencoder_path}ae.pt"
    autoencoder_config_path = f"{autoencoder_path}config.json"

    with open(autoencoder_config_path, "r") as f:
        config = json.load(f)

    use_identity_dict = False

    if "dict_class" in config["trainer"]:
        if config["trainer"]["dict_class"] == "Identity":
            use_identity_dict = True

    if use_identity_dict:
        ae = get_identity_autoencoder(config)
    else:
        # rangell: this is a super hacky way to get the correct dictionary class from the config
        # TODO (rangell): make this better in the future.
        # old: ae = AutoEncoder.from_pretrained(autoencoder_model_path, device=device)
        config_args = []
        for k, v in config["trainer"].items():
            if k not in ["trainer_class", "sparsity_penalty"]:
                if isinstance(v, str) and k != "dict_class":
                    config_args.append(k + "=" + "'" + v + "'")
                else:
                    config_args.append(k + "=" + str(v))
        config_str = ", ".join(config_args)
        ae = eval(
            config["trainer"]["trainer_class"] + f"({config_str})"
        ).ae.__class__.from_pretrained(autoencoder_model_path, device=device)
        ae = ae.to(device)

    model_name = config["trainer"]["lm_name"]

    layer = config["trainer"]["layer"]

    # The following commented lines are for some legacy autoencoders
    # that don't have the layer specified in the config file.
    # if "layer_0" in autoencoder_model_path:
    #     layer = 0
    # elif "layer_1" in autoencoder_model_path:
    #     layer = 1
    # elif "layer_2" in autoencoder_model_path:
    #     layer = 2
    # elif "layer_3" in autoencoder_model_path:
    #     layer = 3
    # elif "layer_4" in autoencoder_model_path:
    #     layer = 4
    # elif "layer_5" in autoencoder_model_path:
    #     layer = 5
    # elif "layer_6" in autoencoder_model_path:
    #     layer = 6
    # elif "layer_7" in autoencoder_model_path:
    #     layer = 7
    # elif "layer_8" in autoencoder_model_path:
    #     layer = 8
    # else:
    #     raise ValueError("layer not specified in autoencoder_model_path")

    print("Evaluating on layer: ", layer)

    activation_dim = ae.activation_dim
    dictionary_size = ae.dict_size

    model = get_model(model_name, device)

    if activation_dim == 512:
        submodule_type = SubmoduleType.resid_post
        print("Using resid_post submodule")
    elif activation_dim == 512 * 4:
        submodule_type = SubmoduleType.mlp_act
        print("Using mlp_act submodule")
    else:
        raise ValueError("activation_dim not recognized")

    submodule = get_submodule(model_name, layer, model, submodule_type)

    context_length = config["buffer"]["ctx_len"]

    if include_buffer:
        buffer = NNsightActivationBuffer(
            data,
            model,
            submodule,
            n_ctxs=n_ctxs,
            ctx_len=context_length,
            refresh_batch_size=batch_size,
            io="out",
            d_submodule=activation_dim,
            device=device,
            out_batch_size=batch_size,
        )
    else:
        buffer = None

    return AutoEncoderBundle(
        ae=ae,
        buffer=buffer,
        model=model,
        activation_dim=activation_dim,
        dictionary_size=dictionary_size,
        context_length=context_length,
        submodule=submodule,
    )


def get_first_n_dataset_rows(dataset_name: str, n: int, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        count = 0
        for x in iter(dataset):
            if count >= n:
                break
            yield x["text"]
            count += 1

    return gen()


@torch.no_grad()
def get_feature(
    activations,
    ae: AutoEncoder,
    device,
) -> torch.Tensor:
    try:
        x = next(activations).to(device)
    except StopIteration:
        raise StopIteration(
            "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
        )

    x_hat, f = ae(x, output_features=True)

    return f


def get_firing_features(
    ae_bundle: AutoEncoderBundle,
    total_inputs: int,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Note: total inputs means the number of model activations, not the number of inputs to the model.
    total_inputs == n_inputs * context_length.
    For sparse autoencoders with larger expansion factors (16+), over 75% of the features can be dead.
    """

    num_iters = total_inputs // batch_size
    max_features = torch.full((ae_bundle.dictionary_size,), float("-inf"), device=device)

    features_F = torch.zeros((ae_bundle.dictionary_size,), device=device)
    for i in tqdm(range(num_iters), desc="Collecting features"):
        feature_BF = get_feature(ae_bundle.buffer, ae_bundle.ae, device)
        max_features = torch.max(max_features, feature_BF.max(dim=0).values)
        features_F += (feature_BF != 0).float().sum(dim=0)

    features_F /= total_inputs

    assert features_F.shape[0] == ae_bundle.dictionary_size

    mask = features_F > threshold

    alive_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    max_features = max_features[alive_indices]

    # Rarely MLP neurons can have negative max values. This is a simple fix.
    max_features = torch.abs(max_features)

    return alive_indices, max_features


# TODO: This should take a list of dictionaries as input. Maybe in ae_bundle?
# On second thought, activation collection ends up being a relatively cheap operation
# compared to board state aggregation. I'll leave it as is for now.
@torch.no_grad()
def collect_activations_batch(
    ae_bundle: AutoEncoderBundle,
    inputs_BL: torch.Tensor,
    dims: Int[Tensor, "num_dims"],
) -> tuple[Float[Tensor, "num_dims batch_size max_length"], Int[Tensor, "batch_size max_length"]]:
    with ae_bundle.model.trace(
        inputs_BL, invoker_args=dict(max_length=ae_bundle.context_length, truncation=True)
    ):
        cur_tokens = ae_bundle.model.input[0].save()
        cur_activations = ae_bundle.submodule.output
        if type(cur_activations.shape) == tuple:
            cur_activations = cur_activations[0]
        cur_activations = ae_bundle.ae.encode(cur_activations)
        cur_activations_BLF = cur_activations[
            :, :, dims
        ].save()  # Shape: (batch_size, max_length, dim_count)
    cur_activations_FBL = rearrange(
        cur_activations_BLF.value, "b n d -> d b n"
    )  # Shape: (dim_count, batch_size, max_length)

    return cur_activations_FBL, cur_tokens.value[0]


@torch.no_grad()
def get_model_activations(
    ae_bundle: AutoEncoderBundle,
    inputs_AL: torch.Tensor,
    batch_size: int,
) -> tuple[Float[Tensor, "num_dims batch_size max_length"], Int[Tensor, "batch_size max_length"]]:
    batch_results = []
    for i in range(inputs_AL.shape[0] // batch_size + 1):
        start = i * batch_size
        end = min((i + 1) * batch_size, inputs_AL.shape[0])
        if start == end:
            break
        with ae_bundle.model.trace(
            inputs_AL[start:end],
            invoker_args=dict(max_length=ae_bundle.context_length, truncation=True),
        ):
            cur_activations = ae_bundle.submodule.output.save()
            if type(cur_activations.shape) == tuple:
                cur_activations = cur_activations[0]
        batch_results.append(cur_activations.value)
    cur_activations = torch.concat(batch_results)
    return cur_activations


@torch.no_grad()
def get_feature_activations_batch(
    ae_bundle: AutoEncoderBundle,
    model_activations_BLD: torch.Tensor,
    dims: Int[Tensor, "num_dims"],
) -> tuple[Float[Tensor, "num_dims batch_size max_length"], Int[Tensor, "batch_size max_length"]]:
    feature_activations = ae_bundle.ae.encode(model_activations_BLD)
    feature_activations_BLF = feature_activations[:, :, dims]
    feature_activations_FBL = rearrange(
        feature_activations_BLF, "b n d -> d b n"
    )  # Shape: (dim_count, batch_size, max_length)
    return feature_activations_FBL


def get_nested_folders(path: str) -> list[str]:
    """Get a list of folders nested one level deep in the given path which contain an ae.pt file"""
    folder_names = []
    # Process current directory and one level deep subdirectories
    for folder in os.listdir(path):
        if folder == "utils":
            continue
        current_folder = os.path.join(path, folder)
        if os.path.isdir(current_folder):
            if "ae.pt" in os.listdir(current_folder):
                folder_names.append(current_folder + "/")
            for subfolder in os.listdir(current_folder):  # Process subfolders
                subfolder_path = os.path.join(current_folder, subfolder)
                if os.path.isdir(subfolder_path):
                    if "ae.pt" in os.listdir(subfolder_path):
                        folder_names.append(subfolder_path + "/")

    return folder_names


def to_device(data, device):
    """
    Recursively move tensors in a nested dictionary to CPU.
    """
    if isinstance(data, dict):
        # If it's a dictionary, apply recursively to each value
        return {key: to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        # If it's a list, apply recursively to each element
        return [to_device(item, device) for item in data]
    elif isinstance(data, torch.Tensor):
        # If it's a tensor, move it to CPU
        return data.to(device)
    else:
        # If it's neither, return it as is
        return data


def othello_hf_dataset_to_generator(
    dataset_name: str, context_length: int = 59, split="train", streaming=True
):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["tokens"][:context_length]

    return gen()


def chess_hf_dataset_to_generator(
    dataset_name: str, meta: dict, context_length: int = 256, split="train", streaming=True
):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield encode_string(meta, x["text"][:context_length])

    return gen()


def get_model_name(othello: bool) -> str:
    if othello:
        return "Baidicoot/Othello-GPT-Transformer-Lens"
    else:
        return "adamkarvonen/8LayerChessGPT2"


def get_single_ae(
    full_path: str, ae_type: str, device: str
) -> Union[AutoEncoder, GatedAutoEncoder, AutoEncoderNew, IdentityDict]:
    # Initialize the autoencoder
    if ae_type == "standard" or ae_type == "p_anneal":
        ae = AutoEncoder.from_pretrained(os.path.join(full_path, "ae.pt"), device=device)
    elif ae_type == "gated" or ae_type == "gated_anneal":
        ae = GatedAutoEncoder.from_pretrained(os.path.join(full_path, "ae.pt"), device=device)
    elif ae_type == "standard_new":
        ae = AutoEncoderNew.from_pretrained(os.path.join(full_path, "ae.pt"), device=device)
    elif ae_type == "topk":
        config = tl_utils.download_file_from_hf("mntss/Othello-SAEs", os.path.join(full_path, "config.json"))
        state_dict = tl_utils.download_file_from_hf("mntss/Othello-SAEs", os.path.join(full_path, "ae.pt"))
        return AutoEncoderTopK.from_pretrained(state_dict, config['trainer']['k'], device=device)
    elif ae_type == "identity":
        ae = IdentityDict()
    else:
        raise ValueError("Invalid ae_type")
    return ae


def get_ae(
    layer: int,
    node_type: str,
    repo_dir: str,
    trainer_id: int,
    ae_group_name: Optional[str] = None,
    return_ae_group_dir: bool = False,
    device: str = "cpu",
):
    """A default trainer_id is 0 if unsure."""
    if node_type == "sae_feature":
        if ae_group_name is None:
            ae_group_name = "othello_all_layers_p_anneal_0524"
        ae_group_dir = f"{repo_dir}autoencoders/{ae_group_name}"
        ae_type = "p_anneal"
        ae_path = f"{ae_group_dir}/layer_{layer}/trainer{trainer_id}"
    elif node_type == "sae_mlp_feature":
        if ae_group_name is None:
            ae_group_name = "othello_all_mlps_p_anneal_0531"
        ae_group_dir = f"{repo_dir}autoencoders/{ae_group_name}"
        ae_type = "p_anneal"
        ae_path = f"{ae_group_dir}/layer_{layer}"
    elif node_type == "sae_mlp_out_feature":
        if ae_group_name is None:
            ae_group_name = "mlp_out_sweep_all_layers_panneal_0628"
        ae_group_dir = f"{repo_dir}autoencoders/{ae_group_name}"
        ae_type = "p_anneal"
        ae_path = f"{ae_group_dir}/layer_{layer}/trainer{trainer_id}"
    elif node_type == "transcoder":
        if ae_group_name is None:
            ae_group_name = "mlp_transcoder_all_layers_panneal_0628"
        ae_group_dir = f"{repo_dir}autoencoders/{ae_group_name}"
        ae_type = "p_anneal"
        ae_path = f"{ae_group_dir}/layer_{layer}/trainer{trainer_id}"
    elif node_type == "mlp_neuron":
        if ae_group_name is None:
            # append _lines to ae_group_name to get lines data results
            ae_group_name = "othello_mlp_acts_identity_aes"
        ae_group_dir = f"{repo_dir}autoencoders/{ae_group_name}"
        ae_type = "identity"
        ae_path = f"{ae_group_dir}/layer_{layer}"
    elif node_type == "attention_out" or node_type == "mlp_out":
        if ae_group_name is None:
            ae_group_name = "othello_resid_acts_identity_aes"
        ae_group_dir = f"{repo_dir}autoencoders/{ae_group_name}"
        ae_type = "identity"
        ae_path = f"{ae_group_dir}/layer_{layer}"
    elif node_type == "sae_feature_topk":
        assert ae_group_name is None
        ae_group_dir = None
        ae_type = "topk"
        ae_path = f"resid_post/layer_{layer}/trainer_{trainer_id}"
    else:
        raise ValueError("Invalid node_type")

    # download data from huggingface if needed
    if not os.path.exists(f"{repo_dir}autoencoders/{ae_group_name}") and ae_group_name is not None:
        hf_hub_download(
            repo_id="adamkarvonen/othello_saes",
            filename=f"{ae_group_name}.zip",
            local_dir=f"{repo_dir}autoencoders",
        )
        # unzip the data
        os.system(f"unzip {repo_dir}autoencoders/{ae_group_name}.zip -d {repo_dir}autoencoders")

    ae = get_single_ae(ae_path, ae_type, device)

    if return_ae_group_dir:
        return ae, ae_group_dir
    else:
        return ae


def get_aes(node_type: str, repo_dir: str, trainer_id: int):
    aes = []
    for layer in range(8):
        ae = get_ae(layer, node_type, repo_dir, trainer_id=trainer_id)
        aes.append(ae)
    return aes
