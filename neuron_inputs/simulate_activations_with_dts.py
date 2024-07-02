import torch
import numpy as np
import einops
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from joblib import Parallel, delayed
import multiprocessing
from typing import Callable, Optional
import os
import pickle
import itertools

from xgboost import XGBRegressor, XGBClassifier

import circuits.utils as utils
import circuits.othello_utils as othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset

# Setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.set_grad_enabled(False)
tracer_kwargs = {"validate": False, "scan": False}
tracer_kwargs = {"validate": True, "scan": True}


def load_model_and_data(model_name: str, dataset_size: int, custom_functions: list[Callable]):
    model = utils.get_model(model_name, device)
    data = construct_othello_dataset(
        custom_functions=custom_functions,
        n_inputs=dataset_size,
        split="train",
        device=device,
    )
    return model, data


# Cache Neuron Activations
def cache_neuron_activations(
    model, data: dict, layers: list, batch_size: int, n_batches: int
) -> dict:
    """Deprecated in favor of using identity autoencoders"""
    neuron_acts = defaultdict(list)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        data_batch = data["encoded_inputs"][batch_start:batch_end]
        data_batch = torch.tensor(data_batch, device=device)

        with torch.no_grad(), model.trace(data_batch, scan=False, validate=False):
            for layer in layers:
                neuron_activations_BLD = model.blocks[layer].mlp.hook_post.output.save()
                neuron_acts[layer].append(neuron_activations_BLD)

    for layer in neuron_acts:
        neuron_acts[layer] = torch.stack(neuron_acts[layer])
        neuron_acts[layer] = einops.rearrange(neuron_acts[layer], "n b l c -> (n b) l c")

    return neuron_acts


def get_submodule_dict(model, model_name: str, layers: list, input_location: str) -> dict:
    submodule_dict = {}

    for layer in layers:
        if input_location == "sae_feature":
            submodule = utils.get_resid_post_submodule(model_name, layer, model)
        elif input_location == "sae_mlp_feature":
            submodule = utils.get_mlp_activations_submodule(model_name, layer, model)
        elif input_location == "mlp_neuron":
            submodule = utils.get_mlp_activations_submodule(model_name, layer, model)
        elif input_location == "attention_out":
            submodule = model.blocks[layer].hook_attn_out
        elif input_location == "mlp_out" or input_location == "sae_mlp_out_feature":
            submodule = model.blocks[layer].hook_mlp_out
        elif input_location == "transcoder":
            submodule = model.blocks[layer].mlp
        else:
            raise ValueError(f"Invalid input location: {input_location}")
        submodule_dict[layer] = submodule

    return submodule_dict


def cache_sae_activations(
    model,
    model_name: str,
    data: dict,
    layers: list,
    batch_size: int,
    n_batches: int,
    repo_dir: str,
    input_location: str,
    ae_dict: dict,
    submodule_dict: dict,
) -> dict:
    sae_acts = defaultdict(list)

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size
        data_batch = data["encoded_inputs"][batch_start:batch_end]
        data_batch = torch.tensor(data_batch, device=device)

        with torch.no_grad(), model.trace(data_batch, **tracer_kwargs):
            for layer in layers:
                ae = ae_dict[layer]
                submodule = submodule_dict[layer]
                if input_location != "transcoder":
                    x = submodule.output
                else:
                    x = submodule.input[0]
                    if type(submodule.input.shape) == tuple:
                        x = x[0]
                f = ae.encode(x)
                sae_acts[layer].append(f.save())

    for layer in sae_acts:
        sae_acts[layer] = torch.stack(sae_acts[layer])
        sae_acts[layer] = einops.rearrange(sae_acts[layer], "n b l c -> (n b) l c")

    return sae_acts


def get_max_activations(neuron_acts: dict, layer: int) -> torch.Tensor:
    D = neuron_acts[layer].shape[-1]
    max_activations_D = torch.full((D,), float("-inf"), device=device)

    neuron_acts_BLD = neuron_acts[layer]
    neuron_acts_BD = einops.rearrange(neuron_acts_BLD, "b l d -> (b l) d")

    max_activations_D = torch.max(max_activations_D, neuron_acts_BD.max(dim=0).values)
    return max_activations_D


def calculate_binary_activations(neuron_acts: dict, threshold: float = 0.1):
    binary_acts = {}

    for layer in neuron_acts:
        max_activations_D = get_max_activations(neuron_acts, layer)

        binary_acts[layer] = (neuron_acts[layer] > (threshold * max_activations_D)).int()
    return binary_acts


# Prepare data for modeling
def prepare_data(games_BLC: torch.Tensor, mlp_acts_BLD: torch.Tensor):
    X = einops.rearrange(games_BLC, "b l c -> (b l) c").cpu().numpy()
    y = einops.rearrange(mlp_acts_BLD, "b l d -> (b l) d").cpu().numpy()
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Train and evaluate models
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2


def train_and_evaluate_xgb(X_train, X_test, y_train, y_test, is_binary=False):
    if is_binary:
        model = MultiOutputClassifier(
            XGBClassifier(
                n_estimators=25,  # limit number of trees
                max_depth=6,  # limit depth
                learning_rate=0.1,
                random_state=42,
            )
        )
    else:
        model = MultiOutputRegressor(
            XGBRegressor(n_estimators=25, max_depth=6, learning_rate=0.1, random_state=42)
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_binary:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        return model, accuracy, f1
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, r2


def calculate_neuron_metrics(model, X, y):
    y_pred = model.predict(X)

    # Calculate MSE for all neurons at once
    mse_list = np.mean((y - y_pred) ** 2, axis=0)

    # Calculate R2 for all neurons at once
    ss_res = np.sum((y - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)

    # Add divide-by-zero protection
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_list = 1 - (ss_res / ss_tot)

    # Handle cases where ss_tot is zero
    r2_list = np.where(ss_tot == 0, 0, r2_list)

    # Clip R2 values to be between 0 and 1
    r2_list = np.clip(r2_list, 0, 1)

    return mse_list, r2_list


def calculate_binary_metrics(model, X, y):
    y_pred = model.predict(X)

    # Compute true positives, false positives, true negatives, false negatives
    tp = np.sum((y_pred == 1) & (y == 1), axis=0)
    fp = np.sum((y_pred == 1) & (y == 0), axis=0)
    tn = np.sum((y_pred == 0) & (y == 0), axis=0)
    fn = np.sum((y_pred == 0) & (y == 1), axis=0)

    # Compute metrics
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)

    # Compute F1 score
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )

    return accuracy, precision, recall, f1


# Print decision tree rules
def print_decision_tree_rules(model, feature_names, neuron_index, max_depth=None):
    tree = model.estimators_[neuron_index]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree.n_features_in_)]
    tree_rules = export_text(tree, feature_names=feature_names, max_depth=max_depth)
    print(f"Decision Tree Rules for Neuron {neuron_index}:")
    print(tree_rules)


def rc_to_square_notation(row, col):
    letters = "ABCDEFGH"
    letter = letters[row]
    number = 8 - col
    # letter = letters[col]
    return f"{letter}{number}"


def idx_to_square_notation(idx):
    row = idx // 8
    col = idx % 8
    square = rc_to_square_notation(row, col)
    return square


def compute_kl_divergence(logits_clean_BLV, logits_patch_BLV):
    # Apply softmax to get probability distributions
    log_probs_clean_BLV = torch.nn.functional.log_softmax(logits_clean_BLV, dim=-1)
    log_probs_patch_BLV = torch.nn.functional.log_softmax(logits_patch_BLV, dim=-1)

    # Compute KL divergence
    kl_div_BLV = torch.nn.functional.kl_div(
        log_probs_patch_BLV, log_probs_clean_BLV.exp(), reduction="none", log_target=False
    )

    # Sum over the vocabulary dimension
    kl_div_BL = kl_div_BLV.sum(dim=-1)

    return kl_div_BL


def add_output_folders():
    os.makedirs("decision_trees", exist_ok=True)
    os.makedirs("images", exist_ok=True)


def process_layer(
    layer: int,
    games_BLC: torch.Tensor,
    neuron_acts: dict,
    binary_acts: dict,
    linear_reg: bool = False,
    regular_dt: bool = True,
    binary_dt: bool = True,
    max_depth: int = 8,
    random_seed: int = 42,
) -> dict:

    print(f"\nLayer {layer}")

    if regular_dt:
        X_train, X_test, y_train, y_test = prepare_data(games_BLC, neuron_acts[layer])

        # Decision Tree
        dt_model, dt_mse, dt_r2 = train_and_evaluate(
            MultiOutputRegressor(
                DecisionTreeRegressor(
                    random_state=random_seed,
                    max_depth=max_depth,  # min_samples_leaf=5, min_samples_split=5
                )
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )

        dt_mse, dt_r2 = calculate_neuron_metrics(dt_model, X_test, y_test)

    if binary_dt:
        # Binary Decision Tree
        X_binary_train, X_binary_test, y_binary_train, y_binary_test = prepare_data(
            games_BLC, binary_acts[layer]
        )
        dt_binary_model = MultiOutputClassifier(
            DecisionTreeClassifier(
                random_state=random_seed,
                max_depth=max_depth,  # min_samples_leaf=5, min_samples_split=5
            )
        )
        dt_binary_model.fit(X_binary_train, y_binary_train)

        accuracy, precision, recall, f1 = calculate_binary_metrics(
            dt_binary_model, X_binary_test, y_binary_test
        )

    layer_results = {
        "layer": layer,
        "regular_dt": {"model": dt_model, "mse": dt_mse, "r2": dt_r2},
        "binary_dt": {
            "model": dt_binary_model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }

    if linear_reg:
        lasso_model, lasso_mse, lasso_r2 = train_and_evaluate(
            Lasso(alpha=0.005), X_train, X_test, y_train, y_test
        )
        layer_results["lasso"] = {"model": lasso_model, "mse": lasso_mse, "r2": lasso_r2}

    print(f"Finished Layer {layer}")

    return layer_results


def process_layer_xgb(
    layer: int,
    games_BLC: torch.Tensor,
    neuron_acts: dict,
    binary_acts: dict,
    linear_reg: bool = False,
) -> dict:

    print(f"\nLayer {layer}")
    X_train, X_test, y_train, y_test = prepare_data(games_BLC, neuron_acts[layer])

    # Decision Tree
    xgb_model, xgb_mse, xgb_r2 = train_and_evaluate_xgb(X_train, X_test, y_train, y_test)

    # Binary Decision Tree
    X_binary_train, X_binary_test, y_binary_train, y_binary_test = prepare_data(
        games_BLC, binary_acts[layer]
    )
    xgb_binary_model, xgb_accuracy, xgb_f1 = train_and_evaluate_xgb(
        X_binary_train, X_binary_test, y_binary_train, y_binary_test, is_binary=True
    )

    layer_results = {
        "layer": layer,
        "regular_dt": {"model": xgb_model, "mse": xgb_mse, "r2": xgb_r2},
        "binary_dt": {
            "model": xgb_binary_model,
            "accuracy": xgb_accuracy,
            "f1": xgb_f1,
        },
    }

    return layer_results


def interventions(
    model,
    train_data: dict,
    selected_features: dict[int, torch.Tensor],
    ae_dict: dict,
    submodule_dict: dict,
    layers: list[int],
    input_location: str,
    ablation_method: str = "zero",
    decision_trees: Optional[dict] = None,
    custom_function: Optional[Callable] = None,
    ablate_not_selected: bool = False,
    add_error: bool = False,
):
    allowed_methods = ["mean", "zero", "max", "dt"]
    assert (
        ablation_method in allowed_methods
    ), f"Invalid ablation method. Must be one of {allowed_methods}"
    game_batch_BL = torch.tensor(train_data["encoded_inputs"])

    simulated_activations = {}
    mean_activations = {}
    max_activations = {}

    if ablation_method == "dt":
        board_state_BLC = train_data[custom_function.__name__]
        B, L, C = board_state_BLC.shape
        X = einops.rearrange(board_state_BLC, "b l c -> (b l) c").cpu().numpy()

        for layer in layers:
            decision_tree = decision_trees[layer][custom_function.__name__]["decision_tree"][
                "model"
            ]
            simulated_activations_BF = decision_tree.predict(X)
            simulated_activations_BF = torch.tensor(
                simulated_activations_BF, device=device, dtype=torch.float32
            )
            simulated_activations_BLF = einops.rearrange(
                simulated_activations_BF, "(b l) f -> b l f", b=B, l=L
            )
            simulated_activations[layer] = simulated_activations_BLF

    # Get clean logits and mean submodule activations
    with torch.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        for layer in layers:
            submodule = submodule_dict[layer]
            ae = ae_dict[layer]
            if input_location != "transcoder":
                original_input_BLD = submodule.output
            else:
                original_input_BLD = submodule.input[0]
                if type(submodule.input.shape) == tuple:
                    original_input_BLD = original_input_BLD[0]

            encoded_BLF = ae.encode(original_input_BLD)

            mean_activations[layer] = encoded_BLF.mean(dim=(0, 1)).save()

            # TODO: There is a bug here with nnsight and proxy value not set.
            if ablation_method == "max":
                max_activations = encoded_BLF.max(dim=0).values
                mean_activations[layer] = max_activations.max(dim=0).values.save()

        logits_clean_BLV = model.unembed.output.save()

    # Get patch logits
    with torch.no_grad(), model.trace(game_batch_BL, **tracer_kwargs):
        for layer in layers:

            submodule = submodule_dict[layer]
            ae = ae_dict[layer]
            original_output_BLD = submodule.output

            if input_location != "transcoder":
                encoded_BLF = ae.encode(original_output_BLD)
            else:
                original_input_BLD = submodule.input[0]
                if type(submodule.input.shape) == tuple:
                    original_input_BLD = original_input_BLD[0]
                encoded_BLF = ae.encode(original_input_BLD)

            feat_idxs = selected_features[layer]

            decoded_BLD = ae.decode(encoded_BLF)
            error_BLD = original_output_BLD - decoded_BLD

            if ablation_method == "mean":
                encoded_BLF[:, :, feat_idxs] = mean_activations[layer][feat_idxs]
            elif ablation_method == "max":
                encoded_BLF[:, :, feat_idxs] = max_activations[layer][feat_idxs]
            elif ablation_method == "dt":
                encoded_BLF[:, :, feat_idxs] = simulated_activations[layer][:, :, feat_idxs]
                if ablate_not_selected:
                    not_feature_idxs_F = ~feat_idxs
                    encoded_BLF[:, :, not_feature_idxs_F] = mean_activations[layer][
                        not_feature_idxs_F
                    ]
            else:
                encoded_BLF[:, :, feat_idxs] = 0

            modified_decoded_BLD = ae.decode(encoded_BLF)

            submodule.output = modified_decoded_BLD
            if add_error:
                submodule.output += error_BLD

        logits_patch_BLV = model.unembed.output.save()

    return logits_clean_BLV, logits_patch_BLV


def compute_predictors(
    custom_functions: list[Callable],
    num_cores: int,
    layers: list[int],
    data: dict,
    neuron_acts: dict,
    binary_acts: dict,
    input_location: str,
    dataset_size: int,
    force_recompute: bool = False,
    save_results: bool = True,
    max_depth: int = 8,
) -> dict:

    output_filename = f"decision_trees/decision_trees_{input_location}_{dataset_size}.pkl"

    if not force_recompute and os.path.exists(output_filename):
        print(f"Loading decision trees from {output_filename}")
        with open(output_filename, "rb") as f:
            decision_trees = pickle.load(f)
        return decision_trees

    # Use all available cores, but max out at num_cores
    num_cores = min(num_cores, multiprocessing.cpu_count())

    results = {}

    for layer in layers:
        results[layer] = {}

    for custom_function in custom_functions:

        print(f"\n{custom_function.__name__}")
        games_BLC = data[custom_function.__name__]
        games_BLC = utils.to_device(games_BLC, "cpu")

        layer_results = Parallel(n_jobs=num_cores)(
            delayed(process_layer)(layer, games_BLC, neuron_acts, binary_acts, max_depth=max_depth)
            for layer in layers
        )

        for layer_result in layer_results:
            if layer_result is not None:
                layer = layer_result["layer"]
                results[layer][custom_function.__name__] = {
                    "decision_tree": layer_result["regular_dt"],
                    "binary_decision_tree": layer_result["binary_dt"],
                }

        # with ProcessPoolExecutor(max_workers=num_cores) as executor:
        #     future_to_layer = {executor.submit(process_layer, layer, games_BLC, neuron_acts, binary_acts): layer for layer in layers}
        #     for future in concurrent.futures.as_completed(future_to_layer):
        #         layer_result = future.result()
        #         layer = layer_result['layer']
        #         results[layer][custom_function.__name__] = {
        #             'decision_tree': layer_result['regular_dt'],
        #             'binary_decision_tree': layer_result['binary_dt']
        #         }

    if save_results:
        with open(output_filename, "wb") as f:
            pickle.dump(results, f)
    return results


def perform_interventions(
    decision_trees: dict,
    input_location: str,
    ablation_method: str,
    ablate_not_selected: bool,
    add_error: bool,
    custom_function: Callable,
    model,
    intervention_layers: list[list[int]],
    data: dict,
    threshold: float,
    ae_dict: dict,
    submodule_dict: dict,
):

    ablations = {}

    if input_location == "mlp_neuron":
        d_model = 2048
    elif (
        input_location == "sae_feature"
        or input_location == "sae_mlp_out_feature"
        or input_location == "transcoder"
    ):
        d_model = 4096
    else:
        d_model = 512

    effect_per_layer = []
    for selected_layers in intervention_layers:

        selected_features = {}

        for layer in selected_layers:
            if ablation_method == "zero" or ablation_method == "mean" or ablation_method == "max":
                selected_features[layer] = torch.ones(d_model, dtype=torch.bool)
            elif ablation_method == "dt":
                all_f1s = decision_trees[layer][custom_function.__name__]["decision_tree"]["r2"]
                good_f1s = all_f1s > threshold
                selected_features[layer] = good_f1s
                print(good_f1s.shape, good_f1s.dtype, good_f1s.sum())
            else:
                raise ValueError(f"Invalid ablation method: {ablation_method}")

        logits_clean_BLV, logits_patch_BLV = interventions(
            model=model,
            train_data=data,
            selected_features=selected_features,
            decision_trees=decision_trees,
            ae_dict=ae_dict,
            submodule_dict=submodule_dict,
            custom_function=custom_function,
            layers=selected_layers,
            ablation_method=ablation_method,
            ablate_not_selected=ablate_not_selected,
            add_error=add_error,
            input_location=input_location,
        )

        kl_div_BL = compute_kl_divergence(logits_clean_BLV, logits_patch_BLV)
        print(kl_div_BL.mean())
        effect_per_layer.append(kl_div_BL.mean().cpu())

    ablations["ablations"] = effect_per_layer
    ablations["hyperparameters"] = {
        "input_location": input_location,
        "ablation_method": ablation_method,
        "ablate_not_selected": ablate_not_selected,
        "add_error": add_error,
        "threshold": threshold,
        "custom_function": custom_function.__name__,
        "intervention_layers": intervention_layers,
    }

    return ablations


repo_dir = "/home/adam/OthelloUnderstanding"
model_name = "Baidicoot/Othello-GPT-Transformer-Lens"
batch_size = 10
n_batches = 2
dataset_size = batch_size * n_batches
layers = list(range(8))
threshold = 0.7
max_depth = 8

intervention_layers = []

for i in range(8):
    intervention_layers.append([i])


add_output_folders()

custom_functions = [
    othello_utils.games_batch_to_input_tokens_flipped_bs_valid_moves_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_classifier_input_BLC,
    # othello_utils.games_batch_to_board_state_and_input_tokens_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_flipped_classifier_input_BLC,
    # othello_utils.games_batch_to_board_state_classifier_input_BLC,
    # othello_utils.games_batch_to_input_tokens_parity_classifier_input_BLC,
]

true_false_combinations = list(itertools.product([True, False], repeat=2))

combinations = [
    # ("sae_mlp_out_feature", "max"),
    # ("sae_mlp_out_feature", "dt"),
    # ("transcoder", "dt"),
    # ("transcoder", "mean"),
    # ("mlp_neuron", "mean"),
    ("sae_mlp_out_feature", "mean"),
    # ("mlp_neuron", "dt"),
]

model, train_data = load_model_and_data(model_name, dataset_size, custom_functions)
test_data = construct_othello_dataset(
    custom_functions=custom_functions,
    n_inputs=dataset_size,
    split="test",
    device=device,
)

for combination in combinations:

    input_location, ablation_method = combination

    if input_location == "mlp_neuron" or ablation_method == "mean" or ablation_method == "max":
        trainer_ids = [None]
    else:
        trainer_ids = list(range(21))

    for trainer_id in trainer_ids:

        ae_dict = utils.get_aes(node_type=input_location, repo_dir=repo_dir, trainer_id=trainer_id)
        submodule_dict = get_submodule_dict(model, model_name, layers, input_location)

        neuron_acts = cache_sae_activations(
            model,
            model_name,
            train_data,
            layers,
            batch_size,
            n_batches,
            repo_dir,
            input_location,
            ae_dict,
            submodule_dict,
        )

        binary_acts = calculate_binary_activations(neuron_acts)

        neuron_acts = utils.to_device(neuron_acts, "cpu")
        binary_acts = utils.to_device(binary_acts, "cpu")

        results = {
            "hyperparameters": {
                "input_location": input_location,
                "trainer_id": trainer_id,
                "dataset_size": dataset_size,
            }
        }

        decision_trees = compute_predictors(
            custom_functions=custom_functions,
            num_cores=8,
            layers=layers,
            data=train_data,
            neuron_acts=neuron_acts,
            binary_acts=binary_acts,
            input_location=input_location,
            dataset_size=dataset_size,
            force_recompute=False,
            save_results=False,
            max_depth=max_depth,
        )

        for layer in decision_trees:
            results[layer] = {}
            for custom_function in custom_functions:
                results[layer][custom_function.__name__] = {
                    "decision_tree": {
                        "mse": decision_trees[layer][custom_function.__name__]["decision_tree"][
                            "mse"
                        ],
                        "r2": decision_trees[layer][custom_function.__name__]["decision_tree"][
                            "r2"
                        ],
                    },
                    "binary_decision_tree": {
                        "f1": decision_trees[layer][custom_function.__name__][
                            "binary_decision_tree"
                        ]["f1"],
                        "accuracy": decision_trees[layer][custom_function.__name__][
                            "binary_decision_tree"
                        ]["accuracy"],
                    },
                }

        with open(
            f"decision_trees/results_{input_location}_trainer_{trainer_id}_inputs_{dataset_size}.pkl",
            "wb",
        ) as f:
            pickle.dump(results, f)

        for combo in true_false_combinations:

            ablate_not_selected, add_error = combo

            if (ablation_method == "mean" or ablation_method == "max") and (
                ablate_not_selected != False or add_error != False
            ):
                continue

            if input_location == "mlp_neuron" and ablation_method != "mean" and add_error == False:
                continue

            ablation_custom_function = custom_functions[0]

            ablations = perform_interventions(
                decision_trees=decision_trees,
                input_location=input_location,
                ablation_method=ablation_method,
                ablate_not_selected=ablate_not_selected,
                add_error=add_error,
                custom_function=ablation_custom_function,
                model=model,
                intervention_layers=intervention_layers,
                data=test_data,
                threshold=threshold,
                ae_dict=ae_dict,
                submodule_dict=submodule_dict,
            )

            ablations["hyperparameters"]["trainer_id"] = trainer_id
            ablations["hyperparameters"]["dataset_size"] = dataset_size

            with open(
                f"decision_trees/ablation_results_{input_location}_{ablation_method}_ablate_not_selected_{ablate_not_selected}_add_error_{add_error}_trainer_{trainer_id}_inputs_{dataset_size}.pkl",
                "wb",
            ) as f:
                pickle.dump(ablations, f)
