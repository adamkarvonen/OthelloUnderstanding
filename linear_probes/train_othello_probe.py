import os
import torch
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from circuits import utils
from circuits import othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset


def make_dataset(n_games):
    dataset = construct_othello_dataset(
        [othello_utils.games_batch_to_state_stack_mine_yours_BLRRC],
        n_inputs=n_games,
        split="train",
        device="cuda",
    )
    boards = (
        dataset["games_batch_to_state_stack_mine_yours_BLRRC"]
        .reshape(-1, 59, 64, 3)
        .permute(3, 0, 1, 2)
    )
    games = torch.tensor(dataset["encoded_inputs"])
    return games, boards


def get_activations(model, games, layer="all"):
    minibatch_size = 128
    if layer == "all":
        keys = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    else:
        keys = [f"blocks.{layer}.hook_resid_post"]

    all_acts = []
    for i in range(0, len(games), minibatch_size):
        minibatch = games[i : i + minibatch_size]
        logits, cache = model.run_with_cache(
            minibatch, names_filter=lambda name: name in keys
        )
        if layer == "all":
            batch_acts = torch.cat([cache[key] for key in keys], dim=-1)
        else:
            batch_acts = cache[keys[0]]
        all_acts.append(batch_acts)
    return torch.cat(all_acts, dim=0)


def train_least_squares(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    probe = torch.linalg.lstsq(X_train, y_train).solution

    y_pred = X_test @ probe
    y_mean = y_test.mean()
    ss_tot = torch.sum((y_test - y_mean) ** 2)
    ss_res = torch.sum((y_test - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    y_test = y_test.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    same_class = (y_test == y_test[0]).all(axis=0)
    roc_auc = roc_auc_score(y_test[:, ~same_class], y_pred[:, ~same_class])

    per_square_roc_auc = []
    for square in range(64):
        if same_class[square]:
            score = 1
        else:
            score = roc_auc_score(y_test[:, square], y_pred[:, square])
        per_square_roc_auc.append(score)

    return probe, r_squared.item(), roc_auc, per_square_roc_auc


def main(output_dir, n_games=1000, device='cuda', layers=None):
    if layers is None:
        layers = list(range(8)) + ["all"]

    model = utils.get_model("mntss/Othello-GPT", device)
    print(f"Model loaded. Creating dataset with {n_games} games...")
    games, boards = make_dataset(n_games)
    boards, games = boards.to(device), games.to(device)

    for layer in layers:
        print(f"Training for layer {layer}...")
        probe, results = train_probe(model, games, boards, layer=layer)
        print("Training completed. Saving results...")
        os.makedirs(output_dir, exist_ok=True)

        torch.save(probe, output_dir + f"probe_{layer}.pt")
        with open(output_dir + f"results_{layer}.json", "w") as f:
            json.dump(results, f)


def train_probe(model, games, boards, layer="all"):
    all_acts = get_activations(model, games, layer=layer)
    print("Activations obtained. Preparing data for training...")

    X = all_acts.reshape(-1, all_acts.shape[-1])
    results = []
    probes = []
    for mode in range(0, 3):
        print(f"Training for mode {mode}...")
        y = boards[mode].reshape(-1, 64).float()
        probe, r_squared, roc_auc, per_square_roc_auc = train_least_squares(X, y)
        print(f"R-squared score: {r_squared:.4f} for mode {mode}")
        print(f"ROC AUC score: {roc_auc:.4f} for mode {mode}")
        probes.append(probe)
        results.append(
            {
                "r_squared": r_squared,
                "roc_auc": roc_auc,
                "per_square_roc_auc": per_square_roc_auc,
                "layer": layer,
                "mode": mode,
            }
        )

    probe = torch.stack(probes).reshape(3, all_acts.shape[-1], 8, 8).permute(1, 2, 3, 0)
    return probe, results


if __name__ == "__main__":
    output_dir = datetime.now().strftime('%Y%m%d_%H%M%S') + "/"
    main(output_dir)
