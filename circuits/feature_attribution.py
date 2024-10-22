import sys
import torch
import numpy as np
import pandas as pd
import pathlib
import itertools
from sklearn.metrics import roc_auc_score
from datetime import datetime

from circuits import utils, othello_utils
from circuits.eval_sae_as_classifier import construct_othello_dataset


def _patch_attribution_py(lib_path):
    # this is a hack to make the attribution code work with othello model
    with open(lib_path / "attribution.py", "r") as f:
        contents = f.read()
    contents = contents.replace(
        'with model.trace("_"):', "with model.trace(t.tensor([0])):"
    )
    with open(lib_path / "attribution.py", "w") as f:
        f.write(contents)


lib_path = pathlib.Path(__file__).parent / "feature_circuits"
_patch_attribution_py(lib_path)
sys.path.append(str(lib_path))

from attribution import patching_effect


def get_selected_aes(model, target_layer=5, trainer_id=5):
    aes = utils.get_aes(
        node_type="sae_feature_topk",
        trainer_id=trainer_id,
        repo_dir=".",
    )
    selected_aes = aes[:target_layer]
    dictionary, submodules = {}, []
    for i, ae in enumerate(selected_aes):
        submodules.append(model.blocks[i].hook_resid_post)
        dictionary[model.blocks[i].hook_resid_post] = ae
    return dictionary, submodules


@torch.no_grad()
def process_data(model, inpts):
    bs = 64
    all_outs, all_acts = [], []
    for i in range(0, len(inpts), bs):
        batch = inpts[i : i + bs]
        outs = model._model(batch)
        all_outs.append(outs)
        with model.trace(batch):
            act = model.blocks[5].hook_resid_post.output.save()
            all_acts.append(act)
    all_data = torch.cat(all_outs, dim=0)
    all_acts_BLD = torch.cat(all_acts, dim=0)
    return all_data, all_acts_BLD


def compute_roc_scores(probe_outs, y):
    roc_scores = np.zeros((8, 8, 3))
    for i in range(8):
        for j in range(8):
            for k in range(3):
                y_ij = y[:, :, i, j, k].flatten().cpu().numpy()
                x_ij = probe_outs[:, :, i, j, k].flatten().cpu().numpy()
                try:
                    roc_scores[i, j, k] = roc_auc_score(y_ij, x_ij)
                except ValueError:
                    roc_scores[i, j, k] = 1
    return roc_scores


def create_metric_fn(probe, layer):
    def metric_fn(model, labels=None):
        acts = model.blocks[layer].hook_resid_post.output
        probe_preds = acts @ probe
        return torch.where(labels == 1, probe_preds, -probe_preds)

    return metric_fn


def compute_attribution(model, inpts, labels_BL, submodules, dictionary, metric_fn):
    running_total = 0
    running_nodes = None

    for i in range(0, len(inpts), 64):
        batch = inpts[i : i + 64]
        effects, _, _, _ = patching_effect(
            batch,
            None,
            model,
            submodules,
            dictionary,
            metric_fn,
            metric_kwargs=dict(labels=labels_BL[i : i + 64].cuda()),
            method="attrib",
        )

        with torch.no_grad():
            if running_nodes is None:
                running_nodes = {
                    k: len(batch) * v.sum(dim=1).mean(dim=0) for k, v in effects.items()
                }
            else:
                for k, v in effects.items():
                    running_nodes[k] += len(batch) * v.sum(dim=1).mean(dim=0)
            running_total += len(batch)

    nodes = {k: v / running_total for k, v in running_nodes.items()}
    nodes = {k: v.act for k, v in nodes.items()}
    return nodes


def create_attribution_df(nodes, threshold=1e-3):
    results = []
    for hook_point, effects in nodes.items():
        layer = int(hook_point.name.split(".")[1])
        for i, effect in enumerate(effects.tolist()):
            if abs(effect) < threshold:
                continue
            results.append(
                {
                    "layer": layer,
                    "feat_idx": i,
                    "effect": effect,
                }
            )
    return pd.DataFrame(results)


def main(target_layer, probe_path, output_dir, device="cuda"):
    output_dir = pathlib.Path(output_dir)
    model = utils.get_model("mntss/Othello-GPT", device)
    probe = torch.load(probe_path)
    dictionary, submodules = get_selected_aes(model)

    train_data = construct_othello_dataset(
        [othello_utils.games_batch_to_state_stack_mine_yours_BLRRC],
        n_inputs=1024,
        split="train",
    )

    inpts = torch.tensor(train_data["encoded_inputs"]).long().cuda()
    board_data = train_data["games_batch_to_state_stack_mine_yours_BLRRC"]
    all_data, all_acts_BLD = process_data(model, inpts)

    probe_outs = torch.einsum("bld,dmno->blmno", all_acts_BLD, probe)

    roc_scores = compute_roc_scores(probe_outs, board_data)
    assert (roc_scores.mean(axis=(0, 1)) > 0.98).all()

    for x, y, mode in itertools.product(range(8), range(8), range(3)):
        print(f"Processing square {x}, {y}, mode {mode}")
        labels_BL = board_data[:, :, x, y, mode]
        metric_fn = create_metric_fn(probe[:, x, y, mode], target_layer)

        nodes = compute_attribution(
            model, inpts, labels_BL, submodules, dictionary, metric_fn
        )
        attribution_df = create_attribution_df(nodes)

        output_path = output_dir / f"layer_{target_layer}"
        output_path.mkdir(parents=True, exist_ok=True)
        attribution_df.to_pickle(
            str(output_path / f"attribution_df_{x}_{y}_{mode}.pkl")
        )


if __name__ == "__main__":
    from datetime import datetime
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    probe_path = "linear_probes/20241022_043515/probe_5.pt"

    output_dir = f"attribution_dfs_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    main(
        target_layer=5,
        probe_path=probe_path,
        output_dir=output_dir,
    )
