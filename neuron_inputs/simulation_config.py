from dataclasses import dataclass
from typing import Optional

import circuits.othello_utils as othello_utils


@dataclass
class InterventionCombination:
    input_location: str
    ablation_method: str
    ablate_not_selected: list[bool]
    add_error: list[bool]
    trainer_ids: list[Optional[int]]


SAE_config = InterventionCombination(
    input_location="CHANGE_ME",
    ablation_method="dt",
    trainer_ids=list(range(21)),
    ablate_not_selected=[True, False],
    add_error=[True, False],
)

SAE_mean_config = InterventionCombination(
    input_location="CHANGE_ME",
    ablation_method="mean",
    trainer_ids=[None],
    ablate_not_selected=[True],
    add_error=[True],
)

SAE_mlp_out_feature_config = SAE_config
SAE_mlp_out_feature_config.input_location = "SAE_mlp_out_feature"

transcoder_config = SAE_config
transcoder_config.input_location = "transcoder"

MLP_dt_config = InterventionCombination(
    input_location="mlp_neuron",
    ablation_method="dt",
    trainer_ids=[None],
    ablate_not_selected=[True, False],
    add_error=[True],
)

MLP_mean_config = InterventionCombination(
    input_location="mlp_neuron",
    ablation_method="mean",
    trainer_ids=[None],
    ablate_not_selected=[True],
    add_error=[True],
)


@dataclass
class SimulationConfig:
    repo_dir: str = "/home/adam/OthelloUnderstanding"
    model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens"
    batch_size: int = 10
    n_batches: int = 5
    num_layers: int = 8
    layers = list(range(num_layers))
    intervention_threshold: float = 0.7
    max_depth: int = 8
    num_cores: int = 8
    force_recompute: bool = False
    save_decision_trees: bool = False
    binary_threshold: float = 0.1

    intervention_layers = []

    for i in range(num_layers):
        intervention_layers.append([i])

    custom_functions = [
        othello_utils.games_batch_to_input_tokens_flipped_bs_valid_moves_classifier_input_BLC,
        # othello_utils.games_batch_to_input_tokens_classifier_input_BLC,
        # othello_utils.games_batch_to_board_state_and_input_tokens_classifier_input_BLC,
        # othello_utils.games_batch_to_input_tokens_flipped_classifier_input_BLC,
        # othello_utils.games_batch_to_board_state_classifier_input_BLC,
        # othello_utils.games_batch_to_input_tokens_parity_classifier_input_BLC,
    ]

    combinations = [SAE_mlp_out_feature_config, transcoder_config, MLP_dt_config, MLP_mean_config]
