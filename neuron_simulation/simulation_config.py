from dataclasses import dataclass, replace
from typing import Optional

import circuits.othello_utils as othello_utils


@dataclass
class InterventionCombination:
    input_location: str
    ablation_method: str
    ablate_not_selected: list[bool]
    add_error: list[bool]
    trainer_ids: list[int]


SAE_config = InterventionCombination(
    input_location="CHANGE_ME",
    ablation_method="dt",
    trainer_ids=list(range(21)),
    ablate_not_selected=[True, False],
    add_error=[True, False],
)

sae_mean_config = InterventionCombination(
    input_location="CHANGE_ME",
    ablation_method="mean",
    trainer_ids=[0],
    ablate_not_selected=[True],
    add_error=[True],
)


sae_mlp_out_feature_config = replace(SAE_config, input_location="sae_mlp_out_feature")
transcoder_config = replace(SAE_config, input_location="transcoder")

test_sae_mlp_out_feature_config = replace(sae_mlp_out_feature_config, trainer_ids=list(range(10)))
test_transcoder_config = replace(test_sae_mlp_out_feature_config, input_location="transcoder")

# The following are hand selected for good L0 / frac recovered tradeoff
selected_sae_mlp_out_feature_config = replace(sae_mlp_out_feature_config, trainer_ids=[8])
selected_transcoder_config = replace(transcoder_config, trainer_ids=[2])

MLP_dt_config = InterventionCombination(
    input_location="mlp_neuron",
    ablation_method="dt",
    trainer_ids=[0],
    ablate_not_selected=[True, False],
    add_error=[True],
)

MLP_mean_config = InterventionCombination(
    input_location="mlp_neuron",
    ablation_method="mean",
    trainer_ids=[0],
    ablate_not_selected=[True],
    add_error=[True],
)


@dataclass
class SimulationConfig:
    repo_dir: str = ""
    model_name: str = "Baidicoot/Othello-GPT-Transformer-Lens"
    output_location: str = "neuron_simulation/"
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
    previous_layers_as_input: bool = False
    binary_dt: bool = False

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

    combinations = [sae_mlp_out_feature_config, transcoder_config, MLP_dt_config, MLP_mean_config]


test_config = SimulationConfig()
test_config.combinations = [
    test_sae_mlp_out_feature_config,
    test_transcoder_config,
    MLP_dt_config,
    MLP_mean_config,
]

selected_config = SimulationConfig()
selected_config.combinations = [
    selected_sae_mlp_out_feature_config,
    selected_transcoder_config,
    MLP_dt_config,
    MLP_mean_config,
]
