**Setup** 

Create a new virtual python environment (I'm using 3.11).

```

pip install -r requirements.txt
pip install -e .
git submodule update --init --recursive

# Some python scripts will download zipped SAEs off huggingface, this isn't needed by default
apt install zip
apt install unzip
```

Then run `python neuron_simulation/simulate_activations_with_dts.py`.

Claude generated explanation of repo:

# Othello-GPT Interpretable Neurons

This repository contains research work on identifying and utilizing interpretable neurons in Othello-GPT using depth-limited decision trees.

## Project Overview

We identify interpretable neurons in Othello-GPT by training shallow decision trees (depth-5) to approximate their behavior. The decision trees provide inherent interpretability while maintaining reasonable performance. We then use these interpretable approximations to replace parts of the original model and evaluate the impact.

## Data Processing

The project processes Othello game data into PyTorch tensors with custom functions that capture various game aspects:
- Squares flipped by the most recent move
- History of moves played
- Current board state (white and black pieces)

These tensors serve as inputs for training the decision trees. We have various custom functions that can be used:

`games_batch_to_board_state_classifier_input_BLC`: Which squares are mine, yours, or blank.
`games_batch_to_input_tokens_flipped_bs_classifier_input_BLC`: Which moves have been played, which squares were flipped by the most recent move, and the board state.
Many others.

## Training Process

The training process is relatively lightweight:
- Typically requires only ~60 games for good decision tree performance
- Training completes in a few minutes
- Decision trees are trained in parallel across all 8 layers
- Training happens on CPU

### Resource Requirements
- Minimum 8 CPU cores (for parallel training across layers)
- Peak GPU memory usage is approximately 2GB
- Can run entirely on CPU (including model inference)

## Key Features

### Neuron Selection
- We identify neurons that can be explained by decision trees above an R-squared threshold (e.g., 0.7)
- Some layers show high interpretability, with 30-40% of MLP neurons being well-explained

### Model Modification Flags
- `ablate_not_selected`: (Recommended: True) Ablates unexplained neurons while replacing explained ones with decision trees
- `add_error`: Applies to SAEs, determines whether to include the error term

### Performance Metrics
We evaluate modified models using:
- KL divergence from original model
- Legal move prediction accuracy (original model achieves 99.98% accuracy)
- Comparison against mean ablation baseline

## Current Results

The modified model currently:
- Outperforms mean ablation baseline
- Shows lower performance than the original model
- Successfully maintains some of the original model's capabilities while using interpretable components

## Implementation Example

For example, in a model with 2,000 MLP neurons where 1,000 are well-explained by decision trees:
1. The well-explained neurons are replaced with their decision tree approximations
2. The remaining 1,000 unexplained neurons are ablated (when `ablate_not_selected=True`)
3. Performance metrics are calculated to evaluate the impact

**Shape Annotations**

I've been using this tip from Noam Shazeer:

Dimension key (from https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd):

f = All SAE features

F = Batch of SAE features

b = All inputs

B = batch_size

L = seq length (context length)

T = thresholds

R = rows (or cols)

C = classes for one hot encoding

D = GPT d_model

For example, boards begins as shape (Batch, seq_len, rows, rows, classes), and after einops.repeat is shape (num_thresholds, num_features, batch_size, seq_len, rows, rows, classes).


```
boards_BLRRC = batch_data[custom_function.__name__]
boards_TFBLRRC = einops.repeat(
    boards_BLRRC,
    "B L R1 R2 C -> T F B L R1 R2 C",
    F=f_batch_size,
    T=len(thresholds_T111),
)
```

**Tests**

Run `pytest -s` from the root directory to run all tests. This will take a couple minutes, and `-s` is helpful to gauge progress.