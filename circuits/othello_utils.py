import torch as t
from tqdm import tqdm
from datasets import load_dataset

from circuits.othello_engine_utils import OthelloBoardState

DEFAULT_DTYPE = t.int16


def hf_othello_dataset_to_generator(
    dataset_name="taufeeque/othellogpt", split="train", streaming=True, token_mapping=None
):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            tokens = x["tokens"]
            if token_mapping is not None:
                tokens = [token_mapping[token] for token in tokens]
            yield tokens

    return gen()


def board_state_to_RRC(board_state, flip: int = 1) -> t.Tensor:
    board_state = t.tensor(board_state, dtype=DEFAULT_DTYPE)
    board_state *= flip
    one_hot = t.zeros((8, 8, 3), dtype=DEFAULT_DTYPE)
    one_hot[..., 0] = (board_state == -1).int()
    one_hot[..., 1] = (board_state == 0).int()
    one_hot[..., 2] = (board_state == 1).int()
    return one_hot


# TODO Remove duplicated logic from these functions
def games_batch_to_state_stack_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for move in game:
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_valid_moves_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack of valid moves"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            moves_board = t.zeros(8, 8, 1, dtype=DEFAULT_DTYPE)
            board.umpire(move)
            valid_moves_list = board.get_valid_moves()
            for move in valid_moves_list:
                moves_board[move // 8, move % 8] = 1
            states.append(moves_board)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_mine_yours_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state, flip)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_mine_yours_blank_mask_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Sequences of moves (dataset format) to state stack (one-hot) of shape (seq_len, 8, 8, 3)"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_RRC(board.state, flip)
            one_hot[..., 1] = 0
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def board_state_to_lines_RRC(board_state_RR, flip: int) -> t.Tensor:
    board_state_RR = t.tensor(board_state_RR, dtype=DEFAULT_DTYPE)
    board_state_RR *= flip  # Flip the board to standardize the player's perspective

    lines_board_RRC = t.zeros(8, 8, 8, dtype=DEFAULT_DTYPE)

    # Directions for movement in the format [dx, dy]
    eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    for r in range(8):
        for c in range(8):
            if board_state_RR[r, c] != 0:
                continue

            # Check each direction from 'eights'
            for direction_idx, (dx, dy) in enumerate(eights):
                x, y = r + dx, c + dy
                found_opponent = False
                while 0 <= x < 8 and 0 <= y < 8 and board_state_RR[x, y] == 1:
                    found_opponent = True
                    x += dx
                    y += dy

                # Check if the line ends with the player's piece (-1)
                if 0 <= x < 8 and 0 <= y < 8 and board_state_RR[x, y] == -1 and found_opponent:
                    lines_board_RRC[r, c, direction_idx] = 1

    return lines_board_RRC


def board_state_to_length_lines_RRC(board_state_RR, flip: int) -> t.Tensor:
    board_state_RR = t.tensor(board_state_RR, dtype=DEFAULT_DTYPE)
    board_state_RR *= flip  # Flip the board to standardize the player's perspective

    max_length = 6
    n_directions = 8

    lines_board_RRC = t.zeros(8, 8, (max_length * 8), dtype=DEFAULT_DTYPE)

    # Directions for movement in the format [dx, dy]
    eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    for r in range(8):
        for c in range(8):
            if board_state_RR[r, c] != 0:
                continue

            # Check each direction from 'eights'
            for direction_idx, (dx, dy) in enumerate(eights):

                length = 0

                x, y = r + dx, c + dy
                found_opponent = False
                while 0 <= x < 8 and 0 <= y < 8 and board_state_RR[x, y] == 1:
                    found_opponent = True
                    length += 1
                    x += dx
                    y += dy

                # Check if the line ends with the player's piece (-1)
                if 0 <= x < 8 and 0 <= y < 8 and board_state_RR[x, y] == -1 and found_opponent:
                    length = min(length, max_length) - 1
                    line_index = length * n_directions + direction_idx
                    lines_board_RRC[r, c, line_index] = 1

    return lines_board_RRC


def board_state_to_opponent_length_lines_RRC(board_state_RR, flip: int) -> t.Tensor:
    """In this case, we don't check that the end of the line ends with a `mine` piece."""
    board_state_RR = t.tensor(board_state_RR, dtype=DEFAULT_DTYPE)
    board_state_RR *= flip  # Flip the board to standardize the player's perspective

    max_length = 6
    n_directions = 8

    lines_board_RRC = t.zeros(8, 8, (max_length * 8), dtype=DEFAULT_DTYPE)

    # Directions for movement in the format [dx, dy]
    eights = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

    for r in range(8):
        for c in range(8):
            if board_state_RR[r, c] != 0:
                continue

            # Check each direction from 'eights'
            for direction_idx, (dx, dy) in enumerate(eights):
                length = 0

                x, y = r + dx, c + dy
                while 0 <= x < 8 and 0 <= y < 8 and board_state_RR[x, y] == 1:
                    x += dx
                    y += dy
                    length += 1

                length = min(length, max_length) - 1
                line_index = length * n_directions + direction_idx
                lines_board_RRC[r, c, line_index] = 1

    return lines_board_RRC


def games_batch_to_state_stack_lines_mine_BLRCC(batch_str_moves: list[list[int]]) -> t.Tensor:

    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_lines_RRC(board.state, flip)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def board_to_occupied_64(board_state) -> t.Tensor:
    board_state_RR = t.tensor(board_state, dtype=DEFAULT_DTYPE)
    occupied_RR = (board_state_RR != 0).int()
    return occupied_RR.flatten()


def games_batch_to_input_tokens_classifier_input_BLC(batch_str_moves: list[list[int]]) -> t.Tensor:
    """Shape batch, seq len, classes, where classes = (64 + 64 + 60 + 5)
    The first 64 is one hot, indicates which square the player just moved to
    The second 64 indicates which squares are occupied
    The last 60 is one hot and indicates the time position of the most recent move.
    The last 5 are ints indicating the row, col, time position, is_black, is_white"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            state = t.zeros(64 + 64 + 60 + 5, dtype=DEFAULT_DTYPE)
            board.umpire(move)

            if move >= 0:
                if move > 63:
                    raise ValueError(f"Move {move} is out of bounds")
                state[move] = 1
            occupied_64 = board_to_occupied_64(board.state)
            state[64:128] = occupied_64
            move_pos = 128 + i
            state[move_pos] = 1
            states.append(state)

            offset = 128 + 60
            row = i // 8
            col = i % 8
            state[offset + 0] = row
            state[offset + 1] = col
            state[offset + 2] = i
            state[offset + 3] = i % 2 == 1
            state[offset + 4] = i % 2 == 0
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_input_tokens_flipped_classifier_input_BLC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:
    """Shape batch, seq len, classes, where classes = (64 + 64 + 64 + 60)
    The first 64 is one hot, indicates which square the player just moved to
    The second 64 indicates which squares are occupied
    The third 64 indicates which squares have been flipped
    The last 60 is one hot and indicates the time position of the most recent move"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        prev_board_RRC = board_state_to_RRC(board.state, flip=1)
        for i, move in enumerate(game):
            state = t.zeros(64 + 64 + 64 + 60, dtype=DEFAULT_DTYPE)
            board.umpire(move)

            if move >= 0:
                if move > 63:
                    raise ValueError(f"Move {move} is out of bounds")
                state[move] = 1

            occupied_64 = board_to_occupied_64(board.state)
            state[64:128] = occupied_64

            cur_board_RRC = board_state_to_RRC(board.state, flip=1)

            prev_board_RRC[..., 1] = 0
            cur_board_RRC[..., 1] = 0
            diff_board_RRC = cur_board_RRC - prev_board_RRC

            # This finds all squares that have been flipped
            diff_board_RR = (diff_board_RRC[:, :, 0] * diff_board_RRC[:, :, 2] == -1).float()

            state[128:192] = diff_board_RR.flatten()

            prev_board_RRC = cur_board_RRC

            move_pos = 128 + 64 + i
            state[move_pos] = 1
            states.append(state)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_input_tokens_parity_classifier_input_BLC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:
    """Experimental function, indicates which squares were played by either player. Seems to have poor results."""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []

        parity_state = t.zeros(128, dtype=DEFAULT_DTYPE)

        for i, move in enumerate(game):
            state = t.zeros(64 + 64 + 128 + 60, dtype=DEFAULT_DTYPE)
            board.umpire(move)

            flip = 1
            if i % 2 == 1:
                flip = -1

            if move >= 0:
                if move > 63:
                    raise ValueError(f"Move {move} is out of bounds")
                state[move] = 1

                if flip == 1:
                    offset = 0
                else:
                    offset = 64

                parity_state[move + offset] = 1

            occupied_64 = board_to_occupied_64(board.state)
            state[64:128] = occupied_64

            state[128:256] = parity_state

            move_pos = 256 + i
            state[move_pos] = 1
            states.append(state)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_board_state_and_input_tokens_classifier_input_BLC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:
    """Shape batch, seq len, classes, where classes = (64 + 64 + 128 + 60)
    The first 64 is one hot, indicates which square the player just moved to
    The second 64 indicates which squares are occupied
    The next 128 is the board state for mine and yours
    The last 60 is one hot and indicates the time position of the most recent move"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):

            flip = 1
            if i % 2 == 1:
                flip = -1

            state = t.zeros(64 + 64 + 128 + 60, dtype=DEFAULT_DTYPE)
            board.umpire(move)

            if move >= 0:
                if move > 63:
                    raise ValueError(f"Move {move} is out of bounds")
                state[move] = 1
            occupied_64 = board_to_occupied_64(board.state)

            state[64:128] = occupied_64

            board_RRC = board_state_to_RRC(board.state, flip)

            state[128:192] = board_RRC[..., 0].flatten()
            state[192:256] = board_RRC[..., 2].flatten()

            move_pos = 256 + i
            state[move_pos] = 1
            states.append(state)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_input_tokens_flipped_bs_classifier_input_BLC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:
    """Shape batch, seq len, classes, where classes = (64 + 64 + 64 + 192 + 60 + 5)
    The first 64 is one hot, indicates which square the player just moved to
    The second 64 indicates which squares have been flipped
    The next 192 is the board state for mine and yours and blank
    The last 5 are ints indicating the row, col, time position, is_black, is_white"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        prev_board_RRC = board_state_to_RRC(board.state, flip=1)
        for i, move in enumerate(game):

            flip = 1
            if i % 2 == 1:
                flip = -1

            state = t.zeros(64 + 64 + 192 + 5, dtype=DEFAULT_DTYPE)
            board.umpire(move)

            if move >= 0:
                if move > 63:
                    raise ValueError(f"Move {move} is out of bounds")
                state[move] = 1

            cur_board_RRC = board_state_to_RRC(board.state, flip=1)

            diff_board_RRC = cur_board_RRC - prev_board_RRC

            # This finds all squares that have been flipped
            diff_board_RR = (diff_board_RRC[:, :, 0] * diff_board_RRC[:, :, 2] == -1).float()

            state[64:128] = diff_board_RR.flatten()

            prev_board_RRC = cur_board_RRC

            board_state_RRC = board_state_to_RRC(board.state, flip=flip)
            state[128:320] = board_state_RRC.flatten()

            offset = 128 + 192
            row = i // 8
            col = i % 8
            state[offset + 0] = row
            state[offset + 1] = col
            state[offset + 2] = i
            state[offset + 3] = i % 2 == 1
            state[offset + 4] = i % 2 == 0

            states.append(state)

        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_input_tokens_flipped_bs_valid_moves_classifier_input_BLC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:
    """Shape batch, seq len, classes, where classes = (64 + 64 + 64 + 192 + 60 + 5)
    The first 64 is one hot, indicates which square the player just moved to
    The second 64 indicates which squares have been flipped
    The next 192 is the board state for mine and yours and blank
    The next 64 is the valid moves
    The last 5 are ints indicating the row, col, time position, is_black, is_white"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        prev_board_RRC = board_state_to_RRC(board.state, flip=1)
        for i, move in enumerate(game):

            flip = 1
            if i % 2 == 1:
                flip = -1

            state = t.zeros(64 + 64 + 64 + 192 + 5, dtype=DEFAULT_DTYPE)
            board.umpire(move)

            if move >= 0:
                if move > 63:
                    raise ValueError(f"Move {move} is out of bounds")
                state[move] = 1

            cur_board_RRC = board_state_to_RRC(board.state, flip=1)

            diff_board_RRC = cur_board_RRC - prev_board_RRC

            # This finds all squares that have been flipped
            diff_board_RR = (diff_board_RRC[:, :, 0] * diff_board_RRC[:, :, 2] == -1).float()

            state[64:128] = diff_board_RR.flatten()

            prev_board_RRC = cur_board_RRC

            board_state_RRC = board_state_to_RRC(board.state, flip=flip)
            state[128:320] = board_state_RRC.flatten()

            moves_board = t.zeros(8, 8, 1, dtype=DEFAULT_DTYPE)
            valid_moves_list = board.get_valid_moves()
            for move in valid_moves_list:
                moves_board[move // 8, move % 8] = 1

            state[320:384] = moves_board.flatten()

            offset = 128 + 192 + 64
            row = i // 8
            col = i % 8
            state[offset + 0] = row
            state[offset + 1] = col
            state[offset + 2] = i
            state[offset + 3] = i % 2 == 1
            state[offset + 4] = i % 2 == 0

            states.append(state)

        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_board_state_classifier_input_BLC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:
    """Shape batch, seq len, classes, where classes = (64 + 64 + 128 + 60)
    The first 64 is one hot, indicates which square the player just moved to
    The second 64 indicates which squares are occupied
    The next 128 is the board state for mine and yours
    The last 60 is one hot and indicates the time position of the most recent move"""
    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):

            flip = 1
            if i % 2 == 1:
                flip = -1

            state = t.zeros(64 + 192, dtype=DEFAULT_DTYPE)
            board.umpire(move)

            if move >= 0:
                if move > 63:
                    raise ValueError(f"Move {move} is out of bounds")
                state[move] = 1

            board_RRC = board_state_to_RRC(board.state, flip)

            state[64:] = board_RRC.flatten()
            states.append(state)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_length_lines_mine_BLRRC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:

    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_length_lines_RRC(board.state, flip)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_opponent_length_lines_mine_BLRRC(
    batch_str_moves: list[list[int]],
) -> t.Tensor:

    iterable = tqdm(batch_str_moves) if len(batch_str_moves) > 50 else batch_str_moves

    game_stack = []
    for game in iterable:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 1:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_opponent_length_lines_RRC(board.state, flip)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


def games_batch_to_state_stack_lines_yours_BLRRC(batch_str_moves: list[int]) -> t.Tensor:
    """Difference is in `if i % 2 == 0:` instead of `if i % 2 == 1:`
    This function seems to be not required as it scores very poorly on board reconstruction"""

    game_stack = []
    for game in batch_str_moves:
        if isinstance(game, t.Tensor):
            game = game.flatten()

        board = OthelloBoardState()
        states = []
        for i, move in enumerate(game):
            flip = 1
            if i % 2 == 0:
                flip = -1
            board.umpire(move)
            one_hot = board_state_to_lines_RRC(board.state, flip)
            states.append(one_hot)
        states = t.stack(states, axis=0)
        game_stack.append(states)
    return t.stack(game_stack, axis=0)


othello_functions = [
    games_batch_to_state_stack_BLRRC.__name__,
    games_batch_to_state_stack_mine_yours_BLRRC.__name__,
    games_batch_to_state_stack_mine_yours_blank_mask_BLRRC.__name__,
    games_batch_to_valid_moves_BLRRC.__name__,
    games_batch_to_state_stack_lines_mine_BLRCC.__name__,
    games_batch_to_state_stack_lines_yours_BLRRC.__name__,
]
