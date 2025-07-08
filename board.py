import chess
import numpy as np


def board_to_feature_vector(board):
    pieces = board.piece_map()
    features = []

    piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9,
                    'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9}

    material = sum(piece_values.get(p.symbol(), 0) for p in pieces.values())
    features.append(material / 39.0)

    features.append(1.0 if board.turn == chess.WHITE else 0.0)
    features.append(board.fullmove_number / 100.0)

    return np.array(features, dtype=np.float32)
