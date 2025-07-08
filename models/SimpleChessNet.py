import torch.nn as nn
import torch
from board import board_to_feature_vector


class SimpleChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


def choose_best_move_net(board, model):
    model.eval()
    legal_moves = list(board.legal_moves)
    best_score = float('-inf')
    best_move = None

    for move in legal_moves:
        board.push(move)
        features = board_to_feature_vector(board)
        x = torch.tensor(features).unsqueeze(0)
        with torch.no_grad():
            score = model(x).item()
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move
