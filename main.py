import pygame
import chess
import chess.svg
import torch
import io
import cairosvg
from PIL import Image
from SimpleChessNet import SimpleChessNet


def board_to_tensor(board, move):
    piece_map = board.piece_map()
    tensor = torch.zeros(773)
    piece_to_idx = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    for square, piece in piece_map.items():
        idx = 64 * piece_to_idx[piece.symbol()] + square
        tensor[idx] = 1.0

    tensor[768] = move.from_square / 63.0
    tensor[769] = move.to_square / 63.0
    tensor[770] = 1.0 if move.promotion == chess.QUEEN else 0.0
    tensor[771] = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[772] = board.fullmove_number / 100.0

    return tensor


def choose_best_move(board, model):
    legal_moves = list(board.legal_moves)
    best_score = float('-inf')
    best_move = None

    for move in legal_moves:
        board.push(move)
        tensor = board_to_tensor(board, move)
        board.pop()

        with torch.no_grad():
            score = model(tensor.unsqueeze(0)).item()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def board_to_image(board, width):
    svg_data = chess.svg.board(board=board, size=width)
    png_data = cairosvg.svg2png(bytestring=svg_data)
    image = Image.open(io.BytesIO(png_data)).convert("RGB")
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)


if __name__ == "__main__":
    # Dwa niezależne modele (możesz je potem wczytać z różnych plików)
    white_net = SimpleChessNet()
    black_net = SimpleChessNet()

    WIDTH = 480
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, WIDTH))
    pygame.display.set_caption("Szachy: AI vs AI (oddzielne sieci)")
    clock = pygame.time.Clock()

    board = chess.Board()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if board.is_game_over():
            print("Koniec gry:", board.result(), board.outcome().termination.name)
            pygame.time.wait(5000)
            break

        if board.turn == chess.WHITE:
            move = choose_best_move(board, white_net)
        else:
            move = choose_best_move(board, black_net)

        if move is None:
            print("Brak dostępnych ruchów.")
            break
        print(f"{'Białe' if board.turn == chess.WHITE else 'Czarne'} grają: {move}")
        board.push(move)

        img = board_to_image(board, WIDTH)
        screen.blit(img, (0, 0))
        pygame.display.flip()

        clock.tick(1)

    pygame.quit()
