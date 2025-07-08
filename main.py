import pygame
import chess
import chess.svg
import cairosvg
import io
from PIL import Image
import numpy as np
from models.LinearSVM import LinearSVM, choose_best_move_svm
from models.SimpleChessNet import SimpleChessNet, choose_best_move_net


def board_to_image(board, width):
    svg_data = chess.svg.board(board=board, size=width)
    png_data = cairosvg.svg2png(bytestring=svg_data)
    image = Image.open(io.BytesIO(png_data)).convert("RGB")
    return pygame.image.fromstring(image.tobytes(), image.size, image.mode)


def train_dummy_svm():
    X = np.random.uniform(-1, 1, (100, 3))
    y = (X[:, 0] > 0).astype(int)

    model = LinearSVM()
    model.fit(X, y)
    return model


if __name__ == "__main__":
    white_svm = train_dummy_svm()
    black_net = SimpleChessNet()

    WIDTH = 480
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, WIDTH))
    pygame.display.set_caption("SVM Szachy: AI vs AI")
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
            move = choose_best_move_svm(board, white_svm)
        else:
            move = choose_best_move_net(board, black_net)

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
