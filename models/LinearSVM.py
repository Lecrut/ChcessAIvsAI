import numpy as np
from scipy.optimize import minimize
from board import board_to_feature_vector


class LinearSVM:
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.bias = None
        self.w = None
        self.support_vectors = None
        self.support_labels = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = np.where(y == 0, -1, 1).astype(float)

        kernel_matrix = X @ X.T

        def objective(alpha):
            return -np.sum(alpha) + 0.5 * np.sum(
                (alpha * y)[:, None] * (alpha * y)[None, :] * kernel_matrix
            )

        def equality_constraint(alpha):
            return np.dot(alpha, y)

        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {'type': 'eq', 'fun': equality_constraint}
        initial_alphas = np.random.rand(n_samples)

        result = minimize(
            objective,
            initial_alphas,
            bounds=bounds,
            constraints=constraints,
            tol=self.tol,
            method='SLSQP',
            options={'maxiter': self.max_iter, 'disp': False}
        )

        if not result.success:
            print("Warning: Optimization did not succeed:", result.message)

        alphas = result.x
        is_support_vector = alphas > 1e-5
        self.alphas = alphas[is_support_vector]
        self.support_vectors = X[is_support_vector]
        self.support_labels = y[is_support_vector]

        self.w = np.sum((self.alphas * self.support_labels)[:, None] * self.support_vectors, axis=0)

        self.bias = np.mean([
            y_k - np.dot(self.w, x_k)
            for x_k, y_k in zip(self.support_vectors, self.support_labels)
        ])

    def _decision_function(self, X):
        return X @ self.w + self.bias

    def predict(self, X):
        return np.where(self._decision_function(X) >= 0, 1, 0)


def choose_best_move_svm(board, model):
    legal_moves = list(board.legal_moves)
    best_score = float('-inf')
    best_move = None

    for move in legal_moves:
        board.push(move)
        features = board_to_feature_vector(board)
        score = model._decision_function(features.reshape(1, -1))[0]
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move
