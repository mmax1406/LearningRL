import numpy as np

class GridCleanEnv:
    def __init__(self, grid: np.ndarray, start: tuple[int, int]):
        self.grid = grid.astype(np.float32)
        self.n_rows, self.n_cols = grid.shape
        self.start_r, self.start_c = start
        self.r, self.c = self.start_r, self.start_c
        self.state_dim = self.n_rows * self.n_cols

    def _encode_state(self, r: int, c: int) -> np.ndarray:
        g = self.grid.copy()
        g[r, c] = 2.0
        return (g / 2.0).reshape(-1).astype(np.float32)

    def reset(self):
        self.r, self.c = self.start_r, self.start_c
        return self._encode_state(self.r, self.c)

    def step(self, a: int):
        if a == 0: nr, nc = max(0, self.r - 1), self.c       # up
        elif a == 1: nr, nc = min(self.n_rows - 1, self.r + 1), self.c  # down
        elif a == 2: nr, nc = self.r, max(0, self.c - 1)     # left
        elif a == 3: nr, nc = self.r, min(self.n_cols - 1, self.c + 1) # right
        else: raise ValueError("Action must be 0..3")

        prev_s = self._encode_state(self.r, self.c)
        prev_g2 = prev_s.reshape(self.n_rows, self.n_cols) * 2.0

        reward, done = -1.0, False
        if self.grid[nr, nc] == 1.0:
            reward, done = -10.0, True
        elif prev_g2[nr, nc] == 0.0:
            reward = 5.0
        elif prev_s.reshape(self.n_rows, self.n_cols)[nr, nc] == 0.5:
            reward = -0.5

        self.r, self.c = nr, nc
        return self._encode_state(self.r, self.c), reward, done, {}
