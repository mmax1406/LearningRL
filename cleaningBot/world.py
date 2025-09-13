import numpy as np

class GridCleanEnv:
    def __init__(self, gridSize: int, numOfObstacles: int, ObstaclesMaxSize: int):
        self.numOfObstacles = numOfObstacles
        self.ObstaclesMaxSize = ObstaclesMaxSize
        self.n_rows = gridSize
        self.n_cols = gridSize
        self.createMap()
        self.generateValidStartState()
        self.state_dim = (self.n_rows, self.n_cols, 3)  # n x n x 3
        self.updateState()

    def generateValidStartState(self):
        self.start_r = np.random.randint(1, self.n_rows - 1)
        self.start_c = np.random.randint(1, self.n_cols - 1)
        while self.obstacles[self.start_r, self.start_c] == 1:
            self.start_r = np.random.randint(1, self.n_rows - 1)
            self.start_c = np.random.randint(1, self.n_cols - 1)
        self.r, self.c = self.start_r, self.start_c

    def updateState(self):
        # Reset channels
        self.robot = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        self.robot[self.r, self.c] = 1.0

        # Stack channels: [obstacles, cleaned, robot]
        self.state = np.stack((self.obstacles, self.cleaned, self.robot), axis=-1)

    def createMap(self):
        # Obstacles channel
        self.obstacles = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        # Border walls
        self.obstacles[:, 0] = 1
        self.obstacles[:, -1] = 1
        self.obstacles[0, :] = 1
        self.obstacles[-1, :] = 1

        # Random obstacles
        for _ in range(self.numOfObstacles):
            sizeX = np.random.randint(2, self.ObstaclesMaxSize + 1)
            sizeY = np.random.randint(2, self.ObstaclesMaxSize + 1)
            startX = np.random.randint(1, self.n_rows - sizeX - 1)
            startY = np.random.randint(1, self.n_cols - sizeY - 1)
            self.obstacles[startX:(startX+sizeX), startY:(startY+sizeY)] = 1

        # Cleaned channel (all zeros at start)
        self.cleaned = np.zeros((self.n_rows, self.n_cols), dtype=np.float32)

    def reset(self):
        self.createMap()
        self.generateValidStartState()
        self.updateState()
        return self.state.copy()

    def step(self, a: int, end_of_episode: bool = False):
        if a == 0: nr, nc = self.r - 1, self.c       # up
        elif a == 1: nr, nc = self.r + 1, self.c     # down
        elif a == 2: nr, nc = self.r, self.c - 1     # left
        elif a == 3: nr, nc = self.r, self.c + 1     # right
        else: raise ValueError("Action must be 0..3")

        reward, done = 0.0, False

        # Clean current tile
        self.cleaned[self.r, self.c] = 1.0

        # Case 1: move into free uncleaned space
        if self.obstacles[nr, nc] == 0 and self.cleaned[nr, nc] == 0:
            reward += 5.0

        # Case 2: hit obstacle
        if self.obstacles[nr, nc] == 1:
            reward += -20.0
            done = True

        # Case 3: revisiting a cleaned tile
        if self.cleaned[nr, nc] == 1 and self.obstacles[nr, nc] == 0:
            reward += -2.0

        self.r, self.c = nr, nc
        self.updateState()
        return self.state.copy(), reward, done, {}
