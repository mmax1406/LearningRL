import numpy as np

class GridCleanEnv:
    def __init__(self, gridSize: int, numOfObstacles: int, ObstaclesMaxSize: int):
        self.numOfObstacles = numOfObstacles
        self.ObstaclesMaxSize = ObstaclesMaxSize
        self.n_rows = gridSize
        self.n_cols = gridSize
        self.createMap()
        self.start_grid = self.grid.copy()
        self.generateValitStartState()
        self.state_dim = self.n_rows * self.n_cols + 2
        self.updateState()

    def generateValitStartState(self):
        self.start_r = np.random.random_integers(2, self.n_rows - 1)
        self.start_c = np.random.random_integers(2, self.n_cols - 1)
        while (self.start_grid[self.start_r, self.start_c] == 1):
            self.start_r = np.random.random_integers(2, self.n_rows - 1)
            self.start_c = np.random.random_integers(2, self.n_cols - 1)
        self.r, self.c = self.start_r, self.start_c

    def updateState(self):
        self.state = np.concatenate((
            np.array([self.r, self.c], dtype=np.float32).reshape(-1, 1),
            self.grid.reshape(-1).astype(np.float32).reshape(-1, 1)
        ))

    def createMap(self):
        self.grid = np.zeros((self.n_rows, self.n_cols))
        self.grid[:,0] = 1
        self.grid[:, self.n_cols-1] = 1
        self.grid[0, :] = 1
        self.grid[self.n_rows-1, :] = 1

        for ii in range(self.numOfObstacles):
            sizeX = np.random.random_integers(2, self.ObstaclesMaxSize)
            sizeY = np.random.random_integers(2, self.ObstaclesMaxSize)
            startX = np.random.random_integers(1,self.n_rows-2-sizeX)
            startY = np.random.random_integers(1,self.n_cols-2-sizeY)

            self.grid[startX:(startX+sizeX), startY:(startY+sizeY)] = 1

    def reset(self):
        self.generateValitStartState()
        self.grid = self.start_grid.copy()
        self.updateState()
        return self.state.reshape(-1)

    def step(self, a: int, end_of_episode: bool):
        if a == 0: nr, nc = max(0, self.r - 1), self.c       # up
        elif a == 1: nr, nc = min(self.n_rows - 1, self.r + 1), self.c  # down
        elif a == 2: nr, nc = self.r, max(0, self.c - 1)     # left
        elif a == 3: nr, nc = self.r, min(self.n_cols - 1, self.c + 1) # right
        else: raise ValueError("Action must be 0..3")

        self.grid[self.r, self.c] = 0.5

        reward, done = 0.0, False #make a step

        # Case 1 clean new tile
        if self.grid[nr, nc] == 0.0:
            reward += 5.0

        # Case 2: hit a wall → episode ends
        if self.grid[nr, nc] == 1.0:
            reward += -20.0
            done = True

        # # Case 2: episode ended by some external condition
        # if end_of_episode or done:
        #     reward += np.sum(self.grid == 0.5) * 3 # e.g. number of cleaned tiles
        #     done = True

        # Case 3: revisiting a cleaned tile
        if self.grid[nr, nc] == 0.5:
            reward += -2.0

        self.r, self.c = nr, nc
        self.updateState()
        return self.state.reshape(-1), reward, done, {}
