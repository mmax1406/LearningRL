import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridCleanEnvGym(gym.Env):
    """Gymnasium environment for grid cleaning robot."""

    metadata = {"render_modes": []}

    def __init__(self, grid_size=10, num_obstacles=3, obstacles_max_size=3, max_steps=200):
        super().__init__()

        # Environment parameters
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.obstacles_max_size = obstacles_max_size
        self.max_steps = max_steps
        self.position_history = []  # stores last positions
        self.position_history_maxSize = 10
        self.max_repeats = 5

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.float32
        )

        # Internal state
        self.state = None
        self.r = 0
        self.c = 0
        self.steps = 0

        self._generate_map()

    # ---------------- Internal helper methods ----------------
    def _generate_map(self):
        """Create obstacle and cleaned maps."""
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        # Border walls
        self.obstacles[:, 0] = 1
        self.obstacles[:, -1] = 1
        self.obstacles[0, :] = 1
        self.obstacles[-1, :] = 1

        # Random obstacles
        for _ in range(self.num_obstacles):
            size_x = np.random.randint(2, self.obstacles_max_size + 1)
            size_y = np.random.randint(2, self.obstacles_max_size + 1)
            start_x = np.random.randint(1, self.grid_size - size_x - 1)
            start_y = np.random.randint(1, self.grid_size - size_y - 1)
            self.obstacles[start_x:start_x+size_x, start_y:start_y+size_y] = 1

        self.cleaned = np.zeros_like(self.obstacles)

    def _sample_start(self):
        """Pick a valid starting position for the robot."""
        while True:
            r = np.random.randint(1, self.grid_size - 1)
            c = np.random.randint(1, self.grid_size - 1)
            if self.obstacles[r, c] == 0:
                return r, c

    def _update_state(self):
        """Stack channels into observation."""
        robot_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        robot_map[self.r, self.c] = 1.0
        self.state = np.stack((self.obstacles, self.cleaned, robot_map), axis=-1)

    # ---------------- Gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_map()
        self.r, self.c = self._sample_start()
        self.cleaned = np.zeros_like(self.cleaned)
        self.steps = 0
        self._update_state()
        self.position_history = [(self.r, self.c)]  # reset history
        info = {}
        return self.state.copy(), info

    def step(self, action):
        self.steps += 1
        nr, nc = self.r, self.c

        # Determine next position
        if action == 0: nr -= 1  # up
        elif action == 1: nr += 1  # down
        elif action == 2: nc -= 1  # left
        elif action == 3: nc += 1  # right
        else:
            raise ValueError("Action must be 0..3")

        reward = 0.0
        done = False

        # Clean current tile
        self.cleaned[self.r, self.c] = 1.0

        # Move logic
        if self.obstacles[nr, nc] == 1:      # hit obstacle
            reward = -10
            done = True
        elif self.cleaned[nr, nc] == 1:      # revisiting cleaned tile
            reward = -3
            self.r, self.c = nr, nc
        else:                                # free uncleaned space
            reward = 1
            self.r, self.c = nr, nc

        # End if too many steps were made
        if self.steps >= self.max_steps:
            done = True

        self._update_state()
        truncated = self.steps >= self.max_steps

        # Check we are not stuck in a loop
        self.position_history.append((self.r, self.c))
        if len(self.position_history) > self.position_history_maxSize:
            self.position_history.pop(0)
        repeat_count = self.position_history.count((self.r, self.c))
        if repeat_count >= self.max_repeats:
            done = True  # terminate episode

        return self.state.copy(), reward, done, truncated, {}

    def render(self):
        """No rendering during training."""
        pass

    def close(self):
        pass
