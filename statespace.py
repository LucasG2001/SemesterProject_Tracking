import numpy as np


class States:
    def __init__(self):
        self.dx = 0  # x change
        self.dy = 0  # y change
        self.vx = 0  # x velocity
        self.vy = 0  # y velocity
        self.dist = 0  # total distance

    def get_states(self):
        return self.dx, self.dy, self.vx, self.vy, self.dist

    def update_states(self, new_dx, new_dy, loop_time):
        diff_x = new_dx - self.dx
        diff_y = new_dy - self.dy
        dr = np.array([diff_x, diff_y])
        self.vx = (new_dx - self.dx) / loop_time
        self.vy = (new_dy - self.dy) / loop_time
        self.dx = new_dx
        self.dy = new_dy
        self.dist += np.linalg.norm(dr)
