import numpy as np
import torch


class TrigFlow:

    def __init__(self, scale=np.pi/2):
        self.scale = scale

    def alpha_in(self, t):
        return torch.sin(t * self.scale)

    def gamma_in(self, t):
        return torch.cos(t * self.scale)

    def alpha_to(self, t):
        return torch.cos(t * self.scale)

    def gamma_to(self, t):
        return -torch.sin(t * self.scale)
