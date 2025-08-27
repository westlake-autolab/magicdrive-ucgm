class Linear:

    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1


class VLinear:
    def set_timesteps(self, timesteps):
        self.timesteps = timesteps
        num_timesteps = len(timesteps)
        self.reverse = timesteps[0] > timesteps[-1]

        alpha_ins = {}
        gamma_ins = {}
        for i, t_in in enumerate(timesteps):
            t_out = timesteps[i + 1] if i < num_timesteps - 1 else timesteps[i]
            dt = t_in - t_out
            if self.self.reverse:
                dt = -dt

            alpha_ins[self.timesteps[i]] = dt
            gamma_ins[self.timesteps[i]]
        self.alpha_ins = alpha_ins

    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1

class Linear2:
    def __init__(self, inverse=True):
        self.inverse = inverse

    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return -1

    def gamma_to(self, t):
        return 1
