import torch.optim as optim


class LRScheduler:
    def __init__(self, parameters, hidden_size, warmup, step=0):
        self.constant = 2.0 * (hidden_size ** -0.5)
        self.cur_step = step
        self.warmup = warmup
        self.optimizer = optim.Adam(parameters, lr=self.learning_rate(),
                                    betas=(0.9, 0.997), eps=1e-09)

    def step(self):
        self.cur_step += 1
        rate = self.learning_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def learning_rate(self):
        lr = self.constant
        lr *= min(1.0, self.cur_step / self.warmup)
        lr *= max(self.cur_step, self.warmup) ** -0.5
        return lr
