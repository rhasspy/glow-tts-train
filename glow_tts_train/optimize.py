"""Classes for optimizer"""
import typing

import numpy as np
import torch


class Adam:
    def __init__(
        self,
        params,
        scheduler,
        dim_model,
        warmup_steps: int = 4000,
        lr: float = 1e0,
        betas: typing.Tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-9,
    ):
        self.params = params
        self.scheduler = scheduler
        self.dim_model = dim_model
        self.warmup_steps = warmup_steps
        self.lr = lr
        self.betas = betas
        self.eps = eps

        self.step_num = 1
        self.cur_lr = lr * self._get_lr_scale()

        self._optim = torch.optim.Adam(params, lr=self.cur_lr, betas=betas, eps=eps)

    def _get_lr_scale(self):
        if self.scheduler == "noam":
            return np.power(self.dim_model, -0.5) * np.min(
                [
                    np.power(self.step_num, -0.5),
                    self.step_num * np.power(self.warmup_steps, -1.5),
                ]
            )

        return 1

    def _update_learning_rate(self):
        self.step_num += 1
        if self.scheduler == "noam":
            self.cur_lr = self.lr * self._get_lr_scale()
            for param_group in self._optim.param_groups:
                param_group["lr"] = self.cur_lr

    def get_lr(self):
        return self.cur_lr

    def step(self):
        self._optim.step()
        self._update_learning_rate()

    def zero_grad(self):
        self._optim.zero_grad()

    def load_state_dict(self, d):
        self._optim.load_state_dict(d)

    def state_dict(self):
        return self._optim.state_dict()


OptimizerType = Adam
