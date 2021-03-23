import torch

from ..base.modules import ModuleBase
from ..base.utils import StateContainerNew

class StatelessWrapper(ModuleBase):
    def __init__(self, in_shape, out_shape, inner):
        super().__init__(in_shape)
        self.inner = inner
        self.out_shape = out_shape

    def forward(self, x, h):
        time = x.shape[0]
        batch = x.shape[1]
        x = x.reshape((time*batch,)+self.in_shape)
        x = self.inner(x)
        return x.reshape((time, batch)+self.out_shape), ()

    def get_initial_output(self, batch_size):
        return torch.zeros((1, batch_size)+self.out_shape, device=self.device)


class RecurrentWrapper(ModuleBase):
    def __init__(self, in_shape, out_shape, inner, single_step, full_state_unroll):
        super().__init__(in_shape)
        self.out_shape = out_shape
        self.inner = inner
        self.single_step = single_step
        self.full_state_unroll = full_state_unroll
        self._full_state = False

    def forward(self, x, h):
        time = x.shape[0]
        batch = x.shape[1]
        if self.single_step or (self._full_state and self.full_state_unroll):
            output = torch.empty((time, batch)+self.out_shape, device=self.device)
            if self._full_state:
                state = StateContainerNew(h, (time, batch), self.device)
            for t in range(time):
                out, h = self.inner((x[t].reshape((batch,)+self.in_shape) if self.single_step else x[t].reshape((1, batch)+self.in_shape)), h)
                output[t] = out if self.single_step else out[0]
                if self._full_state:
                    for container, entry in state.transfer(h):
                        container[t] = entry.rename(None)
            new_state = state.state if self._full_state else h
        else:
            x = x.reshape((time, batch) + self.in_shape)
            output, new_state = self.inner(x, h)
        return output, new_state

    def get_initial_state(self, batch_size):
        return self.inner.get_initial_state(batch_size)

    def get_initial_output(self, batch_size):
        return self.inner.get_initial_output(batch_size).unsqueeze(0)