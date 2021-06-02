import torch

from ..base._modules import _ModuleBase
from ..base._utils import StateContainerNew

class _StatelessWrapper(_ModuleBase):
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



class RecurrentWrapper(_ModuleBase):
    def __init__(self, in_shape, out_shape, inner, single_step, full_state_unroll, is_unrolled):
        super().__init__(in_shape)
        self.out_shape = out_shape
        self.inner = inner
        self.single_step = single_step
        self.full_state_unroll = full_state_unroll
        self._full_state = False
        self.is_unrolled = is_unrolled

    def forward(self, x, h):
        x_con = StateContainerNew(x)
        time, batch = x_con.get_shape()[:2]
        if (not self.is_unrolled) and (self.single_step or (self._full_state and self.full_state_unroll)):
            output = torch.empty((time, batch)+self.out_shape, device=self.device)
            if self._full_state:
                state = StateContainerNew(h, (time, batch), self.device)
            for t in range(time):
                fn = lambda x, shape : x[t].reshape((batch,)+shape) if self.single_step else x[t].reshape((1, batch)+shape)
                out, h = self.inner(x_con.get(fn, self.in_shape), h)
                output[t] = out if self.single_step else out[0]
                if self._full_state:
                    for container, entry in state.transfer(h):
                        container[t] = entry
            new_state = state.state if self._full_state else h
        else:
            x = x_con.get(lambda x, shape: x.reshape((batch,)+shape) if self.single_step else x.reshape((time, batch) + shape), self.in_shape)
            out, new_state = self.inner(x, h)
            output = out.unsqueeze(0) if self.single_step else out
        return output, new_state

    def get_initial_state(self, batch_size):
        return self.inner.get_initial_state(batch_size)

    def get_initial_output(self, batch_size):
        return self.inner.get_initial_output((batch_size,)+self.out_shape)