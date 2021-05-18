import torch

def flatten_to_int(shape):
    out = 1
    for s in shape:
        out *= s
    return out

def _flatten_shape(shape):
    if shape is None:
        return None
    out = 1
    for s in shape:
        out *= s
    return (out,)

def _shape_sum(shapes):
    if None in shapes:
        return None
    return (sum([flatten_to_int(shape) for shape in shapes]),)

def _any(iterable):
    for x in iterable:
        if x:
            return x
    return None



class StateContainerNew:
    def __init__(self, init_state, add_dims=None, device=None, state_storage_type=None):
        self.state = init_state if not (add_dims or device or state_storage_type) else self._make(init_state, add_dims, device, state_storage_type)

    def _make(self, state, add_dims, device, state_storage_type):
        if isinstance(state, dict):
            new_log = {k: self._make(v, add_dims, device, state_storage_type) for k, v in state.items()}
        elif isinstance(state, (list, tuple)):
            new_log = [self._make(v, add_dims, device, state_storage_type) for v in state]
        elif isinstance(state, torch.Tensor):
            new_log = torch.empty(add_dims + state.shape[1:], device=device, dtype=state_storage_type) #state.to(device).reshape(add_dims + state.shape)
        else:
            raise Exception('Unknown type in model state!')
        return new_log

    def transfer(self, other):
        return self._transfer(self.state, other)

    def _transfer(self, state, other):
        if isinstance(state, dict):
            for k in state:
                for state1, other1 in self._transfer(state[k], other[k]):
                    yield state1, other1
        elif isinstance(state, (list, tuple)):
            for k in range(len(state)):
                for state1, other1 in self._transfer(state[k], other[k]):
                    yield state1, other1
        elif isinstance(state, torch.Tensor):
            yield state, other
        else:
            raise Exception('Unknown type in model state!')


    def get(self, fn, *others):
        return self._get(self.state, fn, *others)

    def _get(self, state, fn, *others):
        if isinstance(state, dict):
            new_log = {k: self._get(v, fn, *[d[k] for d in others]) for k, v in state.items()}
        elif isinstance(state, (list, tuple)):
            new_log = [self._get(v, fn, *[d[i] for d in others]) for i, v in enumerate(state)]
        elif isinstance(state, torch.Tensor):
            new_log = fn(state, *others)
        else:
            raise Exception('Unknown type in model state!')
        return new_log

    def get_shape(self):
        return self._get_shape(self.state)

    def _get_shape(self, state):
        if isinstance(state, dict):
            for v in state.values():
                info = self._get_shape(v)
                if info:
                    return info
        elif isinstance(state, (list, tuple)):
            for v in state:
                info = self._get_shape(v)
                if info:
                    return info
        elif isinstance(state, torch.Tensor):
            return state.shape
        else:
            raise Exception('Unknown type in model state!')
        return None