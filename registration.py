from base import ModuleBase, StatelessWrapper
from factories import ModuleFactory, flatten_shape
from typing import Union, Callable, Literal, Optional, Type
import torch
from torch import nn
from utils import StateContainerNew


class NonRecurrentFactory(ModuleFactory): #TODO: prepare_input
    def __init__(self, make_module, shape_change_method, *args, **kwargs):
        super().__init__()
        self._make_module = make_module
        self._shape_change_method = shape_change_method
        self._args = args
        self._kwargs = kwargs
        self._buffered_module = None
        self._buffered_shape = None


    def _shape_change(self, in_shape):
        if self._shape_change_method == 'none' or in_shape is None:
            return in_shape
        elif self._shape_change_method == 'auto':
            if not self._buffered_module or in_shape != self._buffered_shape:
                self._buffered_module = self._make_module(in_shape, *self._args, **self._kwargs)
                self._buffered_shape = in_shape
            return self._buffered_module(torch.zeros((1,)+in_shape)).shape[1:]
        else:
            return self._shape_change_method(in_shape, *self._args, **self._kwargs)


    def _assemble_module(self, in_shape, unrolled):
        in_shape = flatten_shape(in_shape)
        out_shape = self._shape_change(in_shape)
        module = self._buffered_module or self._make_module(in_shape, *self._args, **self._kwargs)
        self._buffered_module = None
        return StatelessWrapper(in_shape, out_shape, module)


class NonRecurrentGenerator:
    def __init__(self, make_module, shape_change):
        self._make_module = make_module
        self._shape_change = shape_change

    def __call__(self, *args, **kwargs):
        return NonRecurrentFactory(self._make_module, self._shape_change, *args, **kwargs)


def register_non_recurrent_class(*, module_class: Type[nn.Module], prepare_input: Literal['flatten', 'keep'] = 'keep',
                           shape_change: Union[Literal['none', 'auto'], Callable[[Optional[tuple]], tuple]]):
    return NonRecurrentGenerator(lambda in_shape, *args, **kwargs: module_class(*args, **kwargs), shape_change)


def register_non_recurrent_initializer(*, initializer: Callable[..., nn.Module], prepare_input: Literal['flatten', 'keep'] = 'keep',
                                       shape_change: Union[Literal['none', 'auto'], Callable[[Optional[tuple]], tuple]]):
    return NonRecurrentGenerator(initializer, shape_change)



class MultiStepWrapper(ModuleBase):
    def __init__(self, in_shape, out_shape, module):
        super().__init__(in_shape)
        self.out_shape = out_shape
        self.module = module

    def forward(self, x, h):
        time = x.shape[0]
        batch = x.shape[1]
        x = x.reshape((time, batch)+self.in_shape)
        x, n_h = self.inner(x, h)
        return x.reshape((time, batch)+self.out_shape), n_h

    def get_initial_output(self, batch_size):
        return torch.zeros((1, batch_size)+self.out_shape)

class MultiStepFactory(ModuleFactory):
    def __init__(self, module_class, prepare_input, shape_change_method, *args, **kwargs):
        super().__init__()
        self._module_class = module_class
        self._prepare_input = prepare_input
        self._shape_change_method = shape_change_method
        self._args = args
        self._kwargs = kwargs
        self._buffered_module = None
        self._buffered_shape = None

    def _make_module(self, in_shape):
        module = self._module_class(*self._args, **self._kwargs)
        module.enter_in_shape(in_shape)
        return module

    def _assemble_module(self, in_shape, unrolled):
        in_shape = flatten_shape(in_shape) if self._prepare_input == 'flatten' else in_shape
        out_shape = self._shape_change(in_shape)
        module = self._buffered_module or self._make_module(in_shape)
        self._buffered_module = None
        return MultiStepWrapper(in_shape, out_shape, module)


class CustomModule(nn.Module):
    def enter_in_shape(self, in_shape):
        pass


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
            new_state = state.state if self.full_state else h
        else:
            x = x.reshape((time, batch) + self.in_shape)
            output, new_state = self.inner(x, h)
        return output, new_state

    def get_initial_state(self, batch_size):
        return self.inner.get_initial_state(batch_size)

    def get_initial_output(self, batch_size):
        return self.inner.get_initial_output(batch_size)


class RecurrentFactory(ModuleFactory):
    def __init__(self, module_class, prepare_input, single_step, unroll_full_state, shape_change_method, *args, **kwargs):
        super().__init__()
        self._module_class = module_class
        self._shape_change_method = shape_change_method
        self._prepare_input = prepare_input
        self._single_step = single_step
        self._unroll_full_state = unroll_full_state
        self._args = args
        self._kwargs = kwargs
        self._buffered_module = None
        self._buffered_shape = None

    def _make_module(self, in_shape, *args, **kwargs):
        module = self._module_class(*args, **kwargs)
        module.enter_in_shape(in_shape)
        return module


    def _shape_change(self, in_shape):
        if self._shape_change_method == 'none' or in_shape is None:
            return in_shape
        elif self._shape_change_method == 'auto':
            if not self._buffered_module or in_shape != self._buffered_shape:
                self._buffered_module = self._make_module(in_shape, *self._args, **self._kwargs)
                self._buffered_shape = in_shape
            return self._buffered_module(torch.zeros((1,)+in_shape)).shape[1:]
        else:
            return self._shape_change_method(in_shape, *self._args, **self._kwargs)


    def _assemble_module(self, in_shape, unrolled):
        in_shape = flatten_shape(in_shape) if self._prepare_input == 'flatten' else in_shape
        out_shape = self._shape_change(in_shape)
        module = self._buffered_module or self._make_module(in_shape, *self._args, **self._kwargs)
        self._buffered_module = None
        return RecurrentWrapper(in_shape, out_shape, module, self._single_step, self._unroll_full_state)





# TODO: enter in_shape not via init but afterwards
# TODO: don't forget data shape preparation, default to 'flatten', otherwise 'keep'
def register_single_step_recurrent(*, module_class: ModuleBase, prepare_input: Literal['flatten', 'keep'] = 'keep',
                                   shape_change: Union[Literal['none', 'auto'], Callable[[Optional[tuple]], tuple]]):
    pass




class RecurrentGenerator:
    def __init__(self, module_class, prepare_input, single_step, unroll_full_state, shape_change_method):
        self._module_class = module_class
        self._prepare_input = prepare_input
        self._single_step = single_step
        self._unroll_full_state = unroll_full_state
        self._shape_change_method = shape_change_method

    def __call__(self, *args, **kwargs):
        return RecurrentFactory(self._module_class, self._prepare_input, self._single_step, self._unroll_full_state,
                                self._shape_change_method, *args, **kwargs)


#class
#prepare_input
#single or multi -step
#unroll for full_state, what if not unrolled?
#shape_change

def register_recurrent(*, module_class: Type[CustomModule], prepare_input: Literal['flatten', 'keep'] = 'keep',
                       single_step: bool, shape_change: Union[Literal['none', 'auto'], Callable[[Optional[tuple]], tuple]],
                       unroll_full_state: bool = True):
    return RecurrentGenerator(module_class, prepare_input, single_step, unroll_full_state, shape_change)



def register_multi_step_recurrent(*, module_class: ModuleBase, prepare_input: Literal['flatten', 'keep'] = 'keep',
                                  shape_change):
    pass