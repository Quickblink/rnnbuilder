from typing import Union, Callable, Literal, Optional, Type
from torch import nn

from .factories import NonRecurrentFactory, RecurrentFactory


class CustomModule(nn.Module):
    def enter_in_shape(self, in_shape):
        pass

    def get_initial_output(self, batch_size):
        pass

    def get_initial_state(self, batch_size):
        return ()

class _NonRecurrentGenerator:
    def __init__(self, make_module, prepare_input, shape_change):
        self._make_module = make_module
        self._shape_change = shape_change
        self._prepare_input = prepare_input

    def __call__(self, *args, **kwargs):
        return NonRecurrentFactory(self._make_module, self._prepare_input, self._shape_change, *args, **kwargs)


def register_non_recurrent(*, module_class: Union[Type[nn.Module],Callable[..., nn.Module]],
                                 prepare_input: Literal['flatten', 'keep'] = 'keep',
                                 shape_change: Union[Literal['none', 'auto'], Callable[..., tuple]]):
    initializer = (lambda in_shape, *args, **kwargs: module_class(*args, **kwargs)) if type(module_class) is type else module_class
    return _NonRecurrentGenerator(initializer, prepare_input, shape_change)


class _RecurrentGenerator:
    def __init__(self, module_class, prepare_input, single_step, unroll_full_state, shape_change_method):
        self._module_class = module_class
        self._prepare_input = prepare_input
        self._single_step = single_step
        self._unroll_full_state = unroll_full_state
        self._shape_change_method = shape_change_method

    def __call__(self, *args, **kwargs):
        return RecurrentFactory(self._module_class, self._prepare_input, self._single_step, self._unroll_full_state,
                                self._shape_change_method, *args, **kwargs)

# TODO: merge single_step and unroll_full_state into one string variable
def register_recurrent(*, module_class: Type[CustomModule], prepare_input: Literal['flatten', 'keep'] = 'keep',
                       single_step: bool, unroll_full_state: bool = True,
                       shape_change: Union[Literal['none', 'auto'], Callable[[Optional[tuple]], tuple]]):
    return _RecurrentGenerator(module_class, prepare_input, single_step, unroll_full_state, shape_change)




