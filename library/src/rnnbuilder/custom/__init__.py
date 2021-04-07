"""Functions to extend this library with custom modules"""
from typing import Union, Callable, Literal, Optional, Type
from torch import nn
from ._factories import NonRecurrentFactory, RecurrentFactory
from abc import ABC, abstractmethod


class CustomModule(nn.Module, ABC):
    """Abstract base class for custom recurrent modules. Inheriting classes must be registered with `register_recurrent`
    to retrieve a corresponding factory class. The factory class will initialize with the same parameters as the
    registered CustomModule.


    """
    def enter_in_shape(self, in_shape):
        """Gets called immediately after initialization. Overwrite if internal parameters depend on the input shape.
        """
        pass

    def get_initial_output(self, batch_size): #TODO: base? change signature to include shape?
        """Returns the initial output used for `Placeholder`s. This defaults to zero and can be overwritten by
        individual `Placerholder`s"""
        pass

    @abstractmethod
    def forward(self, input, state):
        """Must be implemented with the following signature.
        Args:
            input: the input tensor, the expected shape needs to be reported when registering the module, see
                `register_recurrent`
            state: some state as any combination of dicts, lists, tuples and tensors (needs to have format consistent
                with `get_initial_state`). The empty tuple is used to indicate no state.

        Returns:
            (output, new_state) where
            output: the output tensor, format needs to match input but data dimensions can be different
            new_state: the new state in the same format as the input state
            """
        pass

    def get_initial_state(self, batch_size):
        """Returns initial state (in the same format as `forward`, for a single step, first dimension being batch size).
        """
        return ()

class _NonRecurrentGenerator:
    def __init__(self, make_module, prepare_input, shape_change):
        self._make_module = make_module
        self._shape_change = shape_change
        self._prepare_input = prepare_input

    def __call__(self, *args, **kwargs):
        return NonRecurrentFactory(self._make_module, self._prepare_input, self._shape_change, *args, **kwargs)


def register_non_recurrent(*, module_class: Union[Type[nn.Module], Callable[..., nn.Module]],
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




