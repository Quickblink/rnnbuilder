"""Functions to extend this library with custom modules"""
from typing import Type
import torch as _torch
from ._factories import _NonRecurrentFactory, _RecurrentFactory
from abc import ABC as _ABC, abstractmethod as _abstractmethod

__pdoc__ = {'CustomModule.training' : False, 'CustomModule.dump_patches' : False}


class CustomModule(_torch.nn.Module, _ABC):
    """Abstract base class for custom recurrent modules. Inheriting classes must be registered with `register_recurrent`
    to retrieve a corresponding factory class. The factory class will initialize with the same parameters as the
    registered CustomModule.


    """
    def enter_in_shape(self, in_shape):
        """Gets called immediately after initialization. Overwrite if internal parameters depend on the input shape.
        """
        pass

    @_abstractmethod
    def get_out_shape(self, in_shape):
        """
        Args:
            in_shape: data shape of incoming tensor (excluding time and batch dimensions) before any flattening is
                applied

        In most cases, one of three values should be returned:
        1. 'in_shape' if the module does not perform changes to the shape of the data (flattening does not count)
        2. some_fixed_shape if the output shape is independent of the input e.g. (out_shape,) for Linear layers
        3. 'None' in all other cases (output shape will be inferred automatically)
        """
        pass

    def get_initial_output(self, full_shape):
        """Returns the initial output used for `Placeholder`s. This defaults to zero and can be overwritten by
        individual `Placeholder`s.

        Args:
            full_shape: shape of the output in the format (batch, data0, data1, ...).
        """
        return _torch.zeros(full_shape, device=self.device)

    @_abstractmethod
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
        return _NonRecurrentFactory(self._make_module, self._prepare_input, self._shape_change, *args, **kwargs)


def register_non_recurrent(*, module_class: Type[_torch.nn.Module],
                           flatten_input: bool,
                           shape_change: bool):
    """Register a (non-recurrent) `torch.nn.Module` to retrieve a factory class. The factory class will initialize with
    the same parameters as the registered Module. If this interface is too restrictive, wrap the Module in a
    `CustomModule` and use `register_recurrent` instead.

    Args:
        module_class: a class derived from `torch.nn.Module`. The forward method needs to conform to a fixed signature.
            It must accept a single input tensor of format (batch, data0, data1, ...) and return a single output tensor.
        flatten_input: If True, the input is flattened to shape (batch, data0*data1*...) before given to the module.
        shape_change: Indicate whether the module changes the shape of the tensor going through, i.e. put False if input
            (after the optional flatten) and output shapes of the module are identical otherwise True
    """
    initializer = (lambda in_shape, *args, **kwargs: module_class(*args, **kwargs))\
        if type(module_class) is type else module_class
    return _NonRecurrentGenerator(initializer, 'flatten' if flatten_input else 'keep',
                                  'auto' if shape_change else 'none')


class _RecurrentGenerator:
    def __init__(self, module_class, prepare_input, single_step, unroll_full_state):
        self._module_class = module_class
        self._prepare_input = prepare_input
        self._single_step = single_step
        self._unroll_full_state = unroll_full_state

    def __call__(self, *args, **kwargs):
        return _RecurrentFactory(self._module_class, self._prepare_input, self._single_step, self._unroll_full_state,
                                 *args, **kwargs)

def register_recurrent(*, module_class: Type[CustomModule], flatten_input: bool,
                       single_step: bool, unroll_full_state: bool = True):
    """Register a (possibly recurrent) `CustomModule` to retrieve a factory class. The factory class will initialize
    with the same parameters as the registered `CustomModule`.

    Args:
        module_class: a class derived from `CustomModule`
        flatten_input: If True, the input is flattened to a single data dimension before given to the module.
        single_step: If True, for each time step the forward method of the module will be invoked with a tensor of
            format (batch, data_shape). If False, it will be only invoked once with a tensor of format
            (time, batch, data_shape).
        unroll_full_state: (Optional) If True and the state for every time step is required (full_state mode), the
            module will be invoked for each time step even if single_step=False. If False, the module has to check for
            self._full_state and return a sequence of states appropriately
    """
    if module_class.__abstractmethods__:
        raise Exception("Can't register "+module_class.__name__+". All abstract methods need to be overwritten.")
    return _RecurrentGenerator(module_class, 'flatten' if flatten_input else 'keep', single_step, unroll_full_state)



