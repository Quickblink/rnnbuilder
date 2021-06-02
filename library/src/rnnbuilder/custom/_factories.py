import torch

from ..base import ModuleFactory
from ..base._utils import _flatten_shape
from ._modules import _StatelessWrapper, RecurrentWrapper

class _NonRecurrentFactory(ModuleFactory):
    def __init__(self, make_module, prepare_input, shape_change_method, *args, **kwargs):
        super().__init__()
        self._make_module = make_module
        self._prepare_input = prepare_input
        self._shape_change_method = shape_change_method
        self._args = args
        self._kwargs = kwargs
        self._buffered_module = None
        self._buffered_shape = None


    def _shape_change(self, in_shape):
        if callable(self._shape_change_method):
            return self._shape_change_method(in_shape, *self._args, **self._kwargs)
        elif self._shape_change_method == 'none' or in_shape is None:
            return in_shape
        else:
            if not self._buffered_module or in_shape != self._buffered_shape:
                self._buffered_module = self._make_module(in_shape, *self._args, **self._kwargs)
                self._buffered_shape = in_shape
            in_shape = _flatten_shape(in_shape) if self._prepare_input == 'flatten' else in_shape
            return self._buffered_module(torch.zeros((1,)+in_shape)).shape[1:]



    def _assemble_module(self, in_shape, unrolled):
        in_shape = _flatten_shape(in_shape) if self._prepare_input == 'flatten' else in_shape
        out_shape = self._shape_change(in_shape)
        module = self._buffered_module or self._make_module(in_shape, *self._args, **self._kwargs)
        self._buffered_module = None
        return _StatelessWrapper(in_shape, out_shape, module)

class _RecurrentFactory(ModuleFactory):
    def __init__(self, module_class, prepare_input, single_step, unroll_full_state, *args, **kwargs):
        super().__init__()
        self._module_class = module_class
        self._prepare_input = prepare_input
        self._single_step = single_step
        self._unroll_full_state = unroll_full_state
        self._args = args
        self._kwargs = kwargs
        self._buffered_module = None
        self._buffered_shape = None
        self._test_shape = (1,) if single_step else (10,1)

    def _make_module(self, in_shape, *args, **kwargs):
        module = self._module_class(*args, **kwargs)
        module.enter_in_shape(in_shape)
        return module


    def _shape_change(self, in_shape):
        if not self._buffered_module or in_shape != self._buffered_shape:
            self._buffered_module = self._module_class(*self._args, **self._kwargs)
            self._buffered_shape = in_shape
            if all(in_shape):
                self._buffered_module.enter_in_shape(in_shape)
        out_shape = self._buffered_module.get_out_shape(in_shape)
        if all(in_shape) and not out_shape:
            in_shape = _flatten_shape(in_shape) if self._prepare_input == 'flatten' else in_shape
            return self._buffered_module(torch.zeros(self._test_shape + in_shape),
                                         self._buffered_module.get_initial_state(1))[0].shape[2 - self._single_step:]
        return out_shape



    def _assemble_module(self, in_shape, unrolled):
        in_shape = _flatten_shape(in_shape) if self._prepare_input == 'flatten' else in_shape
        out_shape = self._shape_change(in_shape)
        module = self._buffered_module or self._make_module(in_shape, *self._args, **self._kwargs)
        self._buffered_module = None
        return RecurrentWrapper(in_shape, out_shape, module, self._single_step, self._unroll_full_state, unrolled)