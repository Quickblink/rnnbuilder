from base import ModuleBase, StatelessWrapper
from factories import ModuleFactory, flatten_shape
from typing import Union, Callable, Literal

class NonRecurrentFactory(ModuleFactory):
    def __init__(self, module_class, init_shape, shape_change_method, *args, **kwargs):
        super().__init__()
        self._module_class = module_class
        self._init_shape = init_shape
        self._shape_change_method = shape_change_method
        self._args = args
        self._kwargs = kwargs

    def _make_module(self, in_shape):
        if self._init_shape:
            return self._module_class(in_shape)
        else:

    def _shape_change(self, in_shape):
        if self._shape_change_method == 'none' or in_shape is None:
            return in_shape
        elif self._shape_change_method == 'auto':
            test_module = self._make_module(in_shape)

        return (self.out_size,)


    def _assemble_module(self, in_shape, unrolled):
        in_shape = flatten_shape(in_shape)
        return StatelessWrapper(in_shape, nn.Linear(in_shape[0], self.out_size), (self.out_size,))


class NonRecurrentGenerator:
    def __init__(self, make_module, shape_change: Union[Literal['none', 'auto'], Callable[[tuple], tuple]]):
        self._make_module = make_module
        self._init_shape = init_shape
        self._shape_change = shape_change

    def __call__(self, *args, **kwargs):
        return NonRecurrentFactory(self._module_class, self._init_shape, self._shape_change, *args, **kwargs)


def register_non_recurrent(*, module_class: ModuleBase, init_shape: bool,
                 shape_change: Union[Literal['none', 'auto'], Callable[[tuple], tuple]]):

register_non_recurrent(module_class=None, init_shape=True, shape_change='none')

test_function = lambda shape, *args: (shape, args)
