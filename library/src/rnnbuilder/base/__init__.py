"""Base classes not intended for direct use."""
from ._modules import OuterModule
import rnnbuilder as rb

class ModuleFactory:
    """Factory base class. Factories are used to build networks recursively. Call `make_model` to get a PyTorch model.
    """
    def make_model(self, in_shape):
        """Returns a usable PyTorch model corresponding to the factory. Every call returns a model independent from
        previous calls with its own parameters. This applies to sub-modules as well. Parameter sharing is not supported.
        """
        return OuterModule(self._assemble_module(in_shape, False))

    '''
    def __init__(self):
        self.module_class = None
        self.args = None

    def shape_change(self, in_shape):
        return in_shape

    def make_module(self, in_shape):
        return self.module_class(in_shape, *self.args)

    def assemble_module(self, in_shape, unrolled):
        return self.make_module(in_shape)

    def __call__(self, in_size):
        return self.module_class(in_size, *self.args)
    '''


class LayerInput:
    """Base class for inputs to `Layer` in `Network`"""
    _mode = 'stack'


class LayerBase(LayerInput):
    """Base class for layers of a `Network`"""
    def __init__(self):
        self._registered = False

    def sum(self, *layers: 'LayerBase'):
        """adds up self and given layers to form an input for a `Layer`. Data dimensions need to be identical.
                """
        if not self._registered:
            raise Exception('This layer is not part of a network yet. Chaining is not allowed.')
        return rb.Sum(self, *layers)

    def stack(self, *layers: 'LayerBase') -> 'rb.Stack':
        """stacks self and given layers along the first data dimension using `torch.cat` to form an input for a `Layer`
        """
        if not self._registered:
            raise Exception('This layer is not part of a network yet. Chaining is not allowed.')
        return rb.Stack(self, *layers)

    def apply(self, *module_facs, placeholder=None):
        """A `rnnbuilder.Layer` is formed by applying the given `ModuleFactory`s to the output of the calling layer."""
        if not self._registered:
            raise Exception('This layer is not part of a network yet. Chaining is not allowed.')
        return rb.Layer(self, module_facs, placeholder=placeholder)


class InputBase(LayerInput):
    """Base class for input aggregations
    """
    def __init__(self, *layers: LayerBase):
        self.layers = layers

    def apply(self, *module_facs, placeholder=None):
        """A `rnnbuilder.Layer` is formed by applying the given `ModuleFactory`s to the calling input aggregate."""
        return rb.Layer(self, module_facs, placeholder=placeholder)