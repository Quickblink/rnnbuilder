from .modules import OuterModule

class ModuleFactory:
    def make_model(self, in_shape):
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