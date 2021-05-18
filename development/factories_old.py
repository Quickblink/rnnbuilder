import torch
from networks_old import SequenceWrapperFull, ConvWrapper3d, ConvWrapper2, flatten, NetworkModule, LinearModule,\
    LIFNeuron, DiscontinuousNeuron, ConvModule, FlattenerModule, ReLuWrapper, LSTMWrapperOneStep, OutputNeuron


class ModuleFactory:
    def __init__(self):
        self.module_class = None
        self.args = None

    def size_change(self, in_size):
        return in_size

    def __call__(self, in_size):
        return self.module_class(in_size, *self.args)


class FixedSize(ModuleFactory):
    def __init__(self, out_size):
        super().__init__()
        self.out_size = out_size

    def size_change(self, in_size):
        return self.out_size


class Linear(FixedSize):
    def __init__(self, out_size):
        super().__init__(out_size)
        self.module_class = LinearModule
        self.args = [out_size]





class MF(ModuleFactory):
    def __init__(self, module_class, *args):
        super().__init__()
        self.module_class = module_class
        self.args = args


class Seq(ModuleFactory):
    def __init__(self, inner_factory):
        super().__init__()
        self.inner_factory = inner_factory

    def size_change(self, in_size):
        return self.inner_factory.size_change(in_size)

    def __call__(self, in_size):
        return SequenceWrapperFull(self.inner_factory(in_size))


class Network(ModuleFactory):
    def __init__(self, *args):
        super().__init__()
        self.args = args
        self.module_class = NetworkModule


    def size_change(self, in_size):
        sizes = {'input': in_size}
        for path in self.args:
            sizes[path.name] = path.deduce_size(sizes)
        return sizes['output']

class ExecPath:
    def __init__(self, inputs, module_factories, result_name):
        self.name = result_name
        self.inputs = inputs
        self.strand = module_factories

    def deduce_size(self, other_sizes):
        if all([x in other_sizes for x in self.inputs]):
            cur_size = other_sizes[self.inputs[0]] if len(self.inputs) == 1 else torch.tensor([other_sizes[x] for x in self.inputs]).sum().item()
        else:
            cur_size = 0
        for factory in self.strand:
            #print(cur_size)
            cur_size = factory.size_change(cur_size)
        if not isinstance(cur_size, int):
            cur_size = flatten(cur_size)
        if cur_size <= 0:
            raise Exception('Size deduction failed.')
        return cur_size

class ConvPath:
    def __init__(self, module_factories):
        self.name = 'conv'
        self.inputs = ['input']
        self.strand = module_factories+[Flattener()]

    def deduce_size(self, other_sizes):
        cur_size = other_sizes['input']
        for factory in self.strand:
            #print(cur_size)
            cur_size = factory.size_change(cur_size)
        #cur_size = flatten(cur_size)
        if cur_size <= 0:
            raise Exception('Size deduction failed.')
        return cur_size


class Flattener(ModuleFactory):
    def __init__(self):
        super().__init__()
        self.module_class = FlattenerModule
        self.args = []

    def size_change(self, in_size):
        return flatten(in_size)




class Conv3(ModuleFactory):
    def __init__(self, conv_params, neuron_factory):
        super().__init__()
        self.conv_params = conv_params
        self.neuron_factory = neuron_factory

    def size_change(self, in_size):
        out_size, _ = ConvWrapper3d.calc_shape(in_size, self.conv_params)
        return out_size

    def __call__(self, in_size):
        conv = ConvWrapper3d(in_size, self.conv_params)
        return ConvModule(conv, self.neuron_factory(flatten(conv.out_shape)))


class Conv2(ModuleFactory):
    def __init__(self, conv_params, neuron_factory):
        super().__init__()
        self.conv_params = conv_params
        self.neuron_factory = neuron_factory

    def size_change(self, in_size):
        return ConvWrapper2.calc_shape(in_size, self.conv_params)

    def __call__(self, in_size):
        conv = ConvWrapper2(in_size, self.conv_params)
        return ConvModule(conv, self.neuron_factory(flatten(conv.out_shape)))


class LIF(ModuleFactory):
    def __init__(self, *args):
        super().__init__()
        self.module_class = LIFNeuron
        self.args = args


class Disc(ModuleFactory):
    def __init__(self, *args):
        super().__init__()
        self.module_class = DiscontinuousNeuron
        self.args = args

class PotOut(ModuleFactory):
    def __init__(self, *args):
        super().__init__()
        self.module_class = OutputNeuron
        self.args = args

class ReLU(ModuleFactory):
    def __init__(self):
        super().__init__()
        self.module_class = ReLuWrapper
        self.args = [None]

class LSTM(FixedSize):
    def __init__(self, size):
        super().__init__(size)

    def __call__(self, in_size):
        return SequenceWrapperFull(LSTMWrapperOneStep(in_size, self.out_size))