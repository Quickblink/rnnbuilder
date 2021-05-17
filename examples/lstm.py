import torch
import rnnbuilder as rb
from rnnbuilder.nn import ReLU, Conv2d, Linear, Tanh, Sigmoid


from functools import reduce
import operator
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class HadamardModule(rb.custom.CustomModule):
    def get_out_shape(self, in_shapes):
        return in_shapes[0]

    def forward(self, inputs, _):
        return prod(inputs), ()

Hadamard = custom.register_recurrent(module_class=HadamardModule, flatten_input=False, single_step=False, unroll_full_state=False)


seq = 8
batch = 32
inp_shape = (100,)
example = torch.rand((seq,batch)+inp_shape)


hidden_size = 32
n = rb.Network()

n.output = rb.Placeholder()
n.h_and_i = n.input.stack(n.output)

n.i = n.h_and_i.apply(Linear(hidden_size), Sigmoid())
n.f = n.h_and_i.apply(Linear(hidden_size), Sigmoid())
n.o = n.h_and_i.apply(Linear(hidden_size), Sigmoid())
n.g = n.h_and_i.apply(Linear(hidden_size), Tanh())

n.c = rb.Placeholder()
n.c_1 = n.f.append(n.c).apply(Hadamard())
n.c_2 = n.i.append(n.g).apply(Hadamard())

n.c = n.c_1.sum(n.c_2)
n.tan_c = n.c.apply(Tanh())
n.output = n.o.append(n.tan_c).apply(Hadamard())


model = n.make_model(inp_shape)
