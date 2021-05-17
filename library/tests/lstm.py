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

Hadamard = rb.custom.register_recurrent(module_class=HadamardModule, flatten_input=False, single_step=False, unroll_full_state=False)


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


ref_model = torch.nn.LSTM(inp_shape[0], hidden_size)


with torch.no_grad():
    model.inner.layers.c0.layers.i.mlist[0].inner.weight.data,\
    model.inner.layers.c0.layers.f.mlist[0].inner.weight.data,\
    model.inner.layers.c0.layers.g.mlist[0].inner.weight.data,\
    model.inner.layers.c0.layers.o.mlist[0].inner.weight.data = torch.cat((ref_model.weight_hh_l0, ref_model.weight_ih_l0), dim=-1).chunk(4)

    model.inner.layers.c0.layers.i.mlist[0].inner.bias.data,\
    model.inner.layers.c0.layers.f.mlist[0].inner.bias.data,\
    model.inner.layers.c0.layers.g.mlist[0].inner.bias.data,\
    model.inner.layers.c0.layers.o.mlist[0].inner.bias.data = (ref_model.bias_ih_l0 + ref_model.bias_hh_l0).chunk(4)


out, _ = model(example)


out_ref, _ = ref_model(example)

if not torch.isclose(out, out_ref, atol=1e-7).all():
    out, _ = model(example)

print((out-out_ref).max(), (out-out_ref).var().sqrt())
print((out-out_ref))
print(out)
assert torch.isclose(out, out_ref, atol=1e-7).all()

print('LSTM test finished.')