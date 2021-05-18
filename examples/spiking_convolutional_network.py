import rnnbuilder as rb
import torch
from rnnbuilder import snn, rnn, nn

seq = 8
batch = 32
inp_shape = (1,92,76)
example = torch.rand((seq,batch)+inp_shape)
FRAME_STACK = 4
N_OUT = 3

conv_neuron = snn.Discontinuous()

conv_stack = rb.Sequential(
    rnn.TempConvPlus2d(out_channels=32, kernel_size=8, stride=4, time_kernel_size=FRAME_STACK),conv_neuron,
    nn.Conv2d(out_channels=64, kernel_size=4, stride=2), conv_neuron,
    nn.Conv2d(out_channels=64, kernel_size=3, stride=1), conv_neuron)

ll = rb.Network()
ll.output = rb.Placeholder()
ll.output = ll.input.stack(ll.output).apply(nn.Linear(512), snn.LIF())

model = rb.Sequential(conv_stack, ll, nn.Linear(N_OUT)).make_model(inp_shape)