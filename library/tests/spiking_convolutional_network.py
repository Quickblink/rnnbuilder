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
snn.LIF()

conv_stack = rb.Sequential(
    rnn.TempConvPlus2d(out_channels=32, kernel_size=8, stride=4, time_kernel_size=FRAME_STACK),conv_neuron,
    nn.Conv2d(out_channels=64, kernel_size=4, stride=2), conv_neuron,
    nn.Conv2d(out_channels=64, kernel_size=3, stride=1), conv_neuron)

ll = rb.Network()
ll.output = rb.Placeholder()
ll.output = ll.input.stack(ll.output).apply(nn.Linear(512), snn.LIF())

model = rb.Sequential(conv_stack, ll, nn.Linear(N_OUT)).make_model(inp_shape)



from references.factories_old import Network, Linear, Seq, ExecPath, Conv3, Conv2, ConvPath, Disc, LIF, ReLU, LSTM

lif_config = {
    'SPIKE_FN' : 'bellec',
    'TAU': 5,
    '1-beta': False,
}

disc_config = {
    'SPIKE_FN' : 'bellec',
    'THRESHOLD' : 1
}

CONV_NEURON = Disc(disc_config) # Seq(LIF(lif_config)) # ReLU()
ll_rsnn = Seq(Network(ExecPath(['input', 'output'], [Linear(512), LIF(lif_config)], 'output')))
ll_snn = Network(ExecPath(['input'], [Linear(512), Seq(LIF(lif_config))], 'output'))
ll_lstm = LSTM(512)
ll_ffann = Network(ExecPath(['input'], [Linear(512), ReLU()], 'output'))
LAST_LAYER = ll_rsnn
conv = ConvPath([
    Conv3({'out_channels': 32, 'kernel_size': 8, 'stride': 4, 'frame_stack': FRAME_STACK}, CONV_NEURON),
    Conv2({'out_channels': 64, 'kernel_size': 4, 'stride': 2}, CONV_NEURON),
    Conv2({'out_channels': 64, 'kernel_size': 3, 'stride': 1}, CONV_NEURON)])
new_model_fac = Network(conv, ExecPath(['conv'], [LAST_LAYER, Linear(N_OUT)], 'output'))
make_model = lambda: new_model_fac(inp_shape)


ref_model = make_model()

# Transplant weights
with torch.no_grad():
    for i in range(len(list(model.named_parameters()))):
        par1 = list(model.named_parameters())[i][1]
        par1.data = list(ref_model.named_parameters())[i][1].data.view(par1.shape)



out, _ = model(example)

out_ref, _ = ref_model(example.refine_names('time', 'batch', 'C', 'H', 'W'), ref_model.get_initial_state(batch))



assert torch.isclose(out_ref.rename(None), out).all()

print('SNN test finished.')