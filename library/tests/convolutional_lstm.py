import rnnbuilder as rb
import torch
from rnnbuilder import snn, rnn, nn

seq = 8
batch = 32
inp_shape = (1,92,76)
example = torch.rand((seq,batch)+inp_shape, requires_grad=True)
FRAME_STACK = 4
N_OUT = 3

conv_neuron = nn.ReLU()

conv_stack = rb.Sequential(
    rnn.TempConvPlus2d(out_channels=32, kernel_size=8, stride=4, time_kernel_size=FRAME_STACK),conv_neuron,
    nn.Conv2d(out_channels=64, kernel_size=4, stride=2), conv_neuron,
    nn.Conv2d(out_channels=64, kernel_size=3, stride=1), conv_neuron)

ll = rb.rnn.LSTM(512)

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

CONV_NEURON = ReLU() # Seq(LIF(lif_config)) # ReLU()
ll_rsnn = Seq(Network(ExecPath(['input', 'output'], [Linear(512), LIF(lif_config)], 'output')))
ll_snn = Network(ExecPath(['input'], [Linear(512), Seq(LIF(lif_config))], 'output'))
ll_lstm = LSTM(512)
ll_ffann = Network(ExecPath(['input'], [Linear(512), ReLU()], 'output'))
LAST_LAYER = ll_lstm
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



out1, state1 = model(example)



out1_ref, state1_ref = ref_model(example.refine_names('time', 'batch', 'C', 'H', 'W'), ref_model.get_initial_state(batch))

assert torch.isclose(out1_ref.rename(None), out1).all()


out2, state2 = model(example, state1)

out2_ref, state2_ref = ref_model(example.refine_names('time', 'batch', 'C', 'H', 'W'), state1_ref)

assert torch.isclose(out2_ref.rename(None), out2).all()


model.configure(full_state=True)

for m in ref_model.modules():
        m.full_state = True

out3, state3 = model(example, state2)

out3_ref, state3_ref = ref_model(example.refine_names('time', 'batch', 'C', 'H', 'W'), state2_ref)

assert torch.isclose(out3_ref.rename(None), out3).all()


state_con1 = rb.base._utils.StateContainerNew(state3)
l1 = []
for state, _ in state_con1.transfer(state3):
    l1.append(state)

state_con2 = rb.base._utils.StateContainerNew(state3_ref)
l2 = []
for state, _ in state_con2.transfer(state3_ref):
    l2.append(state)

assert len(l1) == len(l2)

for i in range(len(l1)):
    assert torch.isclose(l2[i].rename(None), l1[i]).all()

out3.sum().backward()
grad = example.grad.clone()
example.grad = None
out3_ref.sum().backward()
grad_ref = example.grad.clone()

assert torch.isclose(grad, grad_ref).all()

print('Conv lstm test finished.')