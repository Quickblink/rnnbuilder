import sys
sys.path.append('library/src')
import rnnbuilder
import rnnbuilder.nn as rnn
from rnnbuilder.custom import register_non_recurrent
from torch import nn


class ResidualBlock(rnnbuilder.Network):
    def __init__(self, channels, stride):
        super().__init__()
        self.main_path = rnnbuilder.Layer(
            self.input,
            rnnbuilder.Sequential(
                rnn.Conv2d(out_channels=channels, kernel_size=3, stride=stride, padding=1),
                rnn.ReLU(),
                rnn.Conv2d(out_channels=channels, kernel_size=3, stride=1, padding=1),
                rnn.ReLU()
            )
        )
        if stride > 1:
            self.res_path = rnnbuilder.Layer(
                self.input,
                rnnbuilder.Sequential(
                    rnn.Conv2d(out_channels=channels, kernel_size=1, stride=stride),
                    rnn.ReLU(),
                )
            )
        self.output = rnnbuilder.Sum(self.main_path, self.res_path if stride > 1 else self.input)


MaxPool = register_non_recurrent(module_class=nn.MaxPool2d, flatten_input=False, shape_change=True)
AdaptiveAvgPool2d = register_non_recurrent(module_class=nn.AdaptiveAvgPool2d, flatten_input=False, shape_change=True)

if __name__ == '__main__':
    resnet18 = rnnbuilder.Sequential(
        rnn.Conv2d(64, kernel_size=7, stride=2, padding=3),
        rnn.ReLU(),
        MaxPool(kernel_size=3, stride=2, padding=1),

        ResidualBlock(64, 1),
        ResidualBlock(64, 1),

        ResidualBlock(128, 2),
        ResidualBlock(128, 1),

        ResidualBlock(256, 2),
        ResidualBlock(256, 1),

        ResidualBlock(512, 2),
        ResidualBlock(512, 1),

        AdaptiveAvgPool2d((1, 1)),
        rnn.Linear(1000)
    )

    r = resnet18.make_model((3, 224, 224))
    print(r)
