from torch import nn
from types import SimpleNamespace

class CNN(nn.Module):

    def __init__(self,
                 c_hidden : list,
                 dilation : int,
                 padding : int,
                 stride : int, 
                 kernel_size : int,
                 downsampling_kernel_size = None):
        super().__init__()
        self.hparams = SimpleNamespace(c_hidden=c_hidden,
                                       downsampling_kernel_size=downsampling_kernel_size,
                                       kernel_size=kernel_size,
                                       dilation=dilation,
                                       padding=padding,
                                       stride=stride)

        self._create_network()

    def _create_network(self):

        c_in = 1
        blocks = []
        for c_hid in self.hparams.c_hidden:
            blocks.append(
                nn.Conv1d(c_in, c_hid, kernel_size=self.hparams.kernel_size, dilation=self.hparams.dilation, padding=self.hparams.padding, padding_mode='circular', stride=self.hparams.stride),
            )
            blocks.append(
                nn.AvgPool1d(kernel_size=2, stride=2)
            )
            blocks.append(
                nn.GELU(),
            )
            c_in = c_hid

        self.net = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.Conv1d(self.hparams.c_hidden[-1], 1, kernel_size=1, dilation=1, padding=0, stride=1),
            nn.AvgPool1d(kernel_size=self.hparams.downsampling_kernel_size, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.output_net(x)
        return x