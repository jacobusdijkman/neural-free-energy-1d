from torch import nn
from types import SimpleNamespace

class CNN(nn.Module):
    
    """

      Changing the dz also means you need to change the nr of layers / output downsampling_kernel_size. 
    
    """

    def __init__(self,
                 c_hidden : list,
                 dilation : int,
                 padding : int,
                 stride : int, 
                 kernel_size : int):
        super().__init__()
        self.hparams = SimpleNamespace(c_hidden=c_hidden,
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
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.output_net(x)
        return x