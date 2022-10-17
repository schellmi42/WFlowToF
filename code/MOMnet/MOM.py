'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
from torch import nn


class MOMNet(nn.Module):
  """ adapted MOtion Module (MOM)
  Qi Guo, Iuri Frosio, Orazio Gallo, Todd Zickler, and Jan Kautz.
  Tackling 3D ToF artifacts through learning and the FLAT dataset.
  In Proceedings of the European Conference on Computer Vision (ECCV) 2018
  """
  def __init__(self, input_dims=9, norm=None, div_flow=1):
    super(MOMNet, self).__init__()

    if norm == 'instance':
      self.norm = nn.InstanceNorm2d(input_dims)
    else:
      self.norm = None
    self.div_flow = div_flow
    # U-Net
    self.n_channels = [input_dims, 32, 32, 64, 64, 128, 128, 256, 256, 256, 256, 512]
    self.kernel_sizes = [7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3]
    pool_size = 2
    pool_stride = 2
    self.pool_bool = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    self.skips = [False, True, False, True, False, True, False, True, False, True, False]

    self.num_layers = len(self.n_channels) - 1
    self.convs = nn.ModuleList()
    for l in range(self.num_layers):
      self.convs.append(nn.Conv2d(in_channels=self.n_channels[l], out_channels=self.n_channels[l + 1], kernel_size=self.kernel_sizes[l], padding=(self.kernel_sizes[l] - 1) // 2))
    self.pool = nn.MaxPool2d(pool_size, pool_stride)

    self.convs_up = nn.ModuleList()
    for l in range(self.num_layers):
      if self.pool_bool[-l - 1]:
        if l != self.num_layers - 1:
          self.convs_up.append(nn.ConvTranspose2d(in_channels=self.n_channels[-l - 1], out_channels=self.n_channels[-l - 2], kernel_size=self.kernel_sizes[-l - 1], stride=pool_stride, padding=(self.kernel_sizes[-l - 1] - 1) // 2, output_padding=1))
        else:
          self.convs_up.append(nn.ConvTranspose2d(in_channels=self.n_channels[-l - 1], out_channels=2 * self.n_channels[-l - 2], kernel_size=self.kernel_sizes[-l - 1], stride=pool_stride, padding=(self.kernel_sizes[-l - 1] - 1) // 2, output_padding=1))
      else:
        if self.skips[-l - 1]:
          self.convs_up.append(nn.Conv2d(in_channels=2 * self.n_channels[-l - 1], out_channels=self.n_channels[-l - 2], kernel_size=self.kernel_sizes[-l - 1], padding=(self.kernel_sizes[-l - 1] - 1) // 2))
        else:
          self.convs_up.append(nn.Conv2d(in_channels=self.n_channels[-l - 1], out_channels=self.n_channels[-l - 2], kernel_size=self.kernel_sizes[-l - 1], padding=(self.kernel_sizes[-l - 1] - 1) // 2))

    self.lrelu = nn.LeakyReLU()

    # Optical Flow head
    self. n_channels_mix = 2 * input_dims
    self.kernel_sizes_mix = 3

    self.head_convs = nn.ModuleList()
    for i in range(3):
      self.head_convs.append(nn.Conv2d(in_channels=self.n_channels_mix, out_channels=self.n_channels_mix, kernel_size=self.kernel_sizes_mix, padding=(self.kernel_sizes_mix - 1) // 2))

  def forward(self, x):

    if self.norm is not None:
      x = self.norm(x)

    # U-Net
    skip = []
    for l in range(self.num_layers):
      x = self.convs[l](x)
      x = self.lrelu(x)
      if self.pool_bool[l]:
        x = self.pool(x)
      if self.skips:
        skip.append(x)
      else:
        skip.append([])

    for l in range(self.num_layers):
      if self.skips[-l - 1]:
        x = torch.cat((x, skip[-l - 1]), dim=1)
      x = self.convs_up[l](x)
      x = self.lrelu(x)

    # Optical Flow head
    for l in range(3):
      x = self.head_convs[l](x)
      if l != 2:
        x = self.lrelu(x)
    outputs = {}
    outputs['flows'] = x * self.div_flow
    return outputs
