'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
from torch import nn
import torch.nn.functional as F


class CoarseFineCNN(nn.Module):
  ''' Coarse Fine Network from Agresti et al.

  Args:
    batch_size: `int`.
  '''
  def __init__(self,
               input_feature_size=5,
               feature_size_coarse=32,
               feature_size_fine=64,
               num_levels_coarse=5,
               num_levels_fine=5,
               num_output_features=1,
               resolution=[128, 128],
               batch_size=None):
    super(CoarseFineCNN, self).__init__()
    self.batch_size = batch_size
    self.feature_sizes = [input_feature_size, feature_size_coarse, feature_size_fine]
    self.num_levels = [num_levels_coarse, num_levels_fine]
    self.resolution = resolution

    # coarse network ###
    self.coarse_convs = nn.ModuleList()
    self.coarse_pools = nn.ModuleList()
    self.coarse_ReLUs = nn.ModuleList()

    # 2 blocks with pooling
    self.coarse_convs.append(
      nn.Conv2d(
        in_channels=input_feature_size,
        out_channels=feature_size_coarse,
        kernel_size=[3, 3],
        padding='same')
    )
    self.coarse_ReLUs.append(nn.ReLU())
    self.coarse_pools.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
    self.coarse_convs.append(
      nn.Conv2d(
        in_channels=feature_size_coarse,
        out_channels=feature_size_coarse,
        kernel_size=[3, 3],
        padding='same')
    )
    self.coarse_ReLUs.append(nn.ReLU())
    self.coarse_pools.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))

    # 2 blocks without pooling
    for i in range(2):
      self.coarse_convs.append(
        nn.Conv2d(
          in_channels=feature_size_coarse,
          out_channels=feature_size_coarse,
          kernel_size=[3, 3],
          padding='same')
      )
      self.coarse_ReLUs.append(nn.ReLU())

    # 1 block without activation for coarse depth
    self.coarse_convs.append(
      nn.Conv2d(
        in_channels=feature_size_coarse,
        out_channels=num_output_features,
        kernel_size=[3, 3],
        padding='same')
    )

    # fine network ###
    self.fine_convs = nn.ModuleList()
    self.fine_ReLUs = nn.ModuleList()

    # 3 blocks with activations
    self.fine_convs.append(
      nn.Conv2d(
                in_channels=input_feature_size,
                out_channels=feature_size_fine,
                kernel_size=[3, 3],
                padding='same')
    )
    self.fine_ReLUs.append(nn.ReLU())
    for level in range(2):
      self.fine_convs.append(
        nn.Conv2d(
                  in_channels=feature_size_fine,
                  out_channels=feature_size_fine,
                  kernel_size=[3, 3],
                  padding='same')
      )
      self.fine_ReLUs.append(nn.ReLU())

    # 1 block with additional input from coarse network
    self.fine_convs.append(
        nn.Conv2d(
                  in_channels=feature_size_fine + num_output_features,
                  out_channels=feature_size_fine,
                  kernel_size=[3, 3],
                  padding='same')
    )
    self.fine_ReLUs.append(nn.ReLU())

    # 1 block without activation for fine depth
    self.fine_convs.append(
        nn.Conv2d(
                  in_channels=feature_size_fine,
                  out_channels=num_output_features,
                  kernel_size=[3, 3],
                  padding='same')
    )

  def forward(self,
              features,
              input_depths):
    ''' Evaluates network.

    Args:
      features: Input features [N, C, H, W]
      depths: Input depths to denoise [N, C_out, H, W]
    '''
    features_coarse = features
    # coarse network
    for level in range(4):
      features_coarse = self.coarse_convs[level](features_coarse)
      features_coarse = self.coarse_ReLUs[level](features_coarse)
      if level < 2:
        features_coarse = self.coarse_pools[level](features_coarse)

    features_coarse = self.coarse_convs[-1](features_coarse)
    depth_coarse = F.interpolate(features_coarse, size=input_depths.shape[-2:], mode='bilinear', align_corners=False)
    # skip connection
    depth_coarse = depth_coarse + input_depths

    features_fine = features
    for level in range(4):
      if level == 3:
        features_fine = torch.cat((features_fine, depth_coarse), dim=1)
      features_fine = self.fine_convs[level](features_fine)
      features_fine = self.fine_ReLUs[level](features_fine)

    features_fine = self.fine_convs[-1](features_fine)
    depths_fine = features_fine + input_depths

    return depths_fine, depth_coarse
