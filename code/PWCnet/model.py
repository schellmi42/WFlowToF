'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Portions of this code copyright (c) 2022 Visual Computing group
                of Ulm University, Germany. See the LICENSE file at the
                top-level directory of this distribution.
    Portions of this code copyright 2017, Clement Pinard
    (to large parts adapted from pwcnet implementation in ptlflow)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from ptlflow.models.pwcnet import pwcnet as PWC

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomPWCNet(nn.Module):
  """ adapoted Pyramid Warping Cost volume (PWC) Network
  Deqing Sun, Xiaodong Yang, Ming-Yu Liu, and Jan Kautz.
  Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume.
  In Proceedings of the IEEE conference on computer vision and pattern recognition 2018
  """
  def __init__(self,
               in_channels=1,
               div_flow=20.0,
               md=4,
               time_steps=4,
               taps=1,
               norm=None):
    super(CustomPWCNet, self).__init__()
    self.time_steps = time_steps
    self.taps = taps
    self.in_channels = in_channels

    self.div_flow = div_flow
    self.md = md

    if norm == 'instance':
      self.norm = nn.InstanceNorm2d(4)
    else:
      self.norm = None

    # same as in ptlflow except for input dimensions and removed args dependency
    self.conv1a  = PWC.conv(in_channels, 16, kernel_size=3, stride=2)
    self.conv1aa = PWC.conv(16, 16, kernel_size=3, stride=1)
    self.conv1b  = PWC.conv(16, 16, kernel_size=3, stride=1)
    self.conv2a  = PWC.conv(16, 32, kernel_size=3, stride=2)
    self.conv2aa = PWC.conv(32, 32, kernel_size=3, stride=1)
    self.conv2b  = PWC.conv(32, 32, kernel_size=3, stride=1)
    self.conv3a  = PWC.conv(32, 64, kernel_size=3, stride=2)
    self.conv3aa = PWC.conv(64, 64, kernel_size=3, stride=1)
    self.conv3b  = PWC.conv(64, 64, kernel_size=3, stride=1)
    self.conv4a  = PWC.conv(64, 96, kernel_size=3, stride=2)
    self.conv4aa = PWC.conv(96, 96, kernel_size=3, stride=1)
    self.conv4b  = PWC.conv(96, 96, kernel_size=3, stride=1)
    self.conv5a  = PWC.conv(96, 128, kernel_size=3, stride=2)
    self.conv5aa = PWC.conv(128, 128, kernel_size=3, stride=1)
    self.conv5b  = PWC.conv(128, 128, kernel_size=3, stride=1)
    self.conv6aa = PWC.conv(128, 196, kernel_size=3, stride=2)
    self.conv6a  = PWC.conv(196, 196, kernel_size=3, stride=1)
    self.conv6b  = PWC.conv(196, 196, kernel_size=3, stride=1)

    self.leakyRELU = nn.LeakyReLU(0.1)

    self.corr = PWC.SpatialCorrelationSampler(kernel_size=1, patch_size=2 * self.md + 1, padding=0)

    nd = (2 * self.md + 1) ** 2
    dd = np.cumsum([128, 128, 96, 64, 32])

    od = nd
    self.conv6_0 = PWC.conv(od, 128, kernel_size=3, stride=1)
    self.conv6_1 = PWC.conv(od + dd[0], 128, kernel_size=3, stride=1)
    self.conv6_2 = PWC.conv(od + dd[1], 96, kernel_size=3, stride=1)
    self.conv6_3 = PWC.conv(od + dd[2], 64, kernel_size=3, stride=1)
    self.conv6_4 = PWC.conv(od + dd[3], 32, kernel_size=3, stride=1)
    self.predict_flow6 = PWC.predict_flow(od + dd[4])
    self.deconv6 = PWC.deconv(2, 2, kernel_size=4, stride=2, padding=1)
    self.upfeat6 = PWC.deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

    od = nd + 128 + 4
    self.conv5_0 = PWC.conv(od, 128, kernel_size=3, stride=1)
    self.conv5_1 = PWC.conv(od + dd[0], 128, kernel_size=3, stride=1)
    self.conv5_2 = PWC.conv(od + dd[1], 96, kernel_size=3, stride=1)
    self.conv5_3 = PWC.conv(od + dd[2], 64, kernel_size=3, stride=1)
    self.conv5_4 = PWC.conv(od + dd[3], 32, kernel_size=3, stride=1)
    self.predict_flow5 = PWC.predict_flow(od + dd[4])
    self.deconv5 = PWC.deconv(2, 2, kernel_size=4, stride=2, padding=1)
    self.upfeat5 = PWC.deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

    od = nd + 96 + 4
    self.conv4_0 = PWC.conv(od, 128, kernel_size=3, stride=1)
    self.conv4_1 = PWC.conv(od + dd[0], 128, kernel_size=3, stride=1)
    self.conv4_2 = PWC.conv(od + dd[1], 96, kernel_size=3, stride=1)
    self.conv4_3 = PWC.conv(od + dd[2], 64, kernel_size=3, stride=1)
    self.conv4_4 = PWC.conv(od + dd[3], 32, kernel_size=3, stride=1)
    self.predict_flow4 = PWC.predict_flow(od + dd[4])
    self.deconv4 = PWC.deconv(2, 2, kernel_size=4, stride=2, padding=1)
    self.upfeat4 = PWC.deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

    od = nd + 64 + 4
    self.conv3_0 = PWC.conv(od, 128, kernel_size=3, stride=1)
    self.conv3_1 = PWC.conv(od + dd[0], 128, kernel_size=3, stride=1)
    self.conv3_2 = PWC.conv(od + dd[1], 96, kernel_size=3, stride=1)
    self.conv3_3 = PWC.conv(od + dd[2], 64, kernel_size=3, stride=1)
    self.conv3_4 = PWC.conv(od + dd[3], 32, kernel_size=3, stride=1)
    self.predict_flow3 = PWC.predict_flow(od + dd[4])
    self.deconv3 = PWC.deconv(2, 2, kernel_size=4, stride=2, padding=1)
    self.upfeat3 = PWC.deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

    od = nd + 32 + 4
    self.conv2_0 = PWC.conv(od, 128, kernel_size=3, stride=1)
    self.conv2_1 = PWC.conv(od + dd[0], 128, kernel_size=3, stride=1)
    self.conv2_2 = PWC.conv(od + dd[1], 96, kernel_size=3, stride=1)
    self.conv2_3 = PWC.conv(od + dd[2], 64, kernel_size=3, stride=1)
    self.conv2_4 = PWC.conv(od + dd[3], 32, kernel_size=3, stride=1)
    self.predict_flow2 = PWC.predict_flow(od + dd[4])

    for m in self.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
            if m.bias is not None:
                m.bias.data.zero_()
    self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

  def twotap_reordering(self, inputs):
    if self.time_steps == 2:
      inputs = torch.cat([inputs[:, [0, 2]], inputs[:, [1, 3]]], dim=1)
    elif self.time_steps == 6:
      inputs = torch.cat([inputs[:, [0, 2]], inputs[:, [1, 3]],
                          inputs[:, [4, 6]], inputs[:, [5, 7]],
                          inputs[:, [8, 10]], inputs[:, [9, 11]]], dim=1)
    return inputs

  def compute_intensity_on_taps(self, inputs):
    B, C, H, W = inputs.shape
    inputs = inputs.view(B, int(self.time_steps), int(self.taps), H, W)
    inputs = inputs.mean(dim=2)
    return inputs

  def forward(self, inputs):
    if self.taps == 2:
      inputs = self.twotap_reordering(inputs)

    if self.taps != 1 and self.in_channels == 1:
      # Lindner method
      inputs = self.compute_intensity_on_taps(inputs)

    B, C, H, W = inputs.shape

    if self.norm is not None:
      inputs = self.norm(inputs)
    # reshape into timesteps x taps
    inputs = inputs.view(B, int(self.time_steps), -1, H, W)

    output = {'flows': []}
    if self.training:
      output['flow_preds'] = []

    # execute encoder network for all inputs
    encodings = []
    for c in range(C):
      encodings.append(self.encode_single(inputs[:, c]))
    # get latent vectors at level 3
    output['latent'] = [encodings[c][2] for c in range(C)]

    # predit flows
    for i in range(self.time_steps - 1):
      curr_output = self.pred_flow(encodings[i], encodings[-1])
      output['flows'].append(curr_output['flows'])
      if self.training:
        output['flow_preds'].append(curr_output['flow_preds'])

    # transpose lists
    if self.training:
      output['flow_preds'] = [list(x) for x in list(zip(*output['flow_preds']))]

    return output

  def encode_single(self, im1):
    c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
    c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
    c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
    c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
    c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
    c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))

    return [c11, c12, c13, c14, c15, c16]

  def encode(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)
    B, C, H, W = inputs.shape

    if self.norm is not None:
      inputs = self.norm(inputs)
    # reshape into timesteps x taps
    inputs = inputs.view(B, int(self.time_steps), -1, H, W)

    encodings = []
    for c in range(C):
      encodings.append(self.encode_single(inputs[:, c]))
    # get latent vectors at level 3
    output = {}

    if self.training:
      output['flow_preds'] = None
      output['latent'] = [encodings[c][2] for c in range(C)]

    output['flows'] = None
    return output

  def pred_flow(self, c1, c2):

    c11, c12, c13, c14, c15, c16 = c1
    c21, c22, c23, c24, c25, c26 = c2

    corr6 = self.corr(c16, c26)
    corr6 = corr6.view(corr6.shape[0], -1, corr6.shape[3], corr6.shape[4])
    corr6 = corr6 / c16.shape[1]
    corr6 = self.leakyRELU(corr6)

    x = torch.cat((self.conv6_0(corr6), corr6), 1)
    x = torch.cat((self.conv6_1(x), x), 1)
    x = torch.cat((self.conv6_2(x), x), 1)
    x = torch.cat((self.conv6_3(x), x), 1)
    x = torch.cat((self.conv6_4(x), x), 1)
    flow6 = self.predict_flow6(x)
    up_flow6 = self.deconv6(flow6)
    up_feat6 = self.upfeat6(x)

    warp5 = self.warp(c25, up_flow6 * 0.625)
    corr5 = self.corr(c15, warp5)
    corr5 = corr5.view(corr5.shape[0], -1, corr5.shape[3], corr5.shape[4])
    corr5 = corr5 / c15.shape[1]
    corr5 = self.leakyRELU(corr5)
    x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
    x = torch.cat((self.conv5_0(x), x), 1)
    x = torch.cat((self.conv5_1(x), x), 1)
    x = torch.cat((self.conv5_2(x), x), 1)
    x = torch.cat((self.conv5_3(x), x), 1)
    x = torch.cat((self.conv5_4(x), x), 1)
    flow5 = self.predict_flow5(x)
    up_flow5 = self.deconv5(flow5)
    up_feat5 = self.upfeat5(x)

    warp4 = self.warp(c24, up_flow5 * 1.25)
    corr4 = self.corr(c14, warp4)
    corr4 = corr4.view(corr4.shape[0], -1, corr4.shape[3], corr4.shape[4])
    corr4 = corr4 / c14.shape[1]
    corr4 = self.leakyRELU(corr4)
    x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
    x = torch.cat((self.conv4_0(x), x), 1)
    x = torch.cat((self.conv4_1(x), x), 1)
    x = torch.cat((self.conv4_2(x), x), 1)
    x = torch.cat((self.conv4_3(x), x), 1)
    x = torch.cat((self.conv4_4(x), x), 1)
    flow4 = self.predict_flow4(x)
    up_flow4 = self.deconv4(flow4)
    up_feat4 = self.upfeat4(x)

    warp3 = self.warp(c23, up_flow4 * 2.5)
    corr3 = self.corr(c13, warp3)
    corr3 = corr3.view(corr3.shape[0], -1, corr3.shape[3], corr3.shape[4])
    corr3 = corr3 / c13.shape[1]
    corr3 = self.leakyRELU(corr3)

    x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
    x = torch.cat((self.conv3_0(x), x), 1)
    x = torch.cat((self.conv3_1(x), x), 1)
    x = torch.cat((self.conv3_2(x), x), 1)
    x = torch.cat((self.conv3_3(x), x), 1)
    x = torch.cat((self.conv3_4(x), x), 1)
    flow3 = self.predict_flow3(x)
    up_flow3 = self.deconv3(flow3)
    up_feat3 = self.upfeat3(x)

    warp2 = self.warp(c22, up_flow3 * 5.0)
    corr2 = self.corr(c12, warp2)
    corr2 = corr2.view(corr2.shape[0], -1, corr2.shape[3], corr2.shape[4])
    corr2 = corr2 / c12.shape[1]
    corr2 = self.leakyRELU(corr2)
    x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
    x = torch.cat((self.conv2_0(x), x), 1)
    x = torch.cat((self.conv2_1(x), x), 1)
    x = torch.cat((self.conv2_2(x), x), 1)
    x = torch.cat((self.conv2_3(x), x), 1)
    x = torch.cat((self.conv2_4(x), x), 1)
    flow2 = self.predict_flow2(x)

    flow_up = self.upsample1(flow2 * self.div_flow)

    outputs = {}
    if self.training:
        outputs['flow_preds'] = [flow2, flow3, flow4, flow5, flow6]
        outputs['flows'] = flow_up
    else:
        outputs['flows'] = flow_up
    return outputs

  def warp(self, x, flo):
      """
      warp an image/tensor (im2) back to im1, according to the optical flow
      x: [B, C, H, W] (im2)
      flo: [B, 2, H, W] flow
      """
      B, C, H, W = x.size()
      # mesh grid
      xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
      yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
      xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
      yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
      grid = torch.cat((xx, yy), 1).float()

      if x.is_cuda:
          grid = grid.to(dtype=x.dtype, device=x.device)
      vgrid = grid + flo

      # scale grid to [-1,1]
      vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
      vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

      vgrid = vgrid.permute(0, 2, 3, 1)
      output = nn.functional.grid_sample(x, vgrid, align_corners=True)
      mask = torch.ones(x.size()).to(dtype=x.dtype, device=x.device)
      mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

      mask[mask < 0.9999] = 0
      mask[mask > 0] = 1

      return output * mask
