'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Portions of this code copyright (c) 2022 Visual Computing group
                of Ulm University, Germany. See the LICENSE file at the
                top-level directory of this distribution.
    Portions of this code copyright 2017, Clement Pinard
    (to large parts adapted from fastflownet implementation in ptlflow)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from ptlflow.models.fastflownet import fastflownet as FFN

import torch
import torch.nn as nn
import torch.nn.functional as F


def centralize(imgs):
  """
  Args:
    imgs: `list` of [B, C, H , W]
  Returns:
    `list` of [B, C, H , W]
    + means
  """
  B, C, H, W = imgs[0].shape
  mean = torch.cat(imgs, dim=2).view(B, C, -1).mean(2).view(B, C, 1, 1)
  return [img - mean for img in imgs] + [mean]


class CustomFastFlowNet(nn.Module):
  """ adapted Fast Flow Net (FFN)
  Lingtong Kong, Chunhua Shen, and Jie Yang.
  Fastflownet: A lightweight network for fast optical flow estimation.
  In 2021 IEEE International Conference on Robotics and Automation (ICRA)
  """
  def __init__(self,
               in_channels=1,
               norm=None,
               md=4,
               groups=3,
               div_flow=20.0,
               time_steps=4,
               taps=1):
    super(CustomFastFlowNet, self).__init__()
    self.md = md
    self.groups = groups
    self.div_flow = div_flow
    self.time_steps = time_steps
    self.C = in_channels
    self.taps = taps
    self.in_channels = in_channels

    if norm == 'instance':
      self.norm = nn.InstanceNorm2d(4)
    else:
      self.norm = None

    self.pconv1_1 = FFN.convrelu(in_channels, 16, 3, 2)
    self.pconv1_2 = FFN.convrelu(16, 16, 3, 1)
    self.pconv2_1 = FFN.convrelu(16, 32, 3, 2)
    self.pconv2_2 = FFN.convrelu(32, 32, 3, 1)
    self.pconv2_3 = FFN.convrelu(32, 32, 3, 1)
    self.pconv3_1 = FFN.convrelu(32, 64, 3, 2)
    self.pconv3_2 = FFN.convrelu(64, 64, 3, 1)
    self.pconv3_3 = FFN.convrelu(64, 64, 3, 1)

    self.corr_layer = FFN.SpatialCorrelationSampler(kernel_size=1, patch_size=2 * self.md + 1, padding=0)
    self.index = torch.tensor(
        [0, 2, 4, 6, 8,
         10, 12, 14, 16,
         18, 20, 21, 22, 23, 24, 26,
         28, 29, 30, 31, 32, 33, 34,
         36, 38, 39, 40, 41, 42, 44,
         46, 47, 48, 49, 50, 51, 52,
         54, 56, 57, 58, 59, 60, 62,
         64, 66, 68, 70,
         72, 74, 76, 78, 80])

    self.rconv2 = FFN.convrelu(32, 32, 3, 1)
    self.rconv3 = FFN.convrelu(64, 32, 3, 1)
    self.rconv4 = FFN.convrelu(64, 32, 3, 1)
    self.rconv5 = FFN.convrelu(64, 32, 3, 1)
    self.rconv6 = FFN.convrelu(64, 32, 3, 1)

    self.up3 = FFN.deconv(2, 2)
    self.up4 = FFN.deconv(2, 2)
    self.up5 = FFN.deconv(2, 2)
    self.up6 = FFN.deconv(2, 2)

    self.decoder2 = FFN.Decoder(87, self.groups)
    self.decoder3 = FFN.Decoder(87, self.groups)
    self.decoder4 = FFN.Decoder(87, self.groups)
    self.decoder5 = FFN.Decoder(87, self.groups)
    self.decoder6 = FFN.Decoder(87, self.groups)

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def corr(self, f1, f2):
    corr = self.corr_layer(f1, f2)
    corr = corr.view(corr.shape[0], -1, corr.shape[3], corr.shape[4])
    corr = corr / f1.shape[1]
    return corr

  def warp(self, x, flo):
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat([xx, yy], 1).to(x)
    vgrid = grid + flo
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
    return output

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

    if self.time_steps == 2:
      return self.forward_two_flows(inputs)
    elif self.time_steps == 3:
      return self.forward_three_flows(inputs)
    elif self.time_steps == 4:
      return self.forward_four_flows(inputs)
    elif self.time_steps == 6:
      return self.forward_six_flows(inputs)
    elif self.time_steps == 12:
      return self.forward_twelve_flows(inputs)
    else:
      raise ValueError('unsupported number of time_steps: ' + str(self.time_steps))

  def forward_two_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:]

    img1, img2, _ = centralize([img1, img2])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
    f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
    f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
    f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
    f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
    f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

    flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
    cv16 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
    r46 = self.rconv6(f26)
    cat16 = torch.cat([cv16, r46, flow7_up], 1)
    flow16 = self.decoder6(cat16)

    flow16_up = self.up6(flow16)
    f15_w = self.warp(f15, flow16_up * 0.625)
    cv15 = torch.index_select(self.corr(f15_w, f25), dim=1, index=self.index.to(f15).long())
    r45 = self.rconv5(f25)
    cat15 = torch.cat([cv15, r45, flow16_up], 1)
    flow15 = self.decoder5(cat15) + flow16_up

    flow15_up = self.up5(flow15)
    f14_w = self.warp(f14, flow15_up * 1.25)
    cv14 = torch.index_select(self.corr(f14_w, f24), dim=1, index=self.index.to(f14).long())
    r44 = self.rconv4(f24)
    cat14 = torch.cat([cv14, r44, flow15_up], 1)
    flow14 = self.decoder4(cat14) + flow15_up

    flow14_up = self.up4(flow14)
    f13_w = self.warp(f13, flow14_up * 2.5)
    cv13 = torch.index_select(self.corr(f13_w, f23), dim=1, index=self.index.to(f13).long())
    r43 = self.rconv3(f23)
    cat13 = torch.cat([cv13, r43, flow14_up], 1)
    flow13 = self.decoder3(cat13) + flow14_up

    flow13_up = self.up3(flow13)
    f12_w = self.warp(f12, flow13_up * 5.0)
    cv12 = torch.index_select(self.corr(f12_w, f22), dim=1, index=self.index.to(f12).long())
    r42 = self.rconv2(f22)
    cat12 = torch.cat([cv12, r42, flow13_up], 1)
    flow12 = self.decoder2(cat12) + flow13_up

    flow_up1 = self.div_flow * F.interpolate(flow12, size=img2.shape[-2:], mode='bilinear', align_corners=False)

    outputs = {}
    if self.training:
      outputs['flow_preds'] = [[flow12],
                               [flow13],
                               [flow14],
                               [flow15],
                               [flow16]]
      outputs['latent'] = [f13, f23]

    outputs['flows'] = [flow_up1]
    return outputs

  def forward_three_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:]

    img1, img2, img3, _ = centralize([img1, img2, img3])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))
    f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
    f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
    f34 = F.avg_pool2d(f33, kernel_size=(2, 2), stride=(2, 2))
    f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
    f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
    f35 = F.avg_pool2d(f34, kernel_size=(2, 2), stride=(2, 2))
    f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
    f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))
    f36 = F.avg_pool2d(f35, kernel_size=(2, 2), stride=(2, 2))

    flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
    cv16 = torch.index_select(self.corr(f16, f36), dim=1, index=self.index.to(f16).long())
    cv26 = torch.index_select(self.corr(f26, f36), dim=1, index=self.index.to(f16).long())
    r46 = self.rconv6(f36)
    cat16 = torch.cat([cv16, r46, flow7_up], 1)
    cat26 = torch.cat([cv26, r46, flow7_up], 1)
    flow16 = self.decoder6(cat16)
    flow26 = self.decoder6(cat26)

    flow16_up = self.up6(flow16)
    flow26_up = self.up6(flow26)
    f15_w = self.warp(f15, flow16_up * 0.625)
    f25_w = self.warp(f25, flow26_up * 0.625)
    cv15 = torch.index_select(self.corr(f15_w, f35), dim=1, index=self.index.to(f15).long())
    cv25 = torch.index_select(self.corr(f25_w, f35), dim=1, index=self.index.to(f15).long())
    r45 = self.rconv5(f35)
    cat15 = torch.cat([cv15, r45, flow16_up], 1)
    cat25 = torch.cat([cv25, r45, flow26_up], 1)
    flow15 = self.decoder5(cat15) + flow16_up
    flow25 = self.decoder5(cat25) + flow26_up

    flow15_up = self.up5(flow15)
    flow25_up = self.up5(flow25)
    f14_w = self.warp(f14, flow15_up * 1.25)
    f24_w = self.warp(f24, flow25_up * 1.25)
    cv14 = torch.index_select(self.corr(f14_w, f34), dim=1, index=self.index.to(f14).long())
    cv24 = torch.index_select(self.corr(f24_w, f34), dim=1, index=self.index.to(f14).long())
    r44 = self.rconv4(f34)
    cat14 = torch.cat([cv14, r44, flow15_up], 1)
    cat24 = torch.cat([cv24, r44, flow25_up], 1)
    flow14 = self.decoder4(cat14) + flow15_up
    flow24 = self.decoder4(cat24) + flow25_up

    flow14_up = self.up4(flow14)
    flow24_up = self.up4(flow24)
    f13_w = self.warp(f13, flow14_up * 2.5)
    f23_w = self.warp(f23, flow24_up * 2.5)
    cv13 = torch.index_select(self.corr(f13_w, f33), dim=1, index=self.index.to(f13).long())
    cv23 = torch.index_select(self.corr(f23_w, f33), dim=1, index=self.index.to(f13).long())
    r43 = self.rconv3(f33)
    cat13 = torch.cat([cv13, r43, flow14_up], 1)
    cat23 = torch.cat([cv23, r43, flow24_up], 1)
    flow13 = self.decoder3(cat13) + flow14_up
    flow23 = self.decoder3(cat23) + flow24_up

    flow13_up = self.up3(flow13)
    flow23_up = self.up3(flow23)
    f12_w = self.warp(f12, flow13_up * 5.0)
    f22_w = self.warp(f22, flow23_up * 5.0)
    cv12 = torch.index_select(self.corr(f12_w, f32), dim=1, index=self.index.to(f12).long())
    cv22 = torch.index_select(self.corr(f22_w, f32), dim=1, index=self.index.to(f12).long())
    r42 = self.rconv2(f32)
    cat12 = torch.cat([cv12, r42, flow13_up], 1)
    cat22 = torch.cat([cv22, r42, flow23_up], 1)
    flow12 = self.decoder2(cat12) + flow13_up
    flow22 = self.decoder2(cat22) + flow23_up

    flow_up1 = self.div_flow * F.interpolate(flow12, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up2 = self.div_flow * F.interpolate(flow22, size=img2.shape[-2:], mode='bilinear', align_corners=False)

    outputs = {}
    if self.training:
      outputs['flow_preds'] = [[flow12, flow22],
                               [flow13, flow23],
                               [flow14, flow24],
                               [flow15, flow25],
                               [flow16, flow26]]
      outputs['latent'] = [f13, f23, f33]

    outputs['flows'] = [flow_up1, flow_up2]
    return outputs

  def forward_four_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:3 * self.C]
    img4 = inputs[:, 3 * self.C:]

    img1, img2, img3, img4, _ = centralize([img1, img2, img3, img4])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f41 = self.pconv1_2(self.pconv1_1(img4))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f42 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f41)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))
    f43 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f42)))
    f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
    f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
    f34 = F.avg_pool2d(f33, kernel_size=(2, 2), stride=(2, 2))
    f44 = F.avg_pool2d(f43, kernel_size=(2, 2), stride=(2, 2))
    f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
    f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
    f35 = F.avg_pool2d(f34, kernel_size=(2, 2), stride=(2, 2))
    f45 = F.avg_pool2d(f44, kernel_size=(2, 2), stride=(2, 2))
    f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
    f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))
    f36 = F.avg_pool2d(f35, kernel_size=(2, 2), stride=(2, 2))
    f46 = F.avg_pool2d(f45, kernel_size=(2, 2), stride=(2, 2))

    flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
    cv16 = torch.index_select(self.corr(f16, f46), dim=1, index=self.index.to(f16).long())
    cv26 = torch.index_select(self.corr(f26, f46), dim=1, index=self.index.to(f16).long())
    cv36 = torch.index_select(self.corr(f36, f46), dim=1, index=self.index.to(f16).long())
    r46 = self.rconv6(f46)
    cat16 = torch.cat([cv16, r46, flow7_up], 1)
    cat26 = torch.cat([cv26, r46, flow7_up], 1)
    cat36 = torch.cat([cv36, r46, flow7_up], 1)
    flow16 = self.decoder6(cat16)
    flow26 = self.decoder6(cat26)
    flow36 = self.decoder6(cat36)

    flow16_up = self.up6(flow16)
    flow26_up = self.up6(flow26)
    flow36_up = self.up6(flow36)
    f15_w = self.warp(f15, flow16_up * 0.625)
    f25_w = self.warp(f25, flow26_up * 0.625)
    f35_w = self.warp(f35, flow36_up * 0.625)
    cv15 = torch.index_select(self.corr(f15_w, f45), dim=1, index=self.index.to(f15).long())
    cv25 = torch.index_select(self.corr(f25_w, f45), dim=1, index=self.index.to(f15).long())
    cv35 = torch.index_select(self.corr(f35_w, f45), dim=1, index=self.index.to(f15).long())
    r45 = self.rconv5(f45)
    cat15 = torch.cat([cv15, r45, flow16_up], 1)
    cat25 = torch.cat([cv25, r45, flow26_up], 1)
    cat35 = torch.cat([cv35, r45, flow36_up], 1)
    flow15 = self.decoder5(cat15) + flow16_up
    flow25 = self.decoder5(cat25) + flow26_up
    flow35 = self.decoder5(cat35) + flow36_up

    flow15_up = self.up5(flow15)
    flow25_up = self.up5(flow25)
    flow35_up = self.up5(flow35)
    f14_w = self.warp(f14, flow15_up * 1.25)
    f24_w = self.warp(f24, flow25_up * 1.25)
    f34_w = self.warp(f34, flow35_up * 1.25)
    cv14 = torch.index_select(self.corr(f14_w, f44), dim=1, index=self.index.to(f14).long())
    cv24 = torch.index_select(self.corr(f24_w, f44), dim=1, index=self.index.to(f14).long())
    cv34 = torch.index_select(self.corr(f34_w, f44), dim=1, index=self.index.to(f14).long())
    r44 = self.rconv4(f44)
    cat14 = torch.cat([cv14, r44, flow15_up], 1)
    cat24 = torch.cat([cv24, r44, flow25_up], 1)
    cat34 = torch.cat([cv34, r44, flow35_up], 1)
    flow14 = self.decoder4(cat14) + flow15_up
    flow24 = self.decoder4(cat24) + flow25_up
    flow34 = self.decoder4(cat34) + flow35_up

    flow14_up = self.up4(flow14)
    flow24_up = self.up4(flow24)
    flow34_up = self.up4(flow34)
    f13_w = self.warp(f13, flow14_up * 2.5)
    f23_w = self.warp(f23, flow24_up * 2.5)
    f33_w = self.warp(f33, flow34_up * 2.5)
    cv13 = torch.index_select(self.corr(f13_w, f43), dim=1, index=self.index.to(f13).long())
    cv23 = torch.index_select(self.corr(f23_w, f43), dim=1, index=self.index.to(f13).long())
    cv33 = torch.index_select(self.corr(f33_w, f43), dim=1, index=self.index.to(f13).long())
    r43 = self.rconv3(f43)
    cat13 = torch.cat([cv13, r43, flow14_up], 1)
    cat23 = torch.cat([cv23, r43, flow24_up], 1)
    cat33 = torch.cat([cv33, r43, flow34_up], 1)
    flow13 = self.decoder3(cat13) + flow14_up
    flow23 = self.decoder3(cat23) + flow24_up
    flow33 = self.decoder3(cat33) + flow34_up

    flow13_up = self.up3(flow13)
    flow23_up = self.up3(flow23)
    flow33_up = self.up3(flow33)
    f12_w = self.warp(f12, flow13_up * 5.0)
    f22_w = self.warp(f22, flow23_up * 5.0)
    f32_w = self.warp(f32, flow33_up * 5.0)
    cv12 = torch.index_select(self.corr(f12_w, f42), dim=1, index=self.index.to(f12).long())
    cv22 = torch.index_select(self.corr(f22_w, f42), dim=1, index=self.index.to(f12).long())
    cv32 = torch.index_select(self.corr(f32_w, f42), dim=1, index=self.index.to(f12).long())
    r42 = self.rconv2(f42)
    cat12 = torch.cat([cv12, r42, flow13_up], 1)
    cat22 = torch.cat([cv22, r42, flow23_up], 1)
    cat32 = torch.cat([cv32, r42, flow33_up], 1)
    flow12 = self.decoder2(cat12) + flow13_up
    flow22 = self.decoder2(cat22) + flow23_up
    flow32 = self.decoder2(cat32) + flow33_up

    flow_up1 = self.div_flow * F.interpolate(flow12, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up2 = self.div_flow * F.interpolate(flow22, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up3 = self.div_flow * F.interpolate(flow32, size=img2.shape[-2:], mode='bilinear', align_corners=False)

    outputs = {}
    if self.training:
      outputs['flow_preds'] = [[flow12, flow22, flow32],
                               [flow13, flow23, flow33],
                               [flow14, flow24, flow34],
                               [flow15, flow25, flow35],
                               [flow16, flow26, flow36]]
      outputs['latent'] = [f13, f23, f33, f43]

    outputs['flows'] = [flow_up1, flow_up2, flow_up3]
    return outputs

  def forward_six_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:3 * self.C]
    img4 = inputs[:, 3 * self.C:4 * self.C]
    img5 = inputs[:, 4 * self.C:5 * self.C]
    img6 = inputs[:, 5 * self.C:]

    img1, img2, img3, img4, img5, img6, _ = centralize([img1, img2, img3, img4, img5, img6])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f41 = self.pconv1_2(self.pconv1_1(img4))
    f51 = self.pconv1_2(self.pconv1_1(img5))
    f61 = self.pconv1_2(self.pconv1_1(img6))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f42 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f41)))
    f52 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f51)))
    f62 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f61)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))
    f43 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f42)))
    f53 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f52)))
    f63 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f62)))
    f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
    f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
    f34 = F.avg_pool2d(f33, kernel_size=(2, 2), stride=(2, 2))
    f44 = F.avg_pool2d(f43, kernel_size=(2, 2), stride=(2, 2))
    f54 = F.avg_pool2d(f53, kernel_size=(2, 2), stride=(2, 2))
    f64 = F.avg_pool2d(f63, kernel_size=(2, 2), stride=(2, 2))
    f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
    f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
    f35 = F.avg_pool2d(f34, kernel_size=(2, 2), stride=(2, 2))
    f45 = F.avg_pool2d(f44, kernel_size=(2, 2), stride=(2, 2))
    f55 = F.avg_pool2d(f54, kernel_size=(2, 2), stride=(2, 2))
    f65 = F.avg_pool2d(f64, kernel_size=(2, 2), stride=(2, 2))
    f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
    f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))
    f36 = F.avg_pool2d(f35, kernel_size=(2, 2), stride=(2, 2))
    f46 = F.avg_pool2d(f45, kernel_size=(2, 2), stride=(2, 2))
    f56 = F.avg_pool2d(f55, kernel_size=(2, 2), stride=(2, 2))
    f66 = F.avg_pool2d(f65, kernel_size=(2, 2), stride=(2, 2))

    flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
    cv16 = torch.index_select(self.corr(f16, f66), dim=1, index=self.index.to(f16).long())
    cv26 = torch.index_select(self.corr(f26, f66), dim=1, index=self.index.to(f16).long())
    cv36 = torch.index_select(self.corr(f36, f66), dim=1, index=self.index.to(f16).long())
    cv46 = torch.index_select(self.corr(f46, f66), dim=1, index=self.index.to(f16).long())
    cv56 = torch.index_select(self.corr(f56, f66), dim=1, index=self.index.to(f16).long())
    r66 = self.rconv6(f66)
    cat16 = torch.cat([cv16, r66, flow7_up], 1)
    cat26 = torch.cat([cv26, r66, flow7_up], 1)
    cat36 = torch.cat([cv36, r66, flow7_up], 1)
    cat46 = torch.cat([cv46, r66, flow7_up], 1)
    cat56 = torch.cat([cv56, r66, flow7_up], 1)
    flow16 = self.decoder6(cat16)
    flow26 = self.decoder6(cat26)
    flow36 = self.decoder6(cat36)
    flow46 = self.decoder6(cat46)
    flow56 = self.decoder6(cat56)

    flow16_up = self.up6(flow16)
    flow26_up = self.up6(flow26)
    flow36_up = self.up6(flow36)
    flow46_up = self.up6(flow46)
    flow56_up = self.up6(flow56)
    f15_w = self.warp(f15, flow16_up * 0.625)
    f25_w = self.warp(f25, flow26_up * 0.625)
    f35_w = self.warp(f35, flow36_up * 0.625)
    f45_w = self.warp(f45, flow46_up * 0.625)
    f55_w = self.warp(f55, flow56_up * 0.625)
    cv15 = torch.index_select(self.corr(f15_w, f65), dim=1, index=self.index.to(f15).long())
    cv25 = torch.index_select(self.corr(f25_w, f65), dim=1, index=self.index.to(f15).long())
    cv35 = torch.index_select(self.corr(f35_w, f65), dim=1, index=self.index.to(f15).long())
    cv45 = torch.index_select(self.corr(f45_w, f65), dim=1, index=self.index.to(f15).long())
    cv55 = torch.index_select(self.corr(f55_w, f65), dim=1, index=self.index.to(f15).long())
    r65 = self.rconv5(f65)
    cat15 = torch.cat([cv15, r65, flow16_up], 1)
    cat25 = torch.cat([cv25, r65, flow26_up], 1)
    cat35 = torch.cat([cv35, r65, flow36_up], 1)
    cat45 = torch.cat([cv45, r65, flow46_up], 1)
    cat55 = torch.cat([cv55, r65, flow56_up], 1)
    flow15 = self.decoder5(cat15) + flow16_up
    flow25 = self.decoder5(cat25) + flow26_up
    flow35 = self.decoder5(cat35) + flow36_up
    flow45 = self.decoder5(cat45) + flow46_up
    flow55 = self.decoder5(cat55) + flow56_up

    flow15_up = self.up5(flow15)
    flow25_up = self.up5(flow25)
    flow35_up = self.up5(flow35)
    flow45_up = self.up5(flow45)
    flow55_up = self.up5(flow55)
    f14_w = self.warp(f14, flow15_up * 1.25)
    f24_w = self.warp(f24, flow25_up * 1.25)
    f34_w = self.warp(f34, flow35_up * 1.25)
    f44_w = self.warp(f44, flow45_up * 1.25)
    f54_w = self.warp(f54, flow55_up * 1.25)
    cv14 = torch.index_select(self.corr(f14_w, f64), dim=1, index=self.index.to(f14).long())
    cv24 = torch.index_select(self.corr(f24_w, f64), dim=1, index=self.index.to(f14).long())
    cv34 = torch.index_select(self.corr(f34_w, f64), dim=1, index=self.index.to(f14).long())
    cv44 = torch.index_select(self.corr(f44_w, f64), dim=1, index=self.index.to(f14).long())
    cv54 = torch.index_select(self.corr(f54_w, f64), dim=1, index=self.index.to(f14).long())
    r64 = self.rconv4(f64)
    cat14 = torch.cat([cv14, r64, flow15_up], 1)
    cat24 = torch.cat([cv24, r64, flow25_up], 1)
    cat34 = torch.cat([cv34, r64, flow35_up], 1)
    cat44 = torch.cat([cv44, r64, flow45_up], 1)
    cat54 = torch.cat([cv54, r64, flow55_up], 1)
    flow14 = self.decoder4(cat14) + flow15_up
    flow24 = self.decoder4(cat24) + flow25_up
    flow34 = self.decoder4(cat34) + flow35_up
    flow44 = self.decoder4(cat44) + flow45_up
    flow54 = self.decoder4(cat54) + flow55_up

    flow14_up = self.up4(flow14)
    flow24_up = self.up4(flow24)
    flow34_up = self.up4(flow34)
    flow44_up = self.up4(flow44)
    flow54_up = self.up4(flow54)
    f13_w = self.warp(f13, flow14_up * 2.5)
    f23_w = self.warp(f23, flow24_up * 2.5)
    f33_w = self.warp(f33, flow34_up * 2.5)
    f43_w = self.warp(f43, flow34_up * 2.5)
    f53_w = self.warp(f53, flow34_up * 2.5)
    cv13 = torch.index_select(self.corr(f13_w, f63), dim=1, index=self.index.to(f13).long())
    cv23 = torch.index_select(self.corr(f23_w, f63), dim=1, index=self.index.to(f13).long())
    cv33 = torch.index_select(self.corr(f33_w, f63), dim=1, index=self.index.to(f13).long())
    cv43 = torch.index_select(self.corr(f43_w, f63), dim=1, index=self.index.to(f13).long())
    cv53 = torch.index_select(self.corr(f53_w, f63), dim=1, index=self.index.to(f13).long())
    r63 = self.rconv3(f63)
    cat13 = torch.cat([cv13, r63, flow14_up], 1)
    cat23 = torch.cat([cv23, r63, flow24_up], 1)
    cat33 = torch.cat([cv33, r63, flow34_up], 1)
    cat43 = torch.cat([cv43, r63, flow44_up], 1)
    cat53 = torch.cat([cv53, r63, flow54_up], 1)
    flow13 = self.decoder3(cat13) + flow14_up
    flow23 = self.decoder3(cat23) + flow24_up
    flow33 = self.decoder3(cat33) + flow34_up
    flow43 = self.decoder3(cat43) + flow44_up
    flow53 = self.decoder3(cat53) + flow54_up

    flow13_up = self.up3(flow13)
    flow23_up = self.up3(flow23)
    flow33_up = self.up3(flow33)
    flow43_up = self.up3(flow43)
    flow53_up = self.up3(flow53)
    f12_w = self.warp(f12, flow13_up * 5.0)
    f22_w = self.warp(f22, flow23_up * 5.0)
    f32_w = self.warp(f32, flow33_up * 5.0)
    f42_w = self.warp(f42, flow43_up * 5.0)
    f52_w = self.warp(f52, flow53_up * 5.0)
    cv12 = torch.index_select(self.corr(f12_w, f62), dim=1, index=self.index.to(f12).long())
    cv22 = torch.index_select(self.corr(f22_w, f62), dim=1, index=self.index.to(f12).long())
    cv32 = torch.index_select(self.corr(f32_w, f62), dim=1, index=self.index.to(f12).long())
    cv42 = torch.index_select(self.corr(f42_w, f62), dim=1, index=self.index.to(f12).long())
    cv52 = torch.index_select(self.corr(f52_w, f62), dim=1, index=self.index.to(f12).long())
    r62 = self.rconv2(f62)
    cat12 = torch.cat([cv12, r62, flow13_up], 1)
    cat22 = torch.cat([cv22, r62, flow23_up], 1)
    cat32 = torch.cat([cv32, r62, flow33_up], 1)
    cat42 = torch.cat([cv42, r62, flow43_up], 1)
    cat52 = torch.cat([cv52, r62, flow53_up], 1)
    flow12 = self.decoder2(cat12) + flow13_up
    flow22 = self.decoder2(cat22) + flow23_up
    flow32 = self.decoder2(cat32) + flow33_up
    flow42 = self.decoder2(cat42) + flow43_up
    flow52 = self.decoder2(cat52) + flow53_up

    flow_up1 = self.div_flow * F.interpolate(flow12, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up2 = self.div_flow * F.interpolate(flow22, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up3 = self.div_flow * F.interpolate(flow32, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up4 = self.div_flow * F.interpolate(flow42, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up5 = self.div_flow * F.interpolate(flow52, size=img2.shape[-2:], mode='bilinear', align_corners=False)

    outputs = {}
    if self.training:
      outputs['flow_preds'] = [[flow12, flow22, flow32, flow42, flow52],
                               [flow13, flow23, flow33, flow43, flow53],
                               [flow14, flow24, flow34, flow44, flow54],
                               [flow15, flow25, flow35, flow45, flow55],
                               [flow16, flow26, flow36, flow46, flow56]]
      outputs['latent'] = [f13, f23, f33, f43, f53, f63]

    outputs['flows'] = [flow_up1, flow_up2, flow_up3, flow_up4, flow_up5]
    return outputs

  def forward_twelve_flows(self, inputs):
    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:3 * self.C]
    img4 = inputs[:, 3 * self.C:4 * self.C]
    img5 = inputs[:, 4 * self.C:5 * self.C]
    img6 = inputs[:, 5 * self.C:6 * self.C]
    img7 = inputs[:, 6 * self.C:7 * self.C]
    img8 = inputs[:, 7 * self.C:8 * self.C]
    img9 = inputs[:, 8 * self.C:9 * self.C]
    img10 = inputs[:, 9 * self.C:10 * self.C]
    img11 = inputs[:, 10 * self.C:11 * self.C]
    img12 = inputs[:, 11 * self.C:]

    img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, _ = centralize([img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f41 = self.pconv1_2(self.pconv1_1(img4))
    f51 = self.pconv1_2(self.pconv1_1(img5))
    f61 = self.pconv1_2(self.pconv1_1(img6))
    f71 = self.pconv1_2(self.pconv1_1(img7))
    f81 = self.pconv1_2(self.pconv1_1(img8))
    f91 = self.pconv1_2(self.pconv1_1(img9))
    f101 = self.pconv1_2(self.pconv1_1(img10))
    f111 = self.pconv1_2(self.pconv1_1(img11))
    f121 = self.pconv1_2(self.pconv1_1(img12))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f42 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f41)))
    f52 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f51)))
    f62 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f61)))
    f72 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f71)))
    f82 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f81)))
    f92 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f91)))
    f102 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f101)))
    f112 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f111)))
    f122 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f121)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))
    f43 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f42)))
    f53 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f52)))
    f63 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f62)))
    f73 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f72)))
    f83 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f82)))
    f93 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f92)))
    f103 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f102)))
    f113 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f112)))
    f123 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f122)))
    f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
    f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
    f34 = F.avg_pool2d(f33, kernel_size=(2, 2), stride=(2, 2))
    f44 = F.avg_pool2d(f43, kernel_size=(2, 2), stride=(2, 2))
    f54 = F.avg_pool2d(f53, kernel_size=(2, 2), stride=(2, 2))
    f64 = F.avg_pool2d(f63, kernel_size=(2, 2), stride=(2, 2))
    f74 = F.avg_pool2d(f73, kernel_size=(2, 2), stride=(2, 2))
    f84 = F.avg_pool2d(f83, kernel_size=(2, 2), stride=(2, 2))
    f94 = F.avg_pool2d(f93, kernel_size=(2, 2), stride=(2, 2))
    f104 = F.avg_pool2d(f103, kernel_size=(2, 2), stride=(2, 2))
    f114 = F.avg_pool2d(f113, kernel_size=(2, 2), stride=(2, 2))
    f124 = F.avg_pool2d(f123, kernel_size=(2, 2), stride=(2, 2))
    f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
    f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
    f35 = F.avg_pool2d(f34, kernel_size=(2, 2), stride=(2, 2))
    f45 = F.avg_pool2d(f44, kernel_size=(2, 2), stride=(2, 2))
    f55 = F.avg_pool2d(f54, kernel_size=(2, 2), stride=(2, 2))
    f65 = F.avg_pool2d(f64, kernel_size=(2, 2), stride=(2, 2))
    f75 = F.avg_pool2d(f74, kernel_size=(2, 2), stride=(2, 2))
    f85 = F.avg_pool2d(f84, kernel_size=(2, 2), stride=(2, 2))
    f95 = F.avg_pool2d(f94, kernel_size=(2, 2), stride=(2, 2))
    f105 = F.avg_pool2d(f104, kernel_size=(2, 2), stride=(2, 2))
    f115 = F.avg_pool2d(f114, kernel_size=(2, 2), stride=(2, 2))
    f125 = F.avg_pool2d(f124, kernel_size=(2, 2), stride=(2, 2))
    f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
    f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))
    f36 = F.avg_pool2d(f35, kernel_size=(2, 2), stride=(2, 2))
    f46 = F.avg_pool2d(f45, kernel_size=(2, 2), stride=(2, 2))
    f56 = F.avg_pool2d(f55, kernel_size=(2, 2), stride=(2, 2))
    f66 = F.avg_pool2d(f65, kernel_size=(2, 2), stride=(2, 2))
    f76 = F.avg_pool2d(f75, kernel_size=(2, 2), stride=(2, 2))
    f86 = F.avg_pool2d(f85, kernel_size=(2, 2), stride=(2, 2))
    f96 = F.avg_pool2d(f95, kernel_size=(2, 2), stride=(2, 2))
    f106 = F.avg_pool2d(f105, kernel_size=(2, 2), stride=(2, 2))
    f116 = F.avg_pool2d(f115, kernel_size=(2, 2), stride=(2, 2))
    f126 = F.avg_pool2d(f125, kernel_size=(2, 2), stride=(2, 2))

    flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3)).to(f15)
    cv16 = torch.index_select(self.corr(f16, f126), dim=1, index=self.index.to(f16).long())
    cv26 = torch.index_select(self.corr(f26, f126), dim=1, index=self.index.to(f16).long())
    cv36 = torch.index_select(self.corr(f36, f126), dim=1, index=self.index.to(f16).long())
    cv46 = torch.index_select(self.corr(f46, f126), dim=1, index=self.index.to(f16).long())
    cv56 = torch.index_select(self.corr(f56, f126), dim=1, index=self.index.to(f16).long())
    cv66 = torch.index_select(self.corr(f66, f126), dim=1, index=self.index.to(f16).long())
    cv76 = torch.index_select(self.corr(f76, f126), dim=1, index=self.index.to(f16).long())
    cv86 = torch.index_select(self.corr(f86, f126), dim=1, index=self.index.to(f16).long())
    cv96 = torch.index_select(self.corr(f96, f126), dim=1, index=self.index.to(f16).long())
    cv106 = torch.index_select(self.corr(f106, f126), dim=1, index=self.index.to(f16).long())
    cv116 = torch.index_select(self.corr(f116, f126), dim=1, index=self.index.to(f16).long())
    r126 = self.rconv6(f126)
    cat16 = torch.cat([cv16, r126, flow7_up], 1)
    cat26 = torch.cat([cv26, r126, flow7_up], 1)
    cat36 = torch.cat([cv36, r126, flow7_up], 1)
    cat46 = torch.cat([cv46, r126, flow7_up], 1)
    cat56 = torch.cat([cv56, r126, flow7_up], 1)
    cat66 = torch.cat([cv66, r126, flow7_up], 1)
    cat76 = torch.cat([cv76, r126, flow7_up], 1)
    cat86 = torch.cat([cv86, r126, flow7_up], 1)
    cat96 = torch.cat([cv96, r126, flow7_up], 1)
    cat106 = torch.cat([cv106, r126, flow7_up], 1)
    cat116 = torch.cat([cv116, r126, flow7_up], 1)
    flow16 = self.decoder6(cat16)
    flow26 = self.decoder6(cat26)
    flow36 = self.decoder6(cat36)
    flow46 = self.decoder6(cat46)
    flow56 = self.decoder6(cat56)
    flow66 = self.decoder6(cat66)
    flow76 = self.decoder6(cat76)
    flow86 = self.decoder6(cat86)
    flow96 = self.decoder6(cat96)
    flow106 = self.decoder6(cat106)
    flow116 = self.decoder6(cat116)

    flow16_up = self.up6(flow16)
    flow26_up = self.up6(flow26)
    flow36_up = self.up6(flow36)
    flow46_up = self.up6(flow46)
    flow56_up = self.up6(flow56)
    flow66_up = self.up6(flow66)
    flow76_up = self.up6(flow76)
    flow86_up = self.up6(flow86)
    flow96_up = self.up6(flow96)
    flow106_up = self.up6(flow106)
    flow116_up = self.up6(flow116)
    f15_w = self.warp(f15, flow16_up * 0.625)
    f25_w = self.warp(f25, flow26_up * 0.625)
    f35_w = self.warp(f35, flow36_up * 0.625)
    f45_w = self.warp(f45, flow46_up * 0.625)
    f55_w = self.warp(f55, flow56_up * 0.625)
    f65_w = self.warp(f65, flow66_up * 0.625)
    f75_w = self.warp(f75, flow76_up * 0.625)
    f85_w = self.warp(f85, flow86_up * 0.625)
    f95_w = self.warp(f95, flow96_up * 0.625)
    f105_w = self.warp(f105, flow106_up * 0.625)
    f115_w = self.warp(f115, flow116_up * 0.625)
    cv15 = torch.index_select(self.corr(f15_w, f125), dim=1, index=self.index.to(f15).long())
    cv25 = torch.index_select(self.corr(f25_w, f125), dim=1, index=self.index.to(f15).long())
    cv35 = torch.index_select(self.corr(f35_w, f125), dim=1, index=self.index.to(f15).long())
    cv45 = torch.index_select(self.corr(f45_w, f125), dim=1, index=self.index.to(f15).long())
    cv55 = torch.index_select(self.corr(f55_w, f125), dim=1, index=self.index.to(f15).long())
    cv65 = torch.index_select(self.corr(f65_w, f125), dim=1, index=self.index.to(f15).long())
    cv75 = torch.index_select(self.corr(f75_w, f125), dim=1, index=self.index.to(f15).long())
    cv85 = torch.index_select(self.corr(f85_w, f125), dim=1, index=self.index.to(f15).long())
    cv95 = torch.index_select(self.corr(f95_w, f125), dim=1, index=self.index.to(f15).long())
    cv105 = torch.index_select(self.corr(f105_w, f125), dim=1, index=self.index.to(f15).long())
    cv115 = torch.index_select(self.corr(f115_w, f125), dim=1, index=self.index.to(f15).long())
    r125 = self.rconv5(f125)
    cat15 = torch.cat([cv15, r125, flow16_up], 1)
    cat25 = torch.cat([cv25, r125, flow26_up], 1)
    cat35 = torch.cat([cv35, r125, flow36_up], 1)
    cat45 = torch.cat([cv45, r125, flow46_up], 1)
    cat55 = torch.cat([cv55, r125, flow56_up], 1)
    cat65 = torch.cat([cv65, r125, flow56_up], 1)
    cat75 = torch.cat([cv75, r125, flow56_up], 1)
    cat85 = torch.cat([cv85, r125, flow56_up], 1)
    cat95 = torch.cat([cv95, r125, flow56_up], 1)
    cat105 = torch.cat([cv105, r125, flow56_up], 1)
    cat115 = torch.cat([cv115, r125, flow56_up], 1)
    flow15 = self.decoder5(cat15) + flow16_up
    flow25 = self.decoder5(cat25) + flow26_up
    flow35 = self.decoder5(cat35) + flow36_up
    flow45 = self.decoder5(cat45) + flow46_up
    flow55 = self.decoder5(cat55) + flow56_up
    flow65 = self.decoder5(cat65) + flow56_up
    flow75 = self.decoder5(cat75) + flow56_up
    flow85 = self.decoder5(cat85) + flow56_up
    flow95 = self.decoder5(cat95) + flow56_up
    flow105 = self.decoder5(cat105) + flow56_up
    flow115 = self.decoder5(cat115) + flow56_up

    flow15_up = self.up5(flow15)
    flow25_up = self.up5(flow25)
    flow35_up = self.up5(flow35)
    flow45_up = self.up5(flow45)
    flow55_up = self.up5(flow55)
    flow65_up = self.up5(flow65)
    flow75_up = self.up5(flow75)
    flow85_up = self.up5(flow85)
    flow95_up = self.up5(flow95)
    flow105_up = self.up5(flow105)
    flow115_up = self.up5(flow115)
    f14_w = self.warp(f14, flow15_up * 1.25)
    f24_w = self.warp(f24, flow25_up * 1.25)
    f34_w = self.warp(f34, flow35_up * 1.25)
    f44_w = self.warp(f44, flow45_up * 1.25)
    f54_w = self.warp(f54, flow55_up * 1.25)
    f64_w = self.warp(f64, flow65_up * 1.25)
    f74_w = self.warp(f74, flow75_up * 1.25)
    f84_w = self.warp(f84, flow85_up * 1.25)
    f94_w = self.warp(f94, flow95_up * 1.25)
    f104_w = self.warp(f104, flow105_up * 1.25)
    f114_w = self.warp(f114, flow115_up * 1.25)
    cv14 = torch.index_select(self.corr(f14_w, f124), dim=1, index=self.index.to(f14).long())
    cv24 = torch.index_select(self.corr(f24_w, f124), dim=1, index=self.index.to(f14).long())
    cv34 = torch.index_select(self.corr(f34_w, f124), dim=1, index=self.index.to(f14).long())
    cv44 = torch.index_select(self.corr(f44_w, f124), dim=1, index=self.index.to(f14).long())
    cv54 = torch.index_select(self.corr(f54_w, f124), dim=1, index=self.index.to(f14).long())
    cv64 = torch.index_select(self.corr(f64_w, f124), dim=1, index=self.index.to(f14).long())
    cv74 = torch.index_select(self.corr(f74_w, f124), dim=1, index=self.index.to(f14).long())
    cv84 = torch.index_select(self.corr(f84_w, f124), dim=1, index=self.index.to(f14).long())
    cv94 = torch.index_select(self.corr(f94_w, f124), dim=1, index=self.index.to(f14).long())
    cv104 = torch.index_select(self.corr(f104_w, f124), dim=1, index=self.index.to(f14).long())
    cv114 = torch.index_select(self.corr(f114_w, f124), dim=1, index=self.index.to(f14).long())
    r124 = self.rconv4(f124)
    cat14 = torch.cat([cv14, r124, flow15_up], 1)
    cat24 = torch.cat([cv24, r124, flow25_up], 1)
    cat34 = torch.cat([cv34, r124, flow35_up], 1)
    cat44 = torch.cat([cv44, r124, flow45_up], 1)
    cat54 = torch.cat([cv54, r124, flow55_up], 1)
    cat64 = torch.cat([cv64, r124, flow55_up], 1)
    cat74 = torch.cat([cv74, r124, flow55_up], 1)
    cat84 = torch.cat([cv84, r124, flow55_up], 1)
    cat94 = torch.cat([cv94, r124, flow55_up], 1)
    cat104 = torch.cat([cv104, r124, flow55_up], 1)
    cat114 = torch.cat([cv114, r124, flow55_up], 1)
    flow14 = self.decoder4(cat14) + flow15_up
    flow24 = self.decoder4(cat24) + flow25_up
    flow34 = self.decoder4(cat34) + flow35_up
    flow44 = self.decoder4(cat44) + flow45_up
    flow54 = self.decoder4(cat54) + flow55_up
    flow64 = self.decoder4(cat64) + flow65_up
    flow74 = self.decoder4(cat74) + flow75_up
    flow84 = self.decoder4(cat84) + flow85_up
    flow94 = self.decoder4(cat94) + flow95_up
    flow104 = self.decoder4(cat104) + flow105_up
    flow114 = self.decoder4(cat114) + flow115_up

    flow14_up = self.up4(flow14)
    flow24_up = self.up4(flow24)
    flow34_up = self.up4(flow34)
    flow44_up = self.up4(flow44)
    flow54_up = self.up4(flow54)
    flow64_up = self.up4(flow64)
    flow74_up = self.up4(flow74)
    flow84_up = self.up4(flow84)
    flow94_up = self.up4(flow94)
    flow104_up = self.up4(flow104)
    flow114_up = self.up4(flow114)
    f13_w = self.warp(f13, flow14_up * 2.5)
    f23_w = self.warp(f23, flow24_up * 2.5)
    f33_w = self.warp(f33, flow34_up * 2.5)
    f43_w = self.warp(f43, flow44_up * 2.5)
    f53_w = self.warp(f53, flow54_up * 2.5)
    f63_w = self.warp(f63, flow64_up * 2.5)
    f73_w = self.warp(f73, flow74_up * 2.5)
    f83_w = self.warp(f83, flow84_up * 2.5)
    f93_w = self.warp(f93, flow94_up * 2.5)
    f103_w = self.warp(f103, flow104_up * 2.5)
    f113_w = self.warp(f113, flow114_up * 2.5)
    cv13 = torch.index_select(self.corr(f13_w, f123), dim=1, index=self.index.to(f13).long())
    cv23 = torch.index_select(self.corr(f23_w, f123), dim=1, index=self.index.to(f13).long())
    cv33 = torch.index_select(self.corr(f33_w, f123), dim=1, index=self.index.to(f13).long())
    cv43 = torch.index_select(self.corr(f43_w, f123), dim=1, index=self.index.to(f13).long())
    cv53 = torch.index_select(self.corr(f53_w, f123), dim=1, index=self.index.to(f13).long())
    cv63 = torch.index_select(self.corr(f63_w, f123), dim=1, index=self.index.to(f13).long())
    cv73 = torch.index_select(self.corr(f73_w, f123), dim=1, index=self.index.to(f13).long())
    cv83 = torch.index_select(self.corr(f83_w, f123), dim=1, index=self.index.to(f13).long())
    cv93 = torch.index_select(self.corr(f93_w, f123), dim=1, index=self.index.to(f13).long())
    cv103 = torch.index_select(self.corr(f103_w, f123), dim=1, index=self.index.to(f13).long())
    cv113 = torch.index_select(self.corr(f113_w, f123), dim=1, index=self.index.to(f13).long())
    r123 = self.rconv3(f123)
    cat13 = torch.cat([cv13, r123, flow14_up], 1)
    cat23 = torch.cat([cv23, r123, flow24_up], 1)
    cat33 = torch.cat([cv33, r123, flow34_up], 1)
    cat43 = torch.cat([cv43, r123, flow44_up], 1)
    cat53 = torch.cat([cv53, r123, flow54_up], 1)
    cat63 = torch.cat([cv63, r123, flow54_up], 1)
    cat73 = torch.cat([cv73, r123, flow54_up], 1)
    cat83 = torch.cat([cv83, r123, flow54_up], 1)
    cat93 = torch.cat([cv93, r123, flow54_up], 1)
    cat103 = torch.cat([cv103, r123, flow54_up], 1)
    cat113 = torch.cat([cv113, r123, flow54_up], 1)
    flow13 = self.decoder3(cat13) + flow14_up
    flow23 = self.decoder3(cat23) + flow24_up
    flow33 = self.decoder3(cat33) + flow34_up
    flow43 = self.decoder3(cat43) + flow44_up
    flow53 = self.decoder3(cat53) + flow54_up
    flow63 = self.decoder3(cat63) + flow64_up
    flow73 = self.decoder3(cat73) + flow74_up
    flow83 = self.decoder3(cat83) + flow84_up
    flow93 = self.decoder3(cat93) + flow94_up
    flow103 = self.decoder3(cat103) + flow104_up
    flow113 = self.decoder3(cat113) + flow114_up

    flow13_up = self.up3(flow13)
    flow23_up = self.up3(flow23)
    flow33_up = self.up3(flow33)
    flow43_up = self.up3(flow43)
    flow53_up = self.up3(flow53)
    flow63_up = self.up3(flow63)
    flow73_up = self.up3(flow73)
    flow83_up = self.up3(flow83)
    flow93_up = self.up3(flow93)
    flow103_up = self.up3(flow103)
    flow113_up = self.up3(flow113)
    f12_w = self.warp(f12, flow13_up * 5.0)
    f22_w = self.warp(f22, flow23_up * 5.0)
    f32_w = self.warp(f32, flow33_up * 5.0)
    f42_w = self.warp(f42, flow43_up * 5.0)
    f52_w = self.warp(f52, flow53_up * 5.0)
    f62_w = self.warp(f62, flow63_up * 5.0)
    f72_w = self.warp(f72, flow73_up * 5.0)
    f82_w = self.warp(f82, flow83_up * 5.0)
    f92_w = self.warp(f92, flow93_up * 5.0)
    f102_w = self.warp(f102, flow103_up * 5.0)
    f112_w = self.warp(f112, flow113_up * 5.0)
    cv12 = torch.index_select(self.corr(f12_w, f122), dim=1, index=self.index.to(f12).long())
    cv22 = torch.index_select(self.corr(f22_w, f122), dim=1, index=self.index.to(f12).long())
    cv32 = torch.index_select(self.corr(f32_w, f122), dim=1, index=self.index.to(f12).long())
    cv42 = torch.index_select(self.corr(f42_w, f122), dim=1, index=self.index.to(f12).long())
    cv52 = torch.index_select(self.corr(f52_w, f122), dim=1, index=self.index.to(f12).long())
    cv62 = torch.index_select(self.corr(f62_w, f122), dim=1, index=self.index.to(f12).long())
    cv72 = torch.index_select(self.corr(f72_w, f122), dim=1, index=self.index.to(f12).long())
    cv82 = torch.index_select(self.corr(f82_w, f122), dim=1, index=self.index.to(f12).long())
    cv92 = torch.index_select(self.corr(f92_w, f122), dim=1, index=self.index.to(f12).long())
    cv102 = torch.index_select(self.corr(f102_w, f122), dim=1, index=self.index.to(f12).long())
    cv112 = torch.index_select(self.corr(f112_w, f122), dim=1, index=self.index.to(f12).long())
    r122 = self.rconv2(f122)
    cat12 = torch.cat([cv12, r122, flow13_up], 1)
    cat22 = torch.cat([cv22, r122, flow23_up], 1)
    cat32 = torch.cat([cv32, r122, flow33_up], 1)
    cat42 = torch.cat([cv42, r122, flow43_up], 1)
    cat52 = torch.cat([cv52, r122, flow53_up], 1)
    cat62 = torch.cat([cv62, r122, flow53_up], 1)
    cat72 = torch.cat([cv72, r122, flow53_up], 1)
    cat82 = torch.cat([cv82, r122, flow53_up], 1)
    cat92 = torch.cat([cv92, r122, flow53_up], 1)
    cat102 = torch.cat([cv102, r122, flow53_up], 1)
    cat112 = torch.cat([cv112, r122, flow53_up], 1)
    flow12 = self.decoder2(cat12) + flow13_up
    flow22 = self.decoder2(cat22) + flow23_up
    flow32 = self.decoder2(cat32) + flow33_up
    flow42 = self.decoder2(cat42) + flow43_up
    flow52 = self.decoder2(cat52) + flow53_up
    flow62 = self.decoder2(cat62) + flow63_up
    flow72 = self.decoder2(cat72) + flow73_up
    flow82 = self.decoder2(cat82) + flow83_up
    flow92 = self.decoder2(cat92) + flow93_up
    flow102 = self.decoder2(cat102) + flow103_up
    flow112 = self.decoder2(cat112) + flow103_up

    flow_up1 = self.div_flow * F.interpolate(flow12, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up2 = self.div_flow * F.interpolate(flow22, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up3 = self.div_flow * F.interpolate(flow32, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up4 = self.div_flow * F.interpolate(flow42, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up5 = self.div_flow * F.interpolate(flow52, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up6 = self.div_flow * F.interpolate(flow62, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up7 = self.div_flow * F.interpolate(flow72, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up8 = self.div_flow * F.interpolate(flow82, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up9 = self.div_flow * F.interpolate(flow92, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up10 = self.div_flow * F.interpolate(flow102, size=img2.shape[-2:], mode='bilinear', align_corners=False)
    flow_up11 = self.div_flow * F.interpolate(flow112, size=img2.shape[-2:], mode='bilinear', align_corners=False)

    outputs = {}
    if self.training:
      outputs['flow_preds'] = [[flow12, flow22, flow32, flow42, flow52, flow62, flow72, flow82, flow92, flow102, flow112],
                               [flow13, flow23, flow33, flow43, flow53, flow63, flow73, flow83, flow93, flow103, flow113],
                               [flow14, flow24, flow34, flow44, flow54, flow64, flow74, flow84, flow94, flow104, flow114],
                               [flow15, flow25, flow35, flow45, flow55, flow65, flow75, flow85, flow95, flow105, flow115],
                               [flow16, flow26, flow36, flow46, flow56, flow66, flow76, flow86, flow96, flow106, flow116]]
      outputs['latent'] = [f13, f23, f33, f43, f53, f63, f73, f83, f93, f103, f113, f123]

    outputs['flows'] = [flow_up1, flow_up2, flow_up3, flow_up4, flow_up5, flow_up6, flow_up7, flow_up8, flow_up9, flow_up10, flow_up11]
    return outputs

  def encode(self, inputs):
    if self.taps == 2:
      inputs = self.twotap_reordering(inputs)

    if self.taps != 1 and self.in_channels == 1:
      # Lindner method
      inputs = self.compute_intensity_on_taps(inputs)

    if self.time_steps == 2:
      return self.encode_two_flows(inputs)
    elif self.time_steps == 3:
      return self.encode_three_flows(inputs)
    elif self.time_steps == 4:
      return self.encode_four_flows(inputs)
    elif self.time_steps == 6:
      return self.encode_six_flows(inputs)
    elif self.time_steps == 12:
      return self.encode_twelve_flows(inputs)

  def encode_two_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:]

    img1, img2, _ = centralize([img1, img2])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))

    outputs = {}
    if self.training:
      outputs['flow_preds'] = None
      outputs['latent'] = [f13, f23]

    outputs['flows'] = None
    return outputs

  def encode_three_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:]

    img1, img2, img3, _ = centralize([img1, img2, img3])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))

    outputs = {}
    if self.training:
      outputs['flow_preds'] = None
      outputs['latent'] = [f13, f23, f33]

    outputs['flows'] = None
    return outputs

  def encode_four_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:3 * self.C]
    img4 = inputs[:, 3 * self.C:]

    img1, img2, img3, img4, _ = centralize([img1, img2, img3, img4])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f41 = self.pconv1_2(self.pconv1_1(img4))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f42 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f41)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))
    f43 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f42)))
    outputs = {}
    if self.training:
      outputs['flow_preds'] = None
      outputs['latent'] = [f13, f23, f33, f43]

    outputs['flows'] = None
    return outputs

  def encode_six_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:3 * self.C]
    img4 = inputs[:, 3 * self.C:4 * self.C]
    img5 = inputs[:, 4 * self.C:5 * self.C]
    img6 = inputs[:, 5 * self.C:]

    img1, img2, img3, img4, img5, img6, _ = centralize([img1, img2, img3, img4, img5, img6])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f41 = self.pconv1_2(self.pconv1_1(img4))
    f51 = self.pconv1_2(self.pconv1_1(img5))
    f61 = self.pconv1_2(self.pconv1_1(img6))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f42 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f41)))
    f52 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f51)))
    f62 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f61)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))
    f43 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f42)))
    f53 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f52)))
    f63 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f62)))

    outputs = {}
    if self.training:
      outputs['flow_preds'] = None
      outputs['latent'] = [f13, f23, f33, f43, f53, f63]

    outputs['flows'] = None
    return outputs

  def encode_twelve_flows(self, inputs):

    if self.norm is not None:
      inputs = self.norm(inputs)

    img1 = inputs[:, :self.C]
    img2 = inputs[:, self.C:2 * self.C]
    img3 = inputs[:, 2 * self.C:3 * self.C]
    img4 = inputs[:, 3 * self.C:4 * self.C]
    img5 = inputs[:, 4 * self.C:5 * self.C]
    img6 = inputs[:, 5 * self.C:6 * self.C]
    img7 = inputs[:, 6 * self.C:7 * self.C]
    img8 = inputs[:, 7 * self.C:8 * self.C]
    img9 = inputs[:, 8 * self.C:9 * self.C]
    img10 = inputs[:, 9 * self.C:10 * self.C]
    img11 = inputs[:, 10 * self.C:11 * self.C]
    img12 = inputs[:, 11 * self.C:]

    img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12, _ = centralize([img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12])
    f11 = self.pconv1_2(self.pconv1_1(img1))
    f21 = self.pconv1_2(self.pconv1_1(img2))
    f31 = self.pconv1_2(self.pconv1_1(img3))
    f41 = self.pconv1_2(self.pconv1_1(img4))
    f51 = self.pconv1_2(self.pconv1_1(img5))
    f61 = self.pconv1_2(self.pconv1_1(img6))
    f71 = self.pconv1_2(self.pconv1_1(img7))
    f81 = self.pconv1_2(self.pconv1_1(img8))
    f91 = self.pconv1_2(self.pconv1_1(img9))
    f101 = self.pconv1_2(self.pconv1_1(img10))
    f111 = self.pconv1_2(self.pconv1_1(img11))
    f121 = self.pconv1_2(self.pconv1_1(img12))
    f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
    f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
    f32 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f31)))
    f42 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f41)))
    f52 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f51)))
    f62 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f61)))
    f72 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f71)))
    f82 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f81)))
    f92 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f91)))
    f102 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f101)))
    f112 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f111)))
    f122 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f121)))
    f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
    f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
    f33 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f32)))
    f43 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f42)))
    f53 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f52)))
    f63 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f62)))
    f73 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f72)))
    f83 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f82)))
    f93 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f92)))
    f103 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f102)))
    f113 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f112)))
    f123 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f122)))

    outputs = {}
    if self.training:
      outputs['flow_preds'] = None
      outputs['latent'] = [f13, f23, f33, f43, f53, f63, f73, f83, f93, f103, f113, f123]

    outputs['flows'] = None
    return outputs
