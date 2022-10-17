'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Portions of this code copyright (c) 2022 Visual Computing group
                of Ulm University, Germany. See the LICENSE file at the
                top-level directory of this distribution.
    Portions of this code copyright 2017, Clement Pinard
    (to large parts adapted from flownets implementation in ptlflow)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import torch
import torch.nn as nn
from torch.nn import init


def conv(in_planes, out_planes, kernel_size=3, stride=1):
  return nn.Sequential(
    nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
    nn.LeakyReLU(0.1, inplace=True)
  )


def predict_flow(in_planes, out_planes):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes):
  return nn.Sequential(
    nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
    nn.LeakyReLU(0.1, inplace=True)
  )


class FlowNetS(nn.Module):
  """ adapted FlowNetS
  Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip
  Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van
  Der Smagt, Daniel Cremers, and Thomas Brox.
  Flownet: Learning optical flow with convolutional networks.
  In Pro ceedings of the IEEE international conference on computer vision (ICCV) 2015

  """
  pretrained_checkpoints = {
    'things': 'https://github.com/hmorimitsu/ptlflow/releases/download/weights1/flownets-things-98cde14d.ckpt'
  }

  def __init__(self, input_dims=9, norm=None, div_flow=1):
    super(FlowNetS, self).__init__()
    if norm == 'instance':
      self.norm = nn.InstanceNorm2d(input_dims)
    else:
      self.norm = None
    self.div_flow = div_flow

    self.conv1   = conv(input_dims, 64, kernel_size=7, stride=2)
    self.conv2   = conv(64, 128, kernel_size=5, stride=2)
    self.conv3   = conv(128, 256, kernel_size=5, stride=2)
    self.conv3_1 = conv(256, 256)
    self.conv4   = conv(256, 512, stride=2)
    self.conv4_1 = conv(512, 512)
    self.conv5   = conv(512, 512, stride=2)
    self.conv5_1 = conv(512, 512)
    self.conv6   = conv(512, 1024, stride=2)
    self.conv6_1 = conv(1024, 1024)

    self.deconv5 = deconv(1024, 512)
    self.deconv4 = deconv(1024 + 2 * input_dims, 256)
    self.deconv3 = deconv(768 + 2 * input_dims, 128)
    self.deconv2 = deconv(384 + 2 * input_dims, 64)

    self.predict_flow6 = predict_flow(1024, 2 * input_dims)
    self.predict_flow5 = predict_flow(1024 + 2 * input_dims, 2 * input_dims)
    self.predict_flow4 = predict_flow(768 + 2 * input_dims, 2 * input_dims)
    self.predict_flow3 = predict_flow(384 + 2 * input_dims, 2 * input_dims)
    self.predict_flow2 = predict_flow(192 + 2 * input_dims, 2 * input_dims)

    self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2 * input_dims, 2 * input_dims, 4, 2, 1, bias=False)
    self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2 * input_dims, 2 * input_dims, 4, 2, 1, bias=False)
    self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2 * input_dims, 2 * input_dims, 4, 2, 1, bias=False)
    self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2 * input_dims, 2 * input_dims, 4, 2, 1, bias=False)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if m.bias is not None:
          init.uniform_(m.bias)
        init.xavier_uniform_(m.weight)

      if isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          init.uniform_(m.bias)
        init.xavier_uniform_(m.weight)
    self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

  def forward(self, x):

    if self.norm is not None:
      x = self.norm(x)

    out_conv1 = self.conv1(x)

    out_conv2 = self.conv2(out_conv1)
    out_conv3 = self.conv3_1(self.conv3(out_conv2))
    out_conv4 = self.conv4_1(self.conv4(out_conv3))
    out_conv5 = self.conv5_1(self.conv5(out_conv4))
    out_conv6 = self.conv6_1(self.conv6(out_conv5))

    flow6       = self.predict_flow6(out_conv6)
    flow6_up    = self.upsampled_flow6_to_5(flow6)
    out_deconv5 = self.deconv5(out_conv6)

    concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
    flow5       = self.predict_flow5(concat5)
    flow5_up    = self.upsampled_flow5_to_4(flow5)
    out_deconv4 = self.deconv4(concat5)
    concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
    flow4       = self.predict_flow4(concat4)
    flow4_up    = self.upsampled_flow4_to_3(flow4)
    out_deconv3 = self.deconv3(concat4)

    concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
    flow3       = self.predict_flow3(concat3)
    flow3_up    = self.upsampled_flow3_to_2(flow3)
    out_deconv2 = self.deconv2(concat3)

    concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
    flow2 = self.predict_flow2(concat2)

    outputs = {}

    if self.training:
      outputs['flow_preds'] = [flow2.float(), flow3.float(), flow4.float(), flow5.float(), flow6.float()]
      outputs['flows'] = self.div_flow * self.upsample1(flow2.float())
    else:
      outputs['flows'] = self.div_flow * self.upsample1(flow2.float())

    return outputs
