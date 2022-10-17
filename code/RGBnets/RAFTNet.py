'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Portions of this code copyright (c) 2022 Visual Computing group
                of Ulm University, Germany. See the LICENSE file at the
                top-level directory of this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from ptlflow.models.raft import raft as RAFT

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomRAFTNet(RAFT.RAFT):
  def __init__(self,
               args,
               in_channels=1,
               time_steps=4,
               taps=1,
               convert='RGB'):
    super(CustomRAFTNet, self).__init__(args=args)
    self.forward_pair = super(CustomRAFTNet, self).forward
    self.time_steps = time_steps
    self.taps = taps
    self.convert = convert
    self.in_channels = in_channels

  def twotap_reordering(self, inputs):
    if self.time_steps == 2:
      inputs = torch.cat([inputs[:, [0, 2]], inputs[:, [1, 3]]], dim=1)
    elif self.time_steps == 6:
      inputs = torch.cat([inputs[:, [0, 2]], inputs[:, [1, 3]],
                          inputs[:, [4, 6]], inputs[:, [5, 7]],
                          inputs[:, [8, 10]], inputs[:, [9, 11]]], dim=1)
    return inputs

  def convert_input_data(self, data):
    if self.convert == 'RGB':
      max = 462848.0
      data /= (max / 255)
      data = torch.stack([data, data, data], dim=2)
    return {'images': data}

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

    inputs = inputs.view(B, int(self.time_steps), -1, H, W)

    if self.time_steps == 2:
      curr_input = torch.stack([inputs[:, 1, -1], inputs[:, 0, 0]], dim=1)
      output = self.forward_pair(self.convert_input_data(curr_input))
      output['flows'] = [output['flows'][:, 0]]
    else:
      output = {'flows': []}
      if self.training:
        output['flow_preds'] = []
      for i in range(self.time_steps - 1):
        curr_input = torch.stack([inputs[:, -1, -1], inputs[:, i, 0]], dim=1)
        curr_output = self.forward_pair(self.convert_input_data(curr_input))
        output['flows'].append(curr_output['flows'][:, 0])
        if self.training:
          output['flow_preds'].append(curr_output['flow_preds'])

      # transpose lists
      if self.training:
        output['flow_preds'] = [list(x) for x in list(zip(*output['flow_preds']))]

    return output
