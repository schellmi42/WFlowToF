'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from os import path, makedirs
# filter runtime warning from division by zero
import warnings
warnings.filterwarnings('ignore')
import argparse

from torch.utils.tensorboard import SummaryWriter

from CFNet.model import CoarseFineCNN
from FFNet.model import CustomFastFlowNet
from torch_utils.warp import warp_correlations_n, combine_masks
from data_ops.CBdataset import CBdataset
from data_ops.geom_ops_torch import correlation2depth_n
from data_ops.geom_ops_numpy import _max_length

from torch_utils import train_utils
from utils import plotting as plt
from torch_utils import loss_f

import time
from datetime import datetime

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='SF', help='name for the log files')
parser.add_argument('--bs', default=4)
parser.add_argument('--main_loss', default='L1', help='L1-Loss for L_ToF')
parser.add_argument('--aug_noise', '--noise', action='store_true', help='augment data with shot noise.')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--LRS_patience', '--lrp', default=100, type=int, help='patience for ReduceOnPlateau / step_size in StepLR')
parser.add_argument('--LRS_factor', '--lrf', default=0.1, type=float, help='multiplicative factor for lr decay')
parser.add_argument('--fix_LR_decay', action='store_true', help='switch lr decay strategy from ReduceOnPlateau to exponential LRStep')
parser.add_argument('--no_TB', action='store_true', help='deactivates tensorboard')
parser.add_argument('--no_log', action='store_true', help='deactivates all logging')
parser.add_argument('--epochs', '--n', default=1000, type=int)
parser.add_argument('--log', '--l', default='', help='use for static log directory or to restore checkpoints')
parser.add_argument('--log_OF', '--l_OF', default='', help='path to pre-trained OF network weights')
parser.add_argument('--taps', default=1, type=int, help='Can use single, two or four tap sensor input')
parser.add_argument('--mult_freq', '--mf', action='store_true', help='Use three frequencies instead of only 20MHz')
parser.add_argument('--eval_test', action='store_true')
parser.add_argument('--eval_val', action='store_true')
parser.add_argument('--plot_test', action='store_true')
parser.add_argument('--no_norm', action='store_true', help='disables instance normalization on network input')
parser.add_argument('--PU', action='store_true', help='phase unwrapping for warped tof depths')
parser.add_argument('--PU_v0', action='store_true', help='phase unwrapping for warped tof depths, non iterative')
parser.add_argument('--res_depth_id', default=-1, type=int, help='id of residual depth for CFN, defaults to highest frequency(-1)')
parser.add_argument('--plot_raw', action='store_true', help='plot all data as separate pngs without context figures for one frame per scene')
args = parser.parse_args()

if args.PU_v0:
  args.PU = True
  from data_ops.geom_ops_torch import phase_unwrapping
else:
  from data_ops.geom_ops_torch import phase_unwrapping_iterative as phase_unwrapping

if args.log_OF == '':
  OF = False
else:
  OF = True

if args.plot_raw:
  test_fig_path = 'figures/' + args.log
  if not test_fig_path.endswith('/'):
    test_fig_path += '/'
  if not path.exists(test_fig_path):
    print('[mkdir "{}"]'.format(test_fig_path))
    makedirs(test_fig_path)


def compute_agresti_features(correlations, tof_depths):
  if args.mult_freq:
    if args.PU:
      tof_depths = phase_unwrapping(tof_depths, max_length_torch)

    B, N, H, W = correlations.size()
    C = N // 4
    correlations = correlations.view(B, C, 4, H, W)
    amplitudes = 0.5 * torch.sqrt((correlations[:, :, 0] - correlations[:, :, 2])**2 + \
                                  (correlations[:, :, 3] - correlations[:, :, 1])**2)
    features = torch.stack(
      [
        tof_depths[:, 2],  # tof depth at 70 MHz
        tof_depths[:, 0] - tof_depths[:, 2],  # difference tof depths at 20MHz and at 70MHz
        tof_depths[:, 1] - tof_depths[:, 2],  # difference tof depths at 50MHz and at 70MHz
        (amplitudes[:, 0] / amplitudes[:, 2]) - 1,  # centered amplitudes ratios between 20MHz and 70 MHz
        (amplitudes[:, 1] / amplitudes[:, 2]) - 1,  # centered amplitudes ratios between 50MHz and 70 MHz
      ], axis=1)
    features = torch.nan_to_num(features, nan=0, posinf=0, neginf=0)
  else:
    features = tof_depths
  return features, tof_depths


if args.no_log:
  args.no_TB = True

id_string = 'SF' * (not args.mult_freq) + 'MF' * args.mult_freq + '_' + str(args.taps) + 'Tap/'
tag = 'FFN+CFN_' + args.tag + '_' + datetime.now().strftime('%b%d_%H-%M-%S')

if not args.no_TB:
  if args.log == '':
    writer = SummaryWriter(log_dir='runs/denoising/' + id_string + tag)
  else:
    writer = SummaryWriter(log_dir=args.log)
  log_path = writer.log_dir + '/'
else:
  writer = None
  if args.log == '':
    log_path = 'runs/denoising/' + id_string + 'FFN+CFN' + tag + '/'
  else:
    log_path = args.log

if args.log == '':
  fig_path = 'figures/denoising/' + id_string + 'FFN+CFN' + tag + '/'
else:
  if not tag.endswith('/'):
    tag += '/'
  tag = args.log.split('/')[-2]
  fig_path = args.log + '/figures/'

if not path.exists(fig_path) and ((args.no_TB and not args.no_log) or args.plot_test):
  makedirs(fig_path)


def train(dataloader, OF_model, model, loss_fn, optimizer, plot):
  model.train()
  losses = {'depth': [], 'tof_ref': [], 'valid_ratio': []}
  for batch, data in enumerate(dataloader):
    corrs = data['corrs'][:, 0]
    depths = data['depths'][:, 0]
    input_masks = data['masks'][:, 0]
    corrs, depths, input_masks = corrs.to(device), depths.to(device), input_masks.to(device)
    # Compute prediction error
    tof_depths_motion = correlation2depth_n(corrs, frequencies_GHz)
    with torch.no_grad():
      if OF:
        output = OF_model(corrs)
        flow = torch.cat(output['flows'], dim=1)
        # flow = torch.clip(flow, min=-50, max=50)
        warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)
        valid_masks = combine_masks(masks, reduce_all=True) * input_masks
        pred_tof_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * valid_masks
        input_features, pred_tof_depths_PU = compute_agresti_features(warped_corrs, pred_tof_depths)
      else:
        input_features, pred_tof_depths_PU = compute_agresti_features(corrs, tof_depths_motion)
        valid_masks = input_masks

    pred_depths = model(input_features * valid_masks, pred_tof_depths_PU[:, args.res_depth_id, None])

    loss_tof_coarse = loss_fn(pred_depths[0] * valid_masks, depths * valid_masks)
    loss_tof_fine = loss_fn(pred_depths[1] * valid_masks, depths * valid_masks)
    loss_total = loss_tof_coarse + loss_tof_fine

    # Backpropagation
    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()
    tof_depths_motion = correlation2depth_n(corrs, frequencies_GHz)
    if args.mult_freq:
      tof_depths_motion = phase_unwrapping(tof_depths_motion, max_length_torch)
    loss_tof_ref = loss_fn(tof_depths_motion[:, -1] * input_masks, depths[:, 0] * input_masks)

    losses['depth'].append(loss_tof_fine.item())
    losses['tof_ref'].append(loss_tof_ref.item())
    losses['valid_ratio'].append((input_masks.sum(dim=1) != 0).sum().item() / np.product(depths.shape))

    if plot and batch == 0 and not args.no_log:
      with torch.no_grad():
        plt.plot_depths_denoising(tof_depths_motion, pred_tof_depths_PU, pred_depths, depths, valid_masks, fig_path, epoch, writer, tag='train')
  return losses


def eval(dataloader, OF_model, model, loss_fn, plot):
  with torch.no_grad():
    model.eval()
    losses = {'depth': [], 'tof_ref': [], 'tof_warp': [], 'valid_ratio': []}
    for batch, data in enumerate(dataloader):
      corrs = data['corrs']
      depths = data['depths']
      input_masks = data['masks']
      corrs, depths, input_masks = corrs.to(device), depths.to(device), input_masks.to(device)
      # Compute prediction error
      tof_depths_motion = correlation2depth_n(corrs, frequencies_GHz)
      if OF:
        output = OF_model(corrs)
        flow = torch.cat(output['flows'], dim=1)
        # flow = torch.clip(flow, min=-50, max=50)
        warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)
        valid_masks = combine_masks(masks, reduce_all=True) * input_masks
        pred_tof_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * valid_masks
        input_features, pred_tof_depths_PU = compute_agresti_features(warped_corrs, pred_tof_depths)
      else:
        input_features, pred_tof_depths_PU = compute_agresti_features(corrs, tof_depths_motion)
        valid_masks = input_masks
      pred_depths = model(input_features * valid_masks, pred_tof_depths_PU[:, args.res_depth_id, None])

      loss_tof = loss_fn(pred_depths[1] * valid_masks, depths * valid_masks)

      if args.mult_freq:
        tof_depths_motion = phase_unwrapping(tof_depths_motion, max_length_torch)
      loss_tof_ref = loss_fn(tof_depths_motion[:, -1] * input_masks, depths[:, 0] * input_masks)
      losses['depth'].append(loss_tof.item())
      losses['tof_ref'].append(loss_tof_ref.item())
      loss_tof_warp = loss_fn(pred_tof_depths_PU[:, -1] * valid_masks, depths[:, 0] * valid_masks)
      losses['tof_warp'].append(loss_tof_warp.item())
      losses['valid_ratio'].append((valid_masks.sum(dim=1) != 0).sum().item() / np.product(depths.shape))

      if plot and batch == 0 and not args.no_log:
          plt.plot_depths_denoising(tof_depths_motion, pred_tof_depths_PU, pred_depths, depths, valid_masks, fig_path, epoch, writer, tag='val')
    return losses


def eval_save_figs(dataloader, OF_model, model, loss_fn, curr_writer=None):
  with torch.no_grad():
    model.eval()
    losses = {'depth': [], 'tof_ref': [], 'tof_warp': [], 'valid_ratio': []}
    i = 0
    for batch, data in enumerate(dataloader):
      i += 1
      corrs = data['corrs']
      depths = data['depths']
      input_masks = data['masks']
      corrs, depths, input_masks = corrs.to(device), depths.to(device), input_masks.to(device)
      # Compute prediction error
      tof_depths_motion = correlation2depth_n(corrs, frequencies_GHz)
      if OF:
        output = OF_model(corrs)
        flow = torch.cat(output['flows'], dim=1)
        # flow = torch.clip(flow, min=-50, max=50)
        warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)
        valid_masks = combine_masks(masks, reduce_all=True) * input_masks
        pred_tof_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * valid_masks
        input_features, pred_tof_depths_PU = compute_agresti_features(warped_corrs, pred_tof_depths)
      else:
        input_features, pred_tof_depths_PU = compute_agresti_features(corrs, tof_depths_motion)
        valid_masks = input_masks
      pred_depths = model(input_features * valid_masks, pred_tof_depths_PU[:, args.res_depth_id, None])

      loss_tof = loss_fn(pred_depths[1] * valid_masks, depths * valid_masks)

      if args.mult_freq:
        tof_depths_motion = phase_unwrapping(tof_depths_motion, max_length_torch)
      loss_tof_ref = loss_fn(tof_depths_motion[:, -1] * input_masks, depths[:, 0] * input_masks)
      losses['tof_ref'].append(loss_tof_ref.item())
      loss_tof_warp = loss_fn(pred_tof_depths_PU[:, -1] * valid_masks, depths[:, 0] * valid_masks)
      losses['tof_warp'].append(loss_tof_warp.item())
      losses['depth'].append(loss_tof.item())
      losses['valid_ratio'].append((valid_masks.sum(dim=1) != 0).sum().item() / np.product(depths.shape))

      if not args.no_log and not args.plot_raw:
        plt.plot_depths_denoising(tof_depths_motion, pred_tof_depths_PU, pred_depths, depths, valid_masks, fig_path, batch, curr_writer, tag='val_full')
      if args.plot_raw:
        plt.plot_depths_denoising_raw(tof_depths_input=tof_depths_motion, tof_depths_warped=pred_tof_depths_PU, pred_depths=pred_depths, GT_depths=depths, masks=valid_masks, fig_path=test_fig_path, global_step=i)
  return losses


time_steps = (4 + args.mult_freq * 8) / args.taps
if args.mult_freq:
  frequencies = [20, 50, 70]
  feature_type = 'mf_c2'
  num_features = 5
else:
  frequencies = [20]
  feature_type = 'sf_c'
  num_features = 1

in_channels = args.taps

if args.no_norm:
  norm = None
else:
  norm = 'instance'

frequencies_GHz = np.array(frequencies, dtype=np.float32) / 1e3
max_length = _max_length(frequencies_GHz)
max_length_torch = torch.FloatTensor(max_length).to(device)

model = CoarseFineCNN(input_feature_size=num_features, num_output_features=1).to(device)
if OF:
  OF_model = CustomFastFlowNet(in_channels=in_channels, norm=norm, time_steps=time_steps, taps=args.taps).to(device)
else:
  OF_model = None
# print(model)
# print(train_utils.pytorch_total_params(model))


dataset_train = CBdataset(
    batch_size=1, set='train', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=False, shuffle=True, taps=args.taps,
    height=512, width=512, aug_rot=True, aug_noise=args.aug_noise, aug_noise_tof=args.aug_noise, aug_flip=True, aug_material=True, noise_level=0.02)
# parallel dataloader for speedup during training
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs,
                                               shuffle=False, num_workers=4)

dataset_val = CBdataset(
    batch_size=8, set='val', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=False, shuffle=True, taps=args.taps,
    height=512, width=512, aug_rot=True, aug_noise=True, aug_flip=True, noise_level=0.02)
# dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=8,
#                                              shuffle=False, num_workers=4)

loss = nn.L1Loss()


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
epoch_start = 0

if path.exists(log_path + 'checkpoint.pt'):
  model, optimizer, epoch_start = train_utils.load_ckp(log_path + 'checkpoint.pt', model, optimizer)
  print('[ckpt restored]' + '\n      - ' + log_path + 'checkpoint.pt')
  print('      - epoch: ' + str(epoch_start))

if path.exists(args.log_OF + 'checkpoint.pt'):
  OF_model, _, loaded_epoch = train_utils.load_ckp(args.log_OF + 'checkpoint.pt', OF_model)
  print('[ckpt restored]' + '\n      - ' + args.log_OF + 'checkpoint.pt')
  print('      - epoch: ' + str(loaded_epoch))

if args.fix_LR_decay:
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.LRS_patience, gamma=args.LRS_factor, last_epoch=epoch_start - 1, verbose=True)
else:
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.LRS_factor, patience=args.LRS_patience, verbose=True)

epoch_save = 50
epochs = args.epochs
for epoch in range(epoch_start, epochs):
  print(f'Epoch {epoch}/{epochs}')
  t1 = time.time()
  lt = train(dataloader_train, OF_model, model, loss, optimizer, epoch % 30 == 0)
  t2 = time.time()
  print('train  loss: ' + str(np.mean(lt['depth'])) + ', time: ' + str(t2 - t1))
  if writer is not None:
    writer.add_scalar('loss_tof/train', np.mean(lt['depth']), epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

  lt = eval(dataset_val, OF_model, model, loss, epoch % 30 == 0)
  print('val  loss: ' + str(np.mean(lt['depth'])))
  if writer is not None:
    writer.add_scalar('loss_tof/val', np.mean(lt['depth']), epoch)

  if not args.fix_LR_decay:
    lr_scheduler.step(np.mean(lt['depth']))
  else:
    lr_scheduler.step()

  if (epoch + 1) % epoch_save == 0:
    train_utils.save_ckp({'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()},
                         log_path)
    print("[ckpt saved]")
  dataset_train.on_epoch_end()
  dataset_val.on_epoch_end()

if args.mult_freq:
  results_file = 'results_' + 'MF' + str(args.taps) + '_denoising.txt'
else:
  results_file = 'results_' + 'SF' + str(args.taps) + '_denoising.txt'

if args.plot_raw:
  full_scene_in_epoch = False
else:
  full_scene_in_epoch = True

# save more figures
if args.epochs != 0 or args.eval_val:
  print('[final plotting on validation set]')
  dataset_val_full = CBdataset(
      batch_size=1, set='val', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=full_scene_in_epoch, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  lt = eval_save_figs(dataset_val_full, OF_model, model, loss, curr_writer=writer)

  print('val loss:' + str(np.mean(lt['depth'])))
  with open(results_file, 'a') as rFile:
    rFile.write(tag + '\nval losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(lt.keys(), lt.values())]) + '\n')

if args.eval_test:
  if not path.exists('test_results'):
    print('[mkdir "{}"]'.format('test_results'))
    makedirs('test_results')

  print('[evaluation on test set]')
  dataloader_test_full = CBdataset(
      batch_size=1, set='test', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=full_scene_in_epoch, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  print('Frames: {}'.format(len(dataloader_test_full)))
  if args.plot_test or args.plot_raw:
    lt = eval_save_figs(dataloader_test_full, OF_model, model, loss, curr_writer=None)
  else:
    lt = eval(dataloader_test_full, OF_model, model, loss, plot=False)

  if not args.plot_raw:
    print('test loss:' + str(np.mean(lt['depth'])))
    with open('test_results/' + results_file, 'a') as rFile:
      rFile.write(tag + '\ntest losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(lt.keys(), lt.values())]) + '\n')
