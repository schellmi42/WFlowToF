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

import argparse

from torch.utils.tensorboard import SummaryWriter

from FFNet.model import CustomFastFlowNet
from torch_utils.warp import warp_correlations_n, combine_masks
from data_ops.CBdataset import CBdataset
from data_ops.geom_ops_torch import correlation2depth_n
from data_ops.geom_ops_numpy import _max_length

from torch_utils import train_utils
from utils import plotting as plt
from torch_utils import loss_f

from ptlflow.utils.flow_utils import flow_to_rgb
import time
from datetime import datetime

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using {device} device")

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='SF', help='name for the log files')
parser.add_argument('--batch_size', '--bs', default=8, type=int)
parser.add_argument('--main_loss', default='tof', help='can be `tof` or `photo`')
parser.add_argument('--sim_loss', default='none', help='similarity loss, can be `cosine`, `cost`, `L1`, `L2` or `none`')
parser.add_argument('--sim_loss_factor', default=1e-3, type=float, help='weight of the similarity loss')
parser.add_argument('--edge_aware_loss', '--EA', action='store_true', help='activates edge-aware loss')
parser.add_argument('--edge_aware_loss_factor', '--EA_f', default=0.1, type=float, help='weight of the edge-aware loss')
parser.add_argument('--edge_aware_loss_shift', '--EA_s', default=1e4, type=float, help='shift parameter of the edge-aware loss')
parser.add_argument('--smooth_loss', action='store_true', help='activates smoothing loss')
parser.add_argument('--smooth_loss_lambda', '--sl_lambda', default=1.0, type=float, help='edge-weighting parameter of smoothing loss')
parser.add_argument('--smooth_loss_factor', default=1.0, type=float, help='weight of smoothing loss')
parser.add_argument('--aug_noise', '--noise', action='store_true', help='augment data with shot noise.')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--LRS_patience', '--lrp', default=100, type=int, help='patience for ReduceOnPlateau / step_size in StepLR')
parser.add_argument('--LRS_factor', '--lrf', default=0.5, type=float, help='multiplicative factor for lr decay')
parser.add_argument('--fix_LR_decay', action='store_true', help='switch lr decay strategy from ReduceOnPlateau to exponential LRStep')
parser.add_argument('--no_TB', action='store_true', help='deactivates tensorboard')
parser.add_argument('--no_log', action='store_true', help='deactivates all logging')
parser.add_argument('--epochs', '--n', default=1000, type=int)
parser.add_argument('--log', '--l', default='', help='use for static log directory or to restore checkpoints')
parser.add_argument('--taps', default=1, type=int, help='Can use single, two or four tap sensor input')
parser.add_argument('--lindner', action='store_true', help='uses the method of lindner et al. for motion based on reconstructed intensities.')
parser.add_argument('--mult_freq', '--mf', action='store_true', help='Use three frequencies instead of only 20MHz')
parser.add_argument('--full_epoch', action='store_true', help='include all 50 frames per epoch')
parser.add_argument('--eval_test', action='store_true')
parser.add_argument('--eval_val', action='store_true')
parser.add_argument('--plot_raw', action='store_true', help='plot all data as separate pngs without context figures for one frame per scene')
parser.add_argument('--plot_test', action='store_true')
parser.add_argument('--no_norm', action='store_true', help='disables instance normalization on network input')

args = parser.parse_args()

if args.no_log:
  args.no_TB = True

id_string = 'SF' * (not args.mult_freq) + 'MF' * args.mult_freq + '_' + str(args.taps) + 'Tap/'

model_tag = 'FFN_'

tag = model_tag + args.tag + '_' + datetime.now().strftime('%b%d_%H-%M-%S')
if not args.no_TB:
  if args.log == '':
    writer = SummaryWriter(log_dir='runs/data_v2/' + id_string + tag)
  else:
    writer = SummaryWriter(log_dir=args.log)
  log_path = writer.log_dir + '/'
else:
  writer = None
  if args.log == '':
    log_path = 'runs/data_v2/' + id_string + model_tag + tag + '/'
  else:
    log_path = args.log

if args.log == '':
  fig_path = 'figures/data_v2/' + id_string + model_tag + tag + '/'
else:
  tag = args.log.split('/')[-2]
  fig_path = args.log + '/figures/'

if not path.exists(fig_path) and ((args.no_TB and not args.no_log)):
  print('[mkdir "{}"]'.format(fig_path))
  makedirs(fig_path)

if args.plot_raw:
  test_fig_path = 'figures/' + args.log
  if not test_fig_path.endswith('/'):
    test_fig_path += '/'
  if not path.exists(test_fig_path):
    print('[mkdir "{}"]'.format(test_fig_path))
    makedirs(test_fig_path)


def train(dataloader, model, loss_fn, optimizer, plot):
  model.train()
  losses = {'photo': [], 'tof': []}
  if args.main_loss == 'flow':
    losses['flow'] = []
  for batch, data in enumerate(dataloader):
    corrs = data['corrs'][:, 0]
    depths = data['depths'][:, 0]
    tof_depths = data['tof_depths'][:, 0]
    corrs_static = data['corrs_static'][:, 0]
    valid_masks = data['masks'][:, 0]
    corrs, depths, tof_depths, corrs_static, valid_masks = corrs.to(device), depths.to(device), tof_depths.to(device), corrs_static.to(device), valid_masks.to(device)
    # Compute prediction error
    output = model(corrs)
    flow = torch.cat(output['flows'], dim=1)
    # flow = torch.clip(flow, min=-50, max=50)
    warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)

    loss_photo = loss_fn(warped_corrs, corrs_static * masks)

    masks_per_freq = combine_masks(masks) * valid_masks
    pred_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * masks_per_freq

    loss_tof = loss_fn(pred_depths, tof_depths * masks_per_freq)

    if args.main_loss == 'photo':
      total_loss = loss_photo
    elif args.main_loss == 'tof':
      abs_diff = (pred_depths - tof_depths).abs() * masks_per_freq
      pu_mask = abs_diff > 0.5 * torch.FloatTensor(max_length).to(device).view(1, -1, 1, 1)
      abs_diff[pu_mask] *= -1
      total_loss = abs_diff.mean()
    elif args.main_loss == 'flow':
      flow_GT = data['flows'][:, 0].to(device)
      total_loss = loss_fn(flow, flow_GT[:, :-2])
      losses['flow'].append(total_loss.item())

    if args.sim_loss != 'none':
      latent_vectors = model.encode(corrs_static)['latent']
      loss_sim = loss_f.latent_similarity_n(latent_vectors, type=args.sim_loss)
      total_loss += loss_sim * args.sim_loss_factor

      losses['sim'] = loss_sim.item() * args.sim_loss_factor

    if args.smooth_loss:
      flows_down = torch.cat(output['flow_preds'][0], dim=1)
      corrs_down = F.avg_pool2d(corrs, kernel_size=4, stride=4)
      loss_smooth = loss_f.edge_aware_smoothness_n(corrs_down, flows_down, lambda_param=args.smooth_loss_lambda, taps=args.taps)
      total_loss += loss_smooth * args.smooth_loss_factor

      losses['smooth'] = loss_smooth.item() * args.smooth_loss_factor

    if args.edge_aware_loss:
      loss_edge_aware = loss_f.edge_aware_loss_n(warped_corrs, corrs_static, masks_per_freq, s=args.edge_aware_loss_shift)
      total_loss += loss_edge_aware * args.edge_aware_loss_factor

      losses['edge_aware_loss'] = loss_edge_aware.item() * args.edge_aware_loss_factor

    """ for debugging
    #  catch NaN losses and stop
    if torch.isnan(total_loss).sum().item():
      print('total_loss', total_loss)
      print('tof_loss', abs_diff)
      if args.edge_aware_loss:
        print('EA loss', loss_edge_aware)
      if args.smooth_loss:
        print('smooth loss', loss_smooth)
      if args.sim_loss != 'none':
        print('sim_loss', loss_sim)
      exit()
     """
    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    losses['photo'].append(loss_photo.item())
    losses['tof'].append(loss_tof.item())

    depths_m = correlation2depth_n(corrs, frequencies_GHz)
    if plot and batch == 0 and not args.no_log:
      with torch.no_grad():
        plt.plot_correlation_warp(corrs, warped_corrs, corrs_static, masks, fig_path, epoch, writer, tag='train', taps=args.taps)
        plt.plot_flow(flow, masks, fig_path, epoch, writer, tag='train')
        plt.plot_depths_warp(depths_m, pred_depths, tof_depths, masks_per_freq, fig_path, epoch, writer, tag='train')
  return losses


def eval(dataloader, model, loss_fn, plot, track_valid_ratio=False):
  with torch.no_grad():
    model.eval()
    losses = {'photo': [], 'tof': [], 'tof_ref': [], 'photo_ref': [], 'tof_masked': [], 'tof_masked_ref': [], 'noise_error': []}
    if track_valid_ratio:
      losses['valid_ratio'] = []
    for batch, data in enumerate(dataloader):
      corrs = data['corrs']
      depths = data['depths']
      tof_depths = data['tof_depths']
      corrs_static = data['corrs_static']
      valid_masks = data['masks']
      corrs, depths, tof_depths, corrs_static, valid_masks = corrs.to(device), depths.to(device), tof_depths.to(device), corrs_static.to(device), valid_masks.to(device)
      # Compute prediction error
      output = model(corrs)
      flow = torch.cat(output['flows'], dim=1)
      # flow = torch.clip(flow, min=-50, max=50)
      warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)
      loss_photo = loss_fn(warped_corrs, corrs_static * masks)
      loss_photo_ref = loss_fn(corrs, corrs_static)

      masks_per_freq = combine_masks(masks) * valid_masks
      pred_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * masks_per_freq
      depths_m = correlation2depth_n(corrs, frequencies_GHz)
      if plot and batch == 0 and not args.no_log:
        plt.plot_correlation_warp(corrs, warped_corrs, corrs_static, masks, fig_path, epoch, writer, tag='val', taps=args.taps)
        plt.plot_flow(flow, masks, fig_path, epoch, writer, tag='val')
        plt.plot_depths_warp(depths_m, pred_depths, tof_depths, masks_per_freq, fig_path, epoch, writer, tag='val')
      loss_tof = loss_fn(pred_depths, tof_depths * masks_per_freq)
      loss_tof_ref = loss_fn(depths_m * valid_masks, tof_depths * valid_masks)

      motion_error_masks = (torch.abs(depths_m - tof_depths) > (0.4)).long()
      motion_detected = ((masks_per_freq * motion_error_masks) != 0).sum()
      if motion_detected:
        loss_tof_masked = loss_fn(pred_depths * masks_per_freq * motion_error_masks, tof_depths * masks_per_freq * motion_error_masks) * \
            np.product(depths.shape) / ((masks_per_freq * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked = torch.tensor(0)

      if motion_detected:
        loss_tof_masked_ref = loss_fn(depths_m * valid_masks * motion_error_masks, tof_depths * valid_masks * motion_error_masks) * \
            np.product(depths.shape) / ((valid_masks * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked_ref = torch.tensor(0)

      losses['photo'].append(loss_photo.item())
      losses['photo_ref'].append(loss_photo_ref.item())
      losses['tof'].append(loss_tof.item())
      losses['tof_ref'].append(loss_tof_ref.item())
      losses['tof_masked'].append(loss_tof_masked.item())
      losses['tof_masked_ref'].append(loss_tof_masked_ref.item())
      if track_valid_ratio:
        losses['valid_ratio'].append((masks_per_freq.sum(dim=1) != 0).sum().item() / np.product(depths.shape))
    tof_noise = correlation2depth_n(corrs_static, frequencies_GHz)
    noise_error = loss_fn(tof_noise * valid_masks, tof_depths * valid_masks)
    losses['noise_error'].append(noise_error.item())
    return losses


def eval_save_figs(dataloader, model, loss_fn, curr_writer=None):
  with torch.no_grad():
    model.eval()
    losses = {'photo': [], 'tof': [], 'tof_ref': [], 'valid_ratio': [], 'photo_ref': [], 'tof_masked': [], 'tof_masked_ref': [], 'noise_error': []}
    i = 0
    for batch, data in enumerate(dataloader):
      print('\r' + str(i), end='')
      i += 1
      corrs = data['corrs']
      depths = data['depths']
      tof_depths = data['tof_depths']
      corrs_static = data['corrs_static']
      valid_masks = data['masks']
      corrs, depths, tof_depths, corrs_static, valid_masks = corrs.to(device), depths.to(device), tof_depths.to(device), corrs_static.to(device), valid_masks.to(device)
      # Compute prediction error
      output = model(corrs)
      flow = torch.cat(output['flows'], dim=1)
      # flow = torch.clip(flow, min=-50, max=50)
      warped_corrs, masks = warp_correlations_n(corrs, flow, taps=args.taps)

      masks_per_freq = combine_masks(masks) * valid_masks
      pred_depths = correlation2depth_n(warped_corrs, frequencies_GHz) * masks_per_freq
      depths_m = correlation2depth_n(corrs, frequencies_GHz)
      if not args.no_log and not args.plot_raw:
        plt.plot_correlation_warp(corrs, warped_corrs, corrs_static, masks, fig_path, batch, curr_writer, tag='val_full', taps=args.taps)
        plt.plot_flow(flow, masks, fig_path, batch, curr_writer, tag='val_full')
        plt.plot_depths_warp(depths_m, pred_depths, tof_depths, masks_per_freq, fig_path, batch, curr_writer, tag='val_full')
      if args.plot_raw:
        plt.plot_depths_warp_raw(tof_motion=depths_m, pred_depths=pred_depths, tof_depths=tof_depths, masks=masks_per_freq, fig_path=test_fig_path, global_step=i)
        # plt.plot_correlations_raw(corrs_motion=corrs, corrs_warped=warped_corrs, corrs_static=corrs_static, fig_path=test_fig_path, global_step=i)
        # plt.plot_flow_raw(flow=flow, masks=masks, fig_path=test_fig_path, global_step=i)
      loss_photo = loss_fn(warped_corrs, corrs_static * masks)
      loss_tof = loss_fn(pred_depths, tof_depths * masks_per_freq)
      loss_tof_ref = loss_fn(depths_m * valid_masks, tof_depths * valid_masks)
      loss_photo_ref = loss_fn(corrs, corrs_static)

      motion_error_masks = (torch.abs(depths_m - tof_depths) > (0.4)).long()
      motion_detected = ((masks_per_freq * motion_error_masks) != 0).sum()
      if motion_detected:
        loss_tof_masked = loss_fn(pred_depths * masks_per_freq * motion_error_masks, tof_depths * masks_per_freq * motion_error_masks) * \
            np.product(depths.shape) / ((masks_per_freq * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked = torch.tensor(0)

      if motion_detected:
        loss_tof_masked_ref = loss_fn(depths_m * valid_masks * motion_error_masks, tof_depths * valid_masks * motion_error_masks) * \
            np.product(depths.shape) / ((valid_masks * motion_error_masks) != 0).sum().item()
      else:
        loss_tof_masked_ref = torch.tensor(0)

      losses['photo'].append(loss_photo.item())
      losses['photo_ref'].append(loss_photo_ref.item())
      losses['tof'].append(loss_tof.item())
      losses['tof_ref'].append(loss_tof_ref.item())
      losses['tof_masked'].append(loss_tof_masked.item())
      losses['tof_masked_ref'].append(loss_tof_masked_ref.item())
      losses['valid_ratio'].append((masks_per_freq.sum(dim=1) != 0).sum().item() / np.product(depths.shape))
      tof_noise = correlation2depth_n(corrs_static, frequencies_GHz)
      noise_error = loss_fn(tof_noise * valid_masks, tof_depths * valid_masks)
      losses['noise_error'].append(noise_error.item())
    print('\r')
  return losses

time_steps = (4 + args.mult_freq * 8) // args.taps
if args.mult_freq:
  frequencies = [20, 50, 70]
  feature_type = 'mf_c2'
else:
  frequencies = [20]
  feature_type = 'sf_c2'

feature_type_train = feature_type
if args.main_loss == 'flow':
  feature_type_train += '_sim_mov'

frequencies_GHz = np.array(frequencies, dtype=np.float32) / 1e3
max_length = _max_length(frequencies_GHz)

if not args.lindner:
  in_channels = args.taps
else:
  if args.taps == 1:
    raise ValueError('Lindner method needs taps > 1!')
  in_channels = 1

if args.no_norm:
  norm = None
else:
  norm = 'instance'

model = CustomFastFlowNet(in_channels=in_channels, norm=norm, time_steps=time_steps, taps=args.taps).to(device)
# print(model)
# print(train_utils.pytorch_total_params(model))

dataset_train = CBdataset(
    batch_size=1, set='train', frequencies=frequencies, feature_type=feature_type_train, full_scene_in_epoch=args.full_epoch, shuffle=True, taps=args.taps,
    height=512, width=512, aug_rot=True, aug_noise=bool(args.aug_noise), aug_flip=True, aug_material=True, noise_level=0.02)
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               shuffle=False, num_workers=4)

dataloader_val = CBdataset(
    batch_size=args.batch_size, set='val', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=args.full_epoch, shuffle=True, taps=args.taps,
    height=512, width=512, aug_rot=True, aug_noise=True, aug_flip=True, noise_level=0.02)

loss = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

epoch_start = 0

if path.exists(log_path + 'checkpoint.pt'):
  model, optimizer, epoch_start = train_utils.load_ckp(log_path + 'checkpoint.pt', model, optimizer)
  print('[ckpt restored]' + '\n      - ' + log_path + 'checkpoint.pt')
  print('      - epoch: ' + str(epoch_start))

if args.fix_LR_decay:
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.LRS_patience, gamma=args.LRS_factor, last_epoch=epoch_start - 1, verbose=True)
else:
  lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.LRS_factor, patience=args.LRS_patience, verbose=True)

epoch_save = 50
epochs = args.epochs
for epoch in range(epoch_start, epochs):
  print(f'Epoch {epoch}/{epochs}')
  t1 = time.time()
  l = train(dataloader_train, model, loss, optimizer, epoch % 30 == 0)
  t2 = time.time()
  print('train losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f}, ' for key, val in zip(l.keys(), l.values())]) + f', time: {t2 - t1:3f}')
  if writer is not None:
    writer.add_scalar('loss_photo/train', np.mean(l['photo']), epoch)
    writer.add_scalar('loss_tof/train', np.mean(l['tof']), epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    if args.main_loss == 'flow':
      writer.add_scalar('loss_flow/train', np.mean(l['flow']), epoch)

  l = eval(dataloader_val, model, loss, epoch % 30 == 0)
  print('val losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f}, ' for key, val in zip(l.keys(), l.values())]) + f', time: {t2 - t1:3f}')
  if writer is not None:
    writer.add_scalar('loss_photo/val', np.mean(l['photo']), epoch)
    writer.add_scalar('loss_tof/val', np.mean(l['tof']), epoch)

  # if epoch%20 == 0:
  if not args.fix_LR_decay:
    lr_scheduler.step(np.mean(l['tof']))
  else:
    lr_scheduler.step()

  if (epoch + 1) % epoch_save == 0:
    train_utils.save_ckp({'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()},
                         log_path)
    print("[ckpt saved]")
  dataset_train.on_epoch_end()
  dataloader_val.on_epoch_end()

if args.mult_freq:
  results_file = 'results_' + 'MF' + str(args.taps) + '.txt'
else:
  results_file = 'results_' + 'SF' + str(args.taps) + '.txt'

if args.plot_raw:
  full_scene_in_epoch = False
else:
  full_scene_in_epoch = True


# save more figures
if args.epochs != 0 or args.eval_val:
  print('[final plotting on validation set]')
  dataloader_val_full = CBdataset(
      batch_size=1, set='val', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=full_scene_in_epoch, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  t1 = time.time()
  l = eval_save_figs(dataloader_val_full, model, loss, curr_writer=writer)
  t2 = time.time()
  print('val losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f}, ' for key, val in zip(l.keys(), l.values())]) + f', time: {t2 - t1:3f}')
  with open(results_file, 'a') as rFile:
    rFile.write(tag + '\nval losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(l.keys(), l.values())]) + '\n')

if args.eval_test:
  if not path.exists('test_results'):
    print('[mkdir "{}"]'.format('test_results'))
    makedirs('test_results')

  print('[evaluation on test set]')
  dataloader_test_full = CBdataset(
      batch_size=1, set='test', frequencies=frequencies, feature_type=feature_type, full_scene_in_epoch=full_scene_in_epoch, shuffle=False, taps=args.taps,
      height=512, width=512, aug_rot=False, aug_noise=True, aug_flip=False, noise_level=0.02)
  print('Frames: {}'.format(len(dataloader_test_full)))
  t1 = time.time()
  if args.plot_test or args.plot_raw:
    l = eval_save_figs(dataloader_test_full, model, loss, curr_writer=None)
  else:
    l = eval(dataloader_test_full, model, loss, plot=False, track_valid_ratio=True)
  t2 = time.time()
  if not args.plot_raw:
    print('test losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f}, ' for key, val in zip(l.keys(), l.values())]) + f', time: {t2 - t1:3f}')
    with open('test_results/' + results_file, 'a') as rFile:
      rFile.write(tag + '\ntest losses: ' + ''.join([key + ': ' + f'{np.mean(val):>3f} ' for key, val in zip(l.keys(), l.values())]) + '\n')
