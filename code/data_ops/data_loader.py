'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \copyright Copyright (c) 2022 Visual Computing group of Ulm University,
                Germany. See the LICENSE file at the top-level directory of
                this distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
import numpy as np
import imageio

DATA_PATH = 'data/'

S_train = {'path': DATA_PATH,
           'list': 'train_motion.txt',
           'frequencies': ['20', '50', '70'],
           'shape': [600, 600, 4],
           'num_frames': 50,
           'num_scenes': 116}

S_val = {'path': DATA_PATH,
         'list': 'val_motion.txt',
         'frequencies': ['20', '50', '70'],
         'shape': [600, 600, 4],
         'num_frames': 50,
         'num_scenes': 13}

S_test = {'path': DATA_PATH,
          'list': 'test_motion.txt',
          'frequencies': ['20', '50', '70'],
          'shape': [600, 600, 4],
          'num_frames': 50,
          'num_scenes': 13}


def load_filenames(data_set):
  """ Loads data of the datasets from publications of Agresti et al.
  Args:
    data_set: `str`, can be `'train', 'val', 'test'`.
  Returns:
    filenames: `float`, shape `[S, 50]`.
  """

  if 'train' in data_set:
    Set = S_train
  elif 'val' in data_set:
    Set = S_val
  elif 'test' in data_set:
    Set = S_test
  scenes = []
  with open(Set['path'] + Set['list'], 'r') as inFile:
    for line in inFile:
      scenes.append(line.replace('\n', ''))
  frames = []
  for scene in scenes:
    frames.append([Set['path'] + scene + '/' + str(i).zfill(3) + '_render_' for i in range(50)])
  return np.array(frames, dtype=str)


def load_batch(files, frequencies=None, slice_id=None):
  """ Loads data of the datasets from Cornell dataset
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` [0,1,2], to slice the color channel for different materials.

  Returns:
    depths: `float`, shape `[S, H, W]`.
    tof_depths: `float`, shape `[S, F, H, W]`.
    correlations: `float`, shape `[S, F, 4, H, W]`.
    amplitudes: `float`, shape `[S, F, H, W]`.
    intensities: `float`, shape `[S, F, H, W]`.
  """
  depths = []
  tof_depths = []
  correlations = []
  amplitudes = []
  intensities = []
  if frequencies is None:
    frequencies = ['20', '50', '70']
  for frame in files:
    if slice_id is None:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI'))
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI') for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame + str(freq) + 'MHz_phase0.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase1.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase2.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')]
           for freq in frequencies]
      )
    else:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI')[:, :, slice_id])
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI')[:, :, slice_id] for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame + str(freq) + 'MHz_phase0.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase1.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase2.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')[:, :, slice_id]]
           for freq in frequencies]
      )
  depths = np.array(depths, dtype=np.float32)
  tof_depths = np.array(tof_depths, dtype=np.float32)
  correlations = np.array(correlations, dtype=np.float32)
  amplitudes = 0.5 * np.sqrt((correlations[:, :, 0] - correlations[:, :, 2])**2 + \
                             (correlations[:, :, 3] - correlations[:, :, 1])**2)
  intensities = np.mean(correlations, axis=2)

  return depths, tof_depths, correlations, amplitudes, intensities


def load_batch_correlation(files, frequencies=None, slice_id=None):
  """ Loads data of the datasets from Cornell dataset
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` [0,1,2], to slice the color channel for different materials.

  Returns:
    correlations: `float`, shape `[S, F, 4, H, W]`.
  """
  correlations = []
  if frequencies is None:
    frequencies = ['20', '50', '70']
  for frame in files:
    if slice_id is None:
      correlations.append(
          [[imageio.imread(frame + str(freq) + 'MHz_phase0.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase1.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase2.hdr', format='HDR-FI'),
            imageio.imread(frame + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')]
           for freq in frequencies]
      )
    else:
      correlations.append(
          [[imageio.imread(frame + str(freq) + 'MHz_phase0.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase1.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase2.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')[:, :, slice_id]]
           for freq in frequencies]
      )
  correlations = np.array(correlations, dtype=np.float32)

  return correlations


def load_batch_correlation_motion(files, frequencies=None, slice_id=None):
  """ Loads data of the datasets from Cornell dataset
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` [0,1,2], to slice the color channel for different materials.

  Returns:
    correlations: `float`, shape `[S, F, 4, H, W]`.
  """
  correlations = []
  if frequencies is None:
    frequencies = ['20', '50', '70']
  offsets = range(0, len(frequencies) * 4, 4)[::-1]
  for frame in files:
    curr_timestamp = int(frame.split('/')[-1][:3])
    cut = len(frame.split('/')[-1])
    if slice_id is None:
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 3, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 2, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')]
           for freq, offset in zip(frequencies, offsets)]
      )
    else:
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 3, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 2, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')[:, :, slice_id]]
           for freq, offset in zip(frequencies, offsets)]
      )
  correlations = np.array(correlations, dtype=np.float32)

  return correlations


def load_batch_motion(files, frequencies=None, slice_id=None, taps=1):
  """ Loads data of the datasets from Cornell dataset with motion simlated as n tap sensor.
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` can be `[0,1,2]`, to slice the color channel for different materials.
    taps: `int`, can be `[1, 2, 4]`.

  Returns:
    depths: `float`, shape `[S, H, W]`.
    tof_depths: `float`, shape `[S, F, H, W]`, without motion.
    correlations: `float`, shape `[S, F, 4, H, W]`.
    amplitudes: `float`, shape `[S, F, H, W]`.
    intensities: `float`, shape `[S, F, H, W]`.
  """
  if taps == 1:
    return load_batch_motion_single_tap(files, frequencies, slice_id)
  elif taps == 2:
    return load_batch_motion_2tap(files, frequencies, slice_id)
  elif taps == 4:
    return load_batch_motion_4tap(files, frequencies, slice_id)


def load_batch_motion_single_tap(files, frequencies=None, slice_id=None):
  """ Loads data of the datasets from Cornell dataset with motion simlated as single tap sensor.
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` [0,1,2], to slice the color channel for different materials.

  Returns:
    depths: `float`, shape `[S, H, W]`.
    tof_depths: `float`, shape `[S, F, H, W]`, without motion.
    correlations: `float`, shape `[S, F, 4, H, W]`.
    amplitudes: `float`, shape `[S, F, H, W]`.
    intensities: `float`, shape `[S, F, H, W]`.
  """
  depths = []
  tof_depths = []
  correlations = []
  amplitudes = []
  intensities = []
  if frequencies is None:
    frequencies = ['20', '50', '70']
  offsets = range(0, len(frequencies) * 4, 4)[::-1]
  for frame in files:
    curr_timestamp = int(frame.split('/')[-1][:3])
    cut = len(frame.split('/')[-1])
    if slice_id is None:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI'))
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI') for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 3, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 2, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')]
           for freq, offset in zip(frequencies, offsets)]
      )
    else:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI')[:, :, slice_id])
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI')[:, :, slice_id] for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 3, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 2, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')[:, :, slice_id]]
           for freq, offset in zip(frequencies, offsets)]
      )
  depths = np.array(depths, dtype=np.float32)
  tof_depths = np.array(tof_depths, dtype=np.float32)
  correlations = np.array(correlations, dtype=np.float32)
  amplitudes = 0.5 * np.sqrt((correlations[:, :, 0] - correlations[:, :, 2])**2 + \
                             (correlations[:, :, 3] - correlations[:, :, 1])**2)
  intensities = np.mean(correlations, axis=2)

  return depths, tof_depths, correlations, amplitudes, intensities


def load_batch_motion_2tap(files, frequencies=None, slice_id=None):
  """ Loads data of the datasets from Cornell dataset with motion simulated as 2-tap sensor.
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` [0,1,2], to slice the color channel for different materials.

  Returns:
    depths: `float`, shape `[S, H, W]`.
    tof_depths: `float`, shape `[S, F, H, W]`, without motion.
    correlations: `float`, shape `[S, F, 4, H, W]`.
    amplitudes: `float`, shape `[S, F, H, W]`.
    intensities: `float`, shape `[S, F, H, W]`.
  """
  depths = []
  tof_depths = []
  correlations = []
  amplitudes = []
  intensities = []
  if frequencies is None:
    frequencies = ['20', '50', '70']
  offsets = range(0, len(frequencies) * 2, 2)[::-1]
  for frame in files:
    curr_timestamp = int(frame.split('/')[-1][:3])
    cut = len(frame.split('/')[-1])
    if slice_id is None:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI'))
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI') for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')]
           for freq, offset in zip(frequencies, offsets)]
      )
    else:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI')[:, :, slice_id])
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI')[:, :, slice_id] for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset - 1, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')[:, :, slice_id]]
           for freq, offset in zip(frequencies, offsets)]
      )
  depths = np.array(depths, dtype=np.float32)
  tof_depths = np.array(tof_depths, dtype=np.float32)
  correlations = np.array(correlations, dtype=np.float32)
  amplitudes = 0.5 * np.sqrt((correlations[:, :, 0] - correlations[:, :, 2])**2 + \
                             (correlations[:, :, 3] - correlations[:, :, 1])**2)
  intensities = np.mean(correlations, axis=2)

  return depths, tof_depths, correlations, amplitudes, intensities


def load_batch_motion_4tap(files, frequencies=None, slice_id=None):
  """ Loads data of the datasets from Cornell dataset with motion simulated as 4-tap sensor.
  Args:
    files: `list`of `str`.
    frequencies: `list` of `str`, must be subset of the frequencies in the
      respective dataset.
      If `None` loads all.
    slice_ids: an `int` [0,1,2], to slice the color channel for different materials.

  Returns:
    depths: `float`, shape `[S, H, W]`.
    tof_depths: `float`, shape `[S, F, H, W]`, without motion.
    correlations: `float`, shape `[S, F, 4, H, W]`.
    amplitudes: `float`, shape `[S, F, H, W]`.
    intensities: `float`, shape `[S, F, H, W]`.
  """
  depths = []
  tof_depths = []
  correlations = []
  amplitudes = []
  intensities = []
  if frequencies is None:
    frequencies = ['20', '50', '70']
  offsets = range(0, len(frequencies), 1)[::-1]
  for frame in files:
    curr_timestamp = int(frame.split('/')[-1][:3])
    cut = len(frame.split('/')[-1])
    if slice_id is None:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI'))
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI') for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI'),
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')]
           for freq, offset in zip(frequencies, offsets)]
      )
    else:
      depths.append(imageio.imread(frame + 'depth.hdr', format='HDR-FI')[:, :, slice_id])
      tof_depths.append(
          [imageio.imread(frame + str(freq) + 'MHz_ToF.hdr', format='HDR-FI')[:, :, slice_id] for freq in frequencies]
      )
      correlations.append(
          [[imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase0.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase1.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase2.hdr', format='HDR-FI')[:, :, slice_id],
            imageio.imread(frame[:-cut] + str(max(curr_timestamp - offset, 0)).zfill(3) + '_render_' + str(freq) + 'MHz_phase3.hdr', format='HDR-FI')[:, :, slice_id]]
           for freq, offset in zip(frequencies, offsets)]
      )
  depths = np.array(depths, dtype=np.float32)
  tof_depths = np.array(tof_depths, dtype=np.float32)
  correlations = np.array(correlations, dtype=np.float32)
  amplitudes = 0.5 * np.sqrt((correlations[:, :, 0] - correlations[:, :, 2])**2 + \
                             (correlations[:, :, 3] - correlations[:, :, 1])**2)
  intensities = np.mean(correlations, axis=2)

  return depths, tof_depths, correlations, amplitudes, intensities
