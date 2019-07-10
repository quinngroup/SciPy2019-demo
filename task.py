from pathlib import Path

import pydicom as dicom
import numpy as np
import pandas as pd

import tensorly as tl
from tensorly.decomposition import parafac, tucker, partial_tucker
from joblib import Parallel, delayed
import argparse
import tensorflow as tf
from scipy.linalg import pinv, svd
from scipy.sparse.linalg import svds

def grid_to_slices(grid, slice_shape=(116, 116), n=100):
	'''Iterate over the sub-images in a pixel array.

	Args:
		grid (2D array):
			The raw pixel array from a dicom file.
			This array aranges the slices on a 2D grid.
		slice_shape (2D shape):
			The shape of the sub-images.
		n (positive int):
			The number of sub-images to yield.

	Yields (2D array):
		The sub-images having shape `slice_shape`.
	'''
	height, width = grid.shape
	h, w = slice_shape

	# Extract at most `n` slices from the pixel array
	for i in range(0, height, h):
		for j in range(0, width, w):
			if n > 0:
				s = grid[i:i+h, j:j+w]
				yield s
			n -= 1

	# If we've extracted less than `n`, yield zeros
	while n > 0:
		s = np.zeros(slice_shape)
		yield s
		n -= 1

def load_observation(obs_dir, slice_shape=(116, 116)):
	'''Load a single observation.

	Args:
		obs_dir (path):
			The path to the observation, a directory containing dicom files.
		slice_shape (2D shape):
			The shape of the individual images within a single pixel_array.
			See `grid_to_slices`.

	Returns (5D array):
		An array with the shape
		`(sequence length, features, depth, height, width)`
	'''
	obs_dir = Path(obs_dir)
	paths = sorted(path for path in obs_dir.glob('*.dcm'))
	try:
		dcs = (dicom.read_file(str(p)) for p in paths)
		grids = (dc.pixel_array for dc in dcs)
		slices = (grid_to_slices(g, slice_shape=slice_shape) for g in grids)
		stacks = (np.stack(s) for s in slices)
		obs = np.stack(stacks)
		obs = np.expand_dims(obs, 1)
		obs = obs.astype('float32')
		# print(obs.shape)
		return obs
	except (NotImplementedError, ValueError):
		return np.nan
def metadata(data_dir):
	'''Load the metadata for the dataset.

	A file `metadata.csv` should be stored in the same directory as the data.
	This file is cross-referenced with the actual data to only include metadata
	for the samples that actually exist locally.

	Args:
	data_dir (path):
	The root directory of the dataset.

	Returns (list of dict, one for each observation):
	path (path):
	The directoy containing the dicoms for the observation.
	label (bool):
	True if the observation is in the Parkinson's class.
	'''
	# The list of metadata dicts for each observation.
	ret = []

	# Things which might exist in the data_dir that are not data.
	ignore = ['.DS_Store', 'metadata.csv']

	# Open the metadata csv
	data_dir = Path(data_dir)
	meta = pd.read_csv(data_dir / 'metadata.csv')

	meta = meta.sort_values('Subject')
	meta = meta[['Subject', 'Group']]
	meta = meta.drop_duplicates()
	meta = meta.set_index('Subject')

	data_dir = Path(data_dir)
	for path in data_dir.glob('*'):

		if path.name in ignore: continue
		# print(path.name)
		subject = int(path.name)
		label = meta.loc[subject]['Group'] == 'PD'
		label = int(label)
		paths = Path(path).glob(f'DTI_gated/*/*')
		paths = (p for p in paths if p.name not in ignore)
		ret.extend({
		'path': p,
		'label': label,
		} for p in paths)

	# print(ret)
	return ret

def calculate_transition_models(data, q):
    frames = data.shape[0]
    vertical_slices = data.shape[1]
    height = data.shape[2]
    width = data.shape[3]

    state_transitions = np.zeros(q * q * vertical_slices)
    for slice_num in range(vertical_slices):
        slice_data = data[:,slice_num,:,:]
        Y = np.zeros((height * width, frames))
        for i in range(frames):
            frame = slice_data[i].reshape(height * width)
            Y[:,i] = frame

        U, E, V_t = svds(Y, k=q)
        V = V_t.T
        C = U[:,:q]
        Ehat = np.diag(E[:q])
        Vhat = V[:,:q]
        X = Ehat.dot(Vhat.T)

        x1 = np.zeros((q, frames - 1))
        x2 = np.zeros((q, frames - 1))
        for i in range(x1.shape[1]):
            for j in range(q):
                x1[j,i] = X[j,i]
                x2[j,i] = X[j,i + 1]

        x1_pinv = pinv(x1)
        A = x2.dot(x1_pinv)
        state_transition = A.reshape(q*q)

        start_index = q * q * slice_num
        end_index =  q * q * (slice_num+1)
        state_transitions[start_index:end_index] = state_transition
    return state_transitions

def get_lds_features(i, obs_dir, q, get_svd=False):
	X = load_observation(obs_dir)
	if type(X)==np.ndarray and X.shape[0]==65:
		X = np.reshape(X, (65,100,116,116))

		transition_features = calculate_transition_models(X, q)
		print('Completed line {}'.format(i))
		return transition_features
	else:
		print('Skipped line {}'.format(i))
		transition_features = np.array([])
		return transition_features


def get_decomposed_tensor(i, obs_dir, ranks):
	tf.enable_eager_execution()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	X = load_observation(obs_dir)

	if type(X)==np.ndarray and X.shape[0]==65:
		X = np.reshape(X, (65,100,116,116))
		X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
		tl.set_backend('tensorflow')
		core, factors = tucker(X_tf, ranks=ranks, init='random', tol=10e-5, verbose=False)
		transition_features = core.numpy().flatten()
		line = transition_features
		out= line

		print('Completed line {}'.format(i))

	else:
		print('Skipped line {}'.format(i))

		out = []
	sess.close()
	return out
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', help='Enter data dir path', default='./data')
	parser.add_argument('-r','--ranks',type=int, nargs=4, help='Enter 4 vals for rank', default = [])
	parser.add_argument('-q', '--svd', type=int, help="Enter q val for svd", default=-1)
	args = parser.parse_args()
	data_dir=args.data_dir
	ranks =args.ranks
	q = args.svd

	td_frame = pd.DataFrame(metadata(data_dir))
	td_frame['features'] = [[] for _ in range(len(td_frame))]
	td_frame.features.astype('object')

	lds_frame = td_frame.copy()

	if ranks != []:
		td_frame['features'] = Parallel(n_jobs=-1, verbose=51)(delayed(get_decomposed_tensor)(i=i, obs_dir=x, ranks=ranks) for i, x in enumerate(td_frame['path']))
		td_frame.to_pickle(path=f'tensor_features_{ranks[0]}_{ranks[1]}_{ranks[2]}_{ranks[3]}.pkl')
	if q != -1:
		lds_frame['features'] = Parallel(n_jobs=-1, verbose=51)(delayed(get_lds_features)(i=i, obs_dir=x, q=q) for i, x in enumerate(lds_frame['path']))
		lds_frame.to_pickle(path=f'LDS_features_{q}.pkl')
