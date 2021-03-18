import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import linen as nn
from flax import optim, serialization
from IPython import embed
from jax import jit, random
from skimage import io, transform
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import common

b = 18.17
f = 286.29
min_depth = 4000.
max_depth = 40000.

_max_disp = (b * f) / min_depth
_min_disp = (b * f) / max_depth

# Load in all datasets
# renders_path = '/home/bryan/work/Replica-Dataset/renders'
renders_path = '/home/bryan/Dropbox/School/W21/CS 231A/replica_renders'
target_size = 12 * 36


def _load_image(path):
    return io.imread(path)


def _load_frame(root, frame_num):
    num = f"{frame_num:06d}"
    depth = _load_image(os.path.join(root, f"depth{num}.png"))
    left = _load_image(os.path.join(root, f"frame{num}_left.jpg"))
    right = _load_image(os.path.join(root, f"frame{num}_right.jpg"))
    return {
        'left': left,
        'right': right,
        'depth': depth,
    }


def _depth_to_disparity(depth):
    depth = jnp.clip(depth, a_min=min_depth)
    _disparity = (b * f) / depth

    def _scale(x):
        return (x - _min_disp) / (_max_disp - _min_disp)

    disparity = _scale(_disparity) * common.max_disp
    return disparity


def _crop(img):
    h, w, *_ = img.shape
    pad = (w - h) // 2
    return img[:, pad:-pad]


def _process_frame(frame):
    _left = _crop(frame['left'])
    _right = _crop(frame['right'])
    _depth = _crop(frame['depth'])

    _left = jax.image.resize(_left, (target_size, target_size, 3),
                             method="bilinear")
    _right = jax.image.resize(_right, (target_size, target_size, 3),
                              method="bilinear")

    _left = _left / 255.
    _right = _right / 255.

    _depth = jax.image.resize(_depth, (target_size, target_size),
                              method="bilinear")
    _disparity = _depth_to_disparity(_depth)
    _disparity = _disparity[..., jnp.newaxis]

    return {
        'left': np.array(_left),
        'right': np.array(_right),
        'disparity': np.array(_disparity),
    }


class StereoDepthDataset(Dataset):
    def __init__(self, root, load_count=None):
        self.root = root
        self.num_folder_frames = len(os.listdir(root)) // 3
        self.num_frames = self.num_folder_frames if load_count == None else load_count
        if self.num_frames > self.num_folder_frames:
            raise ValueError(
                f"Attempting to load {self.num_frames} but only {self.num_folder_frames} found."
            )
        print(
            f"Loading {self.num_frames}/{self.num_folder_frames} frames found in {self.root}"
        )

    def __len__(self):
        return self.num_frames

    def __getitem__(self, frame_num):
        frame = _load_frame(self.root, frame_num)
        return _process_frame(frame)


dataset_categories = [('apartment', 3), ('office', 5), ('room', 3),
                      ('hotel', 1), ('frl_apartment', 6)]
assert (sum([count for name, count in dataset_categories]) == 18)

# Amount of frames to load from each scene
load_count = 100
dataset_names = []

for name, count in dataset_categories:
    for i in range(count):
        _name = f"{name}_{i}"
        dataset_names.append(_name)


def _load_datasets(names):
    datasets = []
    for name in names:
        print(f"Creating dataset for {name}")
        root = os.path.join(renders_path, name)
        if not os.path.exists(root):
            print(f"Dataset root {root} does not exist.")
            # raise ValueError(f"Root {root} does not exist.")
        else:
            _ds = StereoDepthDataset(root=root, load_count=load_count)
            datasets.append(_ds)

    ds = ConcatDataset(datasets)
    return ds


eval_names = ["room_1", "office_1", "room_2"]
train_names = [x for x in dataset_names if x not in eval_names]


def _load_train_dataset():
    train_ds = _load_datasets(train_names)
    assert len(train_ds) == len(train_names) * load_count
    return train_ds


def _load_eval_dataset():
    eval_ds = _load_datasets(eval_names)
    assert len(eval_ds) == len(eval_names) * load_count
    return eval_ds
