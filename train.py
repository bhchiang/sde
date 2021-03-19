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
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torch
from tqdm import tqdm

import aggregate
import cost
import disparity
import features
import refinement
import data
import common
import metrics
import serialize

num_epochs = 10000
test_idx = 10
lr = 0.0003
batch_size = 8
run_id = f"_small_intrasceneRESUME3_bs_{batch_size}_lr_{lr}"
# pretrained_name = "model__sde_full_4_epoch_78.pth"
# pretrained_name = "good/aa4_model_10000epoch.pth"
# pretrained_name = "model__test_load_batch_size8_lr_0.0005_epoch_1.pth"
pretrained_name = "model__small_intrasceneRESUME2_bs_8_lr_0.0001_epoch_32.pth"
pretrained_path = os.path.join(serialize.model_path, pretrained_name)
imgs_path = "images/"
os.makedirs(imgs_path, exist_ok=True)
writer = SummaryWriter(comment=run_id)
tensorboard_im_freq = 10  # Save 8 images every 8*8 = 64 images / 8 batches
key = random.PRNGKey(0)


def _show_image(img):
    plt.figure()
    plt.imshow(img)
    plt.show()


class Model(nn.Module):
    def setup(self):
        self.feature_extractor = features.GCNetFeature()
        self.cost_volume_construction = cost.CostVolume(
            max_disp=common.max_disp)
        self.aggregation = aggregate.DeformSimpleBottleneck(
            planes=common.max_disp // (2**0), num_deform_groups=1)
        self.disparity_computation = disparity.DisparityEstimation(
            max_disp=common.max_disp)
        self.disparity_refinment = refinement.StereoDRNetRefinement()

    @nn.compact
    def __call__(self, left_img, right_img):
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)
        # print("Feature shapes", left_feature.shape, right_feature.shape)

        cost_volume = self.cost_volume_construction(left_feature,
                                                    right_feature)
        # print("Cost volume", cost_volume.shape)

        aggregated = self.aggregation(cost_volume)
        # print("Aggregated", aggregated.shape)

        disp = self.disparity_computation(cost_volume)
        # print("Disparity", disp.shape)
        # print(disp.max(), disp.min())

        refined = self.disparity_refinment(disp, left_img, right_img)
        # print("Refined", refined.shape)
        # print(refined.max(), refined.min())
        return refined


# train_ds = data._load_train_dataset()
# eval_ds = data._load_eval_dataset()

names = ["hotel_0", "apartment_1", "office_2", "frl_apartment_3"]
total_ds = data._load_datasets(names)

# eval_ds = data._load_single_dataset("hotel_0", offset=80, load_count=20)
# train_ds = data._load_single_dataset("hotel_0", load_count=80)

train_len = int(len(total_ds) * 0.8)
eval_len = len(total_ds) - train_len
lengths = (train_len, eval_len)
train_ds, eval_ds = random_split(total_ds,
                                 lengths,
                                 generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True)

print(f"train_ds: {len(train_ds)}, eval_ds: {len(eval_ds)}")

# sys.exit()
model = Model()
print("Creating model")


def _retrieve(idx, ds):
    _left = jnp.array(ds[idx]['left'][jnp.newaxis, ...])
    _right = jnp.array(ds[idx]['right'][jnp.newaxis, ...])
    _depth = jnp.array(ds[idx]['disparity'][jnp.newaxis, ...])
    return _left, _right, _depth


test_left, test_right, *_ = _retrieve(test_idx, train_ds)
variables = model.init(key, test_left, test_right)
print("Model created")
if pretrained_path is not None:
    print(f"    - Loading weights from {pretrained_path}")
    ifile = open(pretrained_path, 'rb')
    bytes_input = ifile.read()
    ifile.close()
    variables = serialization.from_bytes(variables, bytes_input)
    print("Weights loaded")


@jax.jit
def apply(variables, left_img, right_img):
    y, modified_vars = model.apply(variables,
                                   left_img,
                                   right_img,
                                   mutable=['batch_stats'])
    return y


def create_optimizer(params, learning_rate=0.0001):
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


@jax.jit
def disparity_loss(disp, gt_disp):
    # Compute average L1 losss for the given batch
    batch_l1 = jnp.sum(jnp.abs(disp - gt_disp), axis=(1, 2, 3))
    return jnp.mean(batch_l1)


def compute_metrics(disp, gt_disp):
    epe = metrics._epe(disp, gt_disp)
    loss = disparity_loss(disp, gt_disp)
    pixel1 = metrics._1pixel(disp, gt_disp)

    # Add more metrics if necessary
    return {
        'epe': epe,
        'loss': loss,
        'pixel1': pixel1,
    }


@jax.jit
def train_step(optimizer, batch, error=disparity_loss):
    left_img = batch['left']
    right_img = batch['right']
    gt_disp = batch['disparity']

    def _loss(params):
        disp = apply(params, left_img, right_img)
        return error(disp, gt_disp), disp

    grad_fn = jax.value_and_grad(_loss, has_aux=True)
    (_, disp), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(disp, gt_disp)
    return optimizer, disp, gt_disp, metrics


@jax.jit
def eval_step(model, batch):
    left_img = batch['left']
    right_img = batch['right']
    gt_disp = batch['disparity']
    disp = apply(model, left_img, right_img)
    metrics = compute_metrics(disp, gt_disp)
    return disp, gt_disp, metrics


def save_model(optimizer, e):
    fname = f'model_{run_id}_epoch_{e}.pth'
    serialize._save_model(optimizer.target, fname)


def _put(x):
    return {k: jnp.array(v) for k, v in x.items()}


def save_image(fname, img):
    plt.imsave(os.path.join(imgs_path, fname), img)


optimizer = create_optimizer(variables, learning_rate=lr)
print("Optimizer defined")


def _format(x):
    return np.array(_scale(x))


def _scale(x):
    return (x - x.min()) / (x.max() - x.min())


def train_epoch(optimizer, train_loader, epoch):
    batch_metrics = []
    for i, batch in enumerate(train_loader):
        _batch = _put(batch)
        optimizer, disp, gt_disp, metrics = train_step(optimizer, _batch)
        print(f"e = {e}, i = {i}, loss = {metrics['loss']}")
        batch_metrics.append(metrics)

        if i == 0:
            writer.add_images(f'train_pred_disp',
                              _format(disp),
                              epoch,
                              dataformats="NHWC")
            writer.add_images(f'train_gt_disp',
                              _format(gt_disp),
                              epoch,
                              dataformats="NHWC")

    training_batch_metrics = jax.device_get(batch_metrics)
    training_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in training_batch_metrics])
        for k in training_batch_metrics[0]
    }
    return optimizer, training_epoch_metrics


def eval_epoch(model, eval_loader, epoch):
    batch_metrics = []
    for i, batch in enumerate(eval_loader):
        _batch = _put(batch)
        disp, gt_disp, metrics = eval_step(model, _batch)
        batch_metrics.append(metrics)

        if i == 0:
            writer.add_images(f'eval_pred_disp',
                              _format(disp),
                              epoch,
                              dataformats="NHWC")
            writer.add_images(f"eval_gt_disp",
                              _format(gt_disp),
                              epoch,
                              dataformats="NHWC")

    eval_batch_metrics = jax.device_get(batch_metrics)
    eval_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in eval_batch_metrics])
        for k in eval_batch_metrics[0]
    }
    return eval_epoch_metrics


for e in range(num_epochs):
    print(f"    - Epoch: {e+1}")
    optimizer, train_metrics = train_epoch(optimizer, train_loader, e + 1)
    print(f"train_metrics: \n {train_metrics}")
    writer.add_scalars('train', train_metrics, e + 1)

    save_model(optimizer, e + 1)

    eval_metrics = eval_epoch(optimizer.target, eval_loader, e + 1)
    writer.add_scalars('eval', eval_metrics, e + 1)
    print(f"eval_metrics: \n {eval_metrics}")
