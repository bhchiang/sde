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
lr = 0.0001
batch_size = 1
run_id = f"_sde_full_5_batch_size{batch_size}_lr_{lr}"
# pretrained_path = "/home/bryan/work/sde/models/model__sde_full_4_epoch_78.pth"
pretrained_path = None

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
    train: bool

    def setup(self):
        self.feature_extractor = features.GCNetFeature(train=self.train)
        self.cost_volume_construction = cost.CostVolume(
            max_disp=common.max_disp)
        self.aggregation = aggregate.DeformSimpleBottleneck(
            planes=common.max_disp // (2**0),
            num_deform_groups=1,
            train=self.train)
        self.disparity_computation = disparity.DisparityEstimation(
            max_disp=common.max_disp)
        self.disparity_refinment = refinement.StereoDRNetRefinement(
            train=self.train)

    @nn.compact
    def __call__(self, left_img, right_img):
        left_feature = self.feature_extractor(left_img)
        right_feature = self.feature_extractor(right_img)
        print("Feature shapes", left_feature.shape, right_feature.shape)
        cost_volume = self.cost_volume_construction(left_feature,
                                                    right_feature)
        print("Cost volume", cost_volume.shape)
        aggregated = self.aggregation(cost_volume)
        print("Aggregated", aggregated.shape)

        disp = self.disparity_computation(cost_volume)
        print("Disparity", disp.shape)
        # print(disp.max(), disp.min())

        refined = self.disparity_refinment(disp, left_img, right_img)
        print("Refined", refined.shape)
        # print(refined.max(), refined.min())
        return refined


train_ds = data._load_train_dataset()
eval_ds = data._load_eval_dataset()

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=True)

print(f"train_ds: {len(train_ds)}, eval_ds: {len(eval_ds)}")


def get_initial_variables(key):
    left = right = jnp.ones((1, common.target_size, common.target_size, 3),
                            jnp.float32)
    return Model(train=True).init(key, left, right)


print("Initializing variables")
variables = get_initial_variables(key)
print("done")

if pretrained_path is not None:
    print(f"    - Loading weights from {pretrained_path}")
    ifile = open(pretrained_path, 'rb')
    bytes_input = ifile.read()
    ifile.close()
    variables = serialization.from_bytes(variables, bytes_input)
    print("Weights loaded")


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

    # Add more metrics if necessary
    return {
        'epe': epe,
        'loss': loss,
    }


def save_model(optimizer, e):
    fname = f'model_{run_id}_epoch_{e}.pth'
    serialize._save_model(optimizer.target, fname)


def _put(x):
    return {k: jnp.array(v) for k, v in x.items()}


def save_image(fname, img):
    plt.imsave(os.path.join(imgs_path, fname), img)


other_vars, params = variables.pop('params')
optimizer = create_optimizer(params, learning_rate=lr)
print("Optimizer defined")
# embed()


def _format(x):
    return np.array(_scale(x))


def _scale(x):
    return (x - x.min()) / (x.max() - x.min())


@jax.jit
def train_step(optimizer, batch, modified_vars, error=disparity_loss):
    left_img = batch['left']
    right_img = batch['right']
    gt_disp = batch['disparity']

    def _loss(params, modified_vars):
        disp, _modified_vars = Model(train=True).apply(
            {
                'params': params,
                'batch_stats': modified_vars['batch_stats']
            },
            left_img,
            right_img,
            mutable=['batch_stats'])
        aux = (disp, _modified_vars)
        return error(disp, gt_disp), aux

    grad_fn = jax.value_and_grad(_loss, has_aux=True)
    (_, aux), grad = grad_fn(optimizer.target, modified_vars)
    disp, modified_vars = aux

    # Update params
    # embed()
    optimizer = optimizer.apply_gradient(grad)
    metrics = compute_metrics(disp, gt_disp)
    return optimizer, disp, gt_disp, metrics, modified_vars


@jax.jit
def eval_step(variables, batch):
    left_img = batch['left']
    right_img = batch['right']
    gt_disp = batch['disparity']
    disp, modified_vars = Model(train=False).apply(variables, left_img,
                                                   right_img)
    metrics = compute_metrics(disp, gt_disp)
    return disp, gt_disp, metrics


def train_epoch(optimizer, train_loader, epoch, modified_vars):
    batch_metrics = []
    for i, batch in enumerate(train_loader):
        _batch = _put(batch)
        optimizer, disp, gt_disp, metrics, modified_vars = train_step(
            optimizer, _batch, modified_vars)
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
    return optimizer, training_epoch_metrics, modified_vars


def eval_epoch(variables, eval_loader, epoch):
    batch_metrics = []
    for i, batch in enumerate(eval_loader):
        _batch = _put(batch)
        disp, gt_disp, metrics = eval_step(variables, _batch)
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


modified_vars = other_vars
for e in range(num_epochs):
    print(f"    - Epoch: {e+1}")
    optimizer, train_metrics, modified_vars = train_epoch(
        optimizer, train_loader, e + 1, modified_vars)
    print(f"train_metrics: \n {train_metrics}")
    writer.add_scalars('train', train_metrics, e + 1)
    eval_metrics = eval_epoch(
        {
            'params': optimizer.target,
            'batch_stats': modified_vars['batch_stats']
        }, eval_loader, e + 1)
    writer.add_scalars('eval', eval_metrics, e + 1)
    print(f"eval_metrics: \n {eval_metrics}")
    embed()
