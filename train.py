import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from flax import linen as nn
from flax import optim
from IPython import embed
from jax import jit, random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import losses
from data import StereoDepthDataset
from model import Model

# Parse arguments
root_dir = '/home/bryan/Dropbox/School/W21/CS 231A/hotel_0'
target_size = 12 * 36
epochs = 10000
max_disp = 64
max_raw_depth = 38003
run_id = "overfit1"
dataset = StereoDepthDataset(root_dir=root_dir,
                             target_size=target_size,
                             max_raw_depth=max_raw_depth,
                             max_disp=max_disp)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

writer = SummaryWriter(run_id)

key = random.PRNGKey(0)
test = next(enumerate(dataloader))[1]
# embed()
print("Creating model.")
model = Model(max_disp=max_disp)
test_left = jnp.array(test['frames']['left'])
test_right = jnp.array(test['frames']['right'])
test_depth = jnp.array(test['depth'])
variables = model.init(key, test_left, test_right)
print("Model initialized.")


@jit
def create_optimizer(params, learning_rate=0.0001):
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


# TODO: Define additional error functions if desired
@jit
def disparity_pyramid_loss(disp_pyramid, depth):
    # embed()
    pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]
    # Quick hack: interpolate upwards and calculate L1 loss for each one
    total_loss = 0
    for i, disp in enumerate(disp_pyramid):
        resized_depth = jax.image.resize(depth, disp.shape, method="bilinear")
        loss = losses.smooth_l1_loss(disp, resized_depth)
        total_loss += pyramid_weight[i] * loss
    return total_loss


@jit
def apply(params, batch):
    left = batch['left']
    right = batch['right']
    return model.apply(params, left, right, mutable=['batch_stats'])


@jit
def train_step(optimizer, batch, error=disparity_pyramid_loss):
    left = batch['left']
    right = batch['right']
    depth = batch['depth']

    def _loss(params):
        # embed()
        disp_pyramid, modified_variables = model.apply(params,
                                                       left,
                                                       right,
                                                       mutable=['batch_stats'])
        return error(disp_pyramid, depth)

    grad_fn = jax.value_and_grad(_loss)
    loss, grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)
    return optimizer, loss


optimizer = create_optimizer(variables)
_batch = {'left': test_left, 'right': test_right, 'depth': test_depth}
for i in tqdm(range(10000)):
    optimizer, loss = train_step(optimizer, _batch)
    print(f"i = {i}, loss = {loss}")
    if i % 250 == 0:
        embed()

for e in range(epochs):
    print(f"Epoch = {e+1}")
    for i, batch in tqdm(enumerate(dataloader)):
        # Convert to jnp.ndarray to pass into jitted context
        left_img = jnp.array(batch['frames']['left'])
        right_img = jnp.array(batch['frames']['right'])
        depth = jnp.array(batch['depth'])

        _batch = {
            'left': left_img,
            'right': right_img,
            'depth': depth,
        }

        optimizer, loss = train_step(optimizer, _batch)
        print(f"Loss = {loss}")
        # embed()
    embed()

embed()
