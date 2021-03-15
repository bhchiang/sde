# import torch
#import torch.nn.functional as F
import jax
import jax.numpy as jnp


def normalize_coords(grid):
    """Normalize coordinates of image scale to [-1, 1]
    Args:
        grid: [B, 2, H, W]
    """

    #OURS: [B, H, W, 2]
    assert grid.shape[-1] == 2
    h, w = grid.shape[1:3]
    grid = grid.at[:, :, :,
                   0].set(2 * (grid[:, :, :, 0] / (w - 1)) - 1)  # x: [-1,1]
    #grid[:, 0, :, :] = 2 * (grid[:, 0, :, :].clone() / (w - 1)) - 1  # x: [-1, 1]
    grid = grid.at[:, :, :,
                   1].set(2 * (grid[:, :, :, 1] / (h - 1)) - 1)  # y: [-1, 1]
    # grid[:, 1, :, :] = 2 * (grid[:, 1, :, :].clone() / (h - 1)) - 1  # y: [-1, 1]
    #grid = grid.permute((0, 2, 3, 1))  # [B, H, W, 2]
    return grid


def meshgrid(img, homogeneous=False):
    """Generate meshgrid in image scale
    Args:
        img: [B, _, H, W]
        homogeneous: whether to return homogeneous coordinates
    Return:
        grid: [B, 2, H, W]
    """
    b, h, w, _ = img.shape  #b, _, h, w = img.size()

    #TODO: make this ~ jaxy ~; nvm
    x_range = jnp.arange(w).reshape(
        1, w, 1
    )  #torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(img)  # [1, H, W]
    x_range = jnp.repeat(x_range, h, axis=0)
    y_range = jnp.arange(h).reshape(
        h, 1,
        1)  #torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(img)
    y_range = jnp.repeat(y_range, w, axis=1)

    grid = jnp.concatenate(
        [x_range, y_range], axis=2
    )  #torch.cat((x_range, y_range), dim=0) [2, H, W], grid[:, i, j] = [j, i]
    grid = jnp.expand_dims(grid, axis=0)  # grid.unsqueeze(0)
    grid = jnp.repeat(grid, b, axis=0)  # .expand(b, 2, h, w)  [B, 2, H, W]

    if homogeneous:
        ones = jnp.ones(
            (b, h, w, 1)
        )  #torch.ones_like(x_range).unsqueeze(0).expand(b, 1, h, w)  # [B, 1, H, W]
        grid = jnp.concatenate(
            [grid, ones],
            axis=3)  #torch.cat((grid, ones), dim=1)  # [B, 3, H, W]
        assert grid.shape[3] == 3
    return grid


def disp_warp(img, disp, padding_mode='border'):
    """Warping by disparity
    Args:
        img: [B, 3, H, W]
        disp: [B, 1, H, W], positive
        padding_mode: 'zeros' or 'border'
    Returns:
        warped_img: [B, 3, H, W]
        valid_mask: [B, 3, H, W]
    """
    # assert disp.min() >= 0

    grid = meshgrid(img)  # [B, 2, H, W] in image scale
    # Note that -disp here
    B, H, W, _ = disp.shape
    offset = jnp.concatenate(
        [-disp, jnp.zeros((B, H, W, 1))], axis=3
    )  #torch.cat((-disp, torch.zeros_like(disp)), dim=1)  # [B, 2, H, W]
    sample_grid = grid + offset
    sample_grid = normalize_coords(sample_grid)  # [B, H, W, 2] in [-1, 1]
    print("sample grid", sample_grid.max(), sample_grid.min())
    y = sample_grid[..., 0]
    x = sample_grid[..., 1]
    print(y.shape, x.shape)
    warped_img = bilinear_sampler(
        img, x, y
    )  #TODO: UH,,, F... #F.grid_sample(img, sample_grid, mode='bilinear', padding_mode=padding_mode)
    print(img.max(), img.min())
    print(warped_img.max(), warped_img.min())
    print(warped_img.shape)
    #
    valid_mask = None
    #TODO: is this mask used ?! the fuck
    # mask = jnp.zeros(img.shape) #torch.ones_like(img)
    # valid_mask = F.grid_sample(mask, sample_grid, mode='bilinear', padding_mode='zeros')
    # valid_mask[valid_mask < 0.9999] = 0
    # valid_mask[valid_mask > 0] = 1
    return warped_img, valid_mask


#################### TODO: CODE BELOW CONVERTED FROM TENSORFLOW IMPLEMENTATION... not sure what jax equivalent of
#TODO: F.grid_sample is... below code from: https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = img.shape[1]
    W = img.shape[2]
    max_y = jnp.int32(H - 1)
    max_x = jnp.int32(W - 1)
    zero = 0  #?? jnp.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = x.astype('float32')
    y = y.astype('float32')
    x = 0.5 * ((x + 1.0) * (max_x - 1).astype('float32'))
    y = 0.5 * ((y + 1.0) * (max_y - 1).astype('float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = jnp.floor(x).astype('int32')
    x1 = x0 + 1
    y0 = jnp.floor(y).astype('int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = jnp.clip(x0, 0, max_x)  #clip_by_value(x0, zero, max_x)
    x1 = jnp.clip(x1, 0, max_x)  #tf.clip_by_value(x1, zero, max_x)
    y0 = jnp.clip(y0, 0, max_y)  #tf.clip_by_value(y0, zero, max_y)
    y1 = jnp.clip(y1, 0, max_y)  #tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = x0.astype('float32')
    x1 = x1.astype('float32')
    y0 = y0.astype('float32')
    y1 = y1.astype('float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    print(wa.max(), wb.max(), wc.max(), wd.max())

    # add dimension for addition
    wa = jnp.expand_dims(wa, axis=3)
    wb = jnp.expand_dims(wb, axis=3)
    wc = jnp.expand_dims(wc, axis=3)
    wd = jnp.expand_dims(wd, axis=3)

    # compute output
    out = jnp.stack([wa * Ia, wb * Ib, wc * Ic, wd * Id])
    print("out range", out.max(), out.min())
    print(out.shape)
    out = jnp.sum(out, axis=0)  #tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
    print(out.shape)

    return out


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = x.shape
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = jnp.arange(batch_size)  #tf.range(0, batch_size)
    batch_idx = jnp.reshape(batch_idx, (batch_size, 1, 1))
    b = jnp.tile(batch_idx, (1, height, width)).astype("int32")

    indices = jnp.stack([b, y, x], 3)

    result = jnp.take(img, indices)
    print("result", result.max(), result.min())
    return result
