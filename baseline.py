import cv2
# import pyrealsense2 as rs
from IPython import embed
import numpy as np
import data
import metrics
import torch
from torch.utils.data import DataLoader, random_split
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

batch_size = 16
names = ["hotel_0", "apartment_1", "office_2", "frl_apartment_3"]
total_ds = data._load_datasets(names)

train_len = int(len(total_ds) * 0.8)
eval_len = len(total_ds) - train_len
lengths = (train_len, eval_len)
train_ds, eval_ds = random_split(total_ds, lengths,
                                 generator=torch.Generator().manual_seed(42))
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
# eval_ds = data._load_eval_dataset()
# eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
#

def _put(x):
    return {k: jnp.array(v) for k, v in x.items()}

def eval_epoch(eval_loader):
    err_epes = []
    err_1pxs = []
    for i, batch in enumerate(eval_loader):
        _batch = _put(batch)
        gt_disp = _batch['disparity']
        left_img = _batch['left']
        right_img = _batch['right']
        pred_disp, gt = _dense_stereo_disparity(left_img, right_img, gt_disp)

        err_1px = metrics._1pixel(pred_disp, gt)
        err_epe = metrics._epe(pred_disp, gt)
        err_epes.append(err_epe)
        err_1pxs.append(err_1px)

    av_epe = np.sum(err_epes) / len(err_epes)
    av_1px = np.sum(err_1pxs) / len(err_1pxs)
    return av_epe, av_1px

def _dense_stereo_disparity(left_img, right_img, gt):
    window_size = 5
    min_disp = 0
    num_disp = 16  # must be divisible by 16

    # stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    #                                numDisparities=num_disp,
    #                                blockSize=16,
    #                                P1=8 * 3 * window_size ** 2,
    #                                P2=32 * 3 * window_size ** 2,
    #                                disp12MaxDiff=10,
    #                                uniquenessRatio=10,
    #                                speckleWindowSize=100,
    #                                speckleRange=32)
    stereo = cv2.StereoBM_create(numDisparities=num_disp)

    for i in range(left_img.shape[0]):  # stereoBM/stereoSGBM doesn't take batches
        grey_left_img = cv2.cvtColor(np.float32(left_img[i]*255.), cv2.COLOR_RGB2GRAY)
        grey_right_img = cv2.cvtColor(np.float32(right_img[i]*255.), cv2.COLOR_RGB2GRAY)
        disp = stereo.compute(grey_left_img.astype(np.uint8), grey_right_img.astype(np.uint8)).astype(np.float32) / 16.0

        # if i == 0:  # occasionally visualize
        #     v = disp[disp > 0]
        #     tmp = disp.copy()
        #     tmp[tmp > 0] = (v - v.min()) / (v.max() - v.min())
        #     tmp = np.clip(tmp, 0, 1)  # clip negative "unmatched" values only for visualization
        #     plt.imsave("stereoBM.png", tmp)
        #     plt.show()
        #     plt.imshow(np.squeeze(gt[i], axis=2))
        #     plt.show()

        matched = (disp > 0) # create boolean mask

        # get matched disparities and scale
        x = disp[disp > 0]
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        matched_disp = x.flatten()

        # apply mask to ground truth disparities and scale
        gti = np.squeeze(gt[i], axis=2)
        gt_ = (gti-np.min(gti))/(np.max(gti)-np.min(gti))
        matched_gt = gt_[matched].flatten()

        if i == 0:
            pred_disp = matched_disp
            gt_disp = matched_gt

        else:
            pred_disp = np.append(pred_disp, matched_disp)
            gt_disp = np.append(gt_disp, matched_gt)

    return 64 * pred_disp, 64 * gt_disp  # back to 0-64 range

av_epe, av_1px = eval_epoch(eval_loader)
print("average epe: " + str(av_epe))
print("average 1px: " + str(av_1px))

