import cv2
# import pyrealsense2 as rs
from IPython import embed
import numpy as np
import data
import metrics
from torch.utils.data import DataLoader
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

batch_size = 16

eval_ds = data._load_eval_dataset()
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)


def _put(x):
    return {k: jnp.array(v) for k, v in x.items()}

def eval_epoch(eval_loader):
    batch_metrics = []
    for i, batch in enumerate(eval_loader):
        _batch = _put(batch)
        gt_disp = _batch['disparity']
        if i ==0:
            print(gt_disp[0])
            print("printed")
        left_img = _batch['left']
        right_img = _batch['right']
        pred_disp = _dense_stereo_disparity(left_img, right_img)
        err_1px = metrics._1pixel(pred_disp, gt_disp)
        err_epe = metrics._epe(pred_disp, gt_disp)
        print(err_epe)
        print(err_1px)
        # if i == 0:
        #     writer.add_images(f'eval_pred_disp',
        #                       _format(disp),
        #                       epoch,
        #                       dataformats="NHWC")
        #     writer.add_images(f"eval_gt_disp",
        #                       _format(gt_disp),
        #                       epoch,
        #                       dataformats="NHWC")
        # eval_batch_metrics = jax.device_get(batch_metrics)
        # eval_epoch_metrics = {
        #     k: np.mean([metrics[k] for metrics in eval_batch_metrics])
        #     for k in eval_batch_metrics[0]
        # }
    # eval_batch_metrics = jax.device_get(batch_metrics)
    # eval_epoch_metrics = {
    #     k: np.mean([metrics[k] for metrics in eval_batch_metrics])
    #     for k in eval_batch_metrics[0]
    # }

    #return eval_epoch_metrics

def _dense_stereo_disparity(left_img, right_img):
    window_size = 5
    min_disp = 0
    # must be divisible by 16
    num_disp = 64  # original:  112 - min_disp
    max_disp = min_disp + num_disp
    # stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
    #                                numDisparities=num_disp,
    #                                blockSize=16,
    #                                P1=8 * 3 * window_size ** 2,
    #                                P2=32 * 3 * window_size ** 2,
    #                                disp12MaxDiff=1,
    #                                uniquenessRatio=10,
    #                                speckleWindowSize=100,
    #                                speckleRange=32)
    stereo = cv2.StereoBM_create(numDisparities=num_disp)
                            #SADWindowSize=_SADWindowSize,
                                 #blockSize=16
    disparities = []
    for i in range(left_img.shape[0]):
        grey_left_img = cv2.cvtColor(np.float32(left_img[i]*255.), cv2.COLOR_RGB2GRAY)
        grey_right_img = cv2.cvtColor(np.float32(right_img[i]*255.),cv2.COLOR_RGB2GRAY)
        # if i ==0:
        #     print(grey_left_img.astype(np.uint8))

        # TODO: CV_8UC1 equivalent to uint8? but then don't u lose some info lmao
        pred_disp = stereo.compute(grey_left_img.astype(np.uint8), grey_right_img.astype(np.uint8))/64. #.astype(np.float32) #/ 16.0
        # plt.imshow(pred_disp)
        # plt.show()

        if i ==0:
            print(pred_disp)
        disparities.append(pred_disp)
    pred_disp_stacked = np.stack(disparities, axis=0)
    pred_disp_stacked = np.expand_dims(pred_disp_stacked, axis=-1)
    # print(pred_disp_stacked.shape)
    return pred_disp_stacked

eval_epoch(eval_loader)

