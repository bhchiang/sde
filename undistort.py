import math
import sys

import cv2 as cv
import numpy as onp
import pyrealsense2 as rs
from camera import calculate_extrinsics, calculate_intrinsics, fisheye_distortion
from IPython import embed
import matplotlib.pyplot as plt

LEFT = 1
RIGHT = 2
WINDOW_TITLE = "LR"

pipe = rs.pipeline()
cfg = rs.config()
pipe.start(cfg)

# Calculate intrinsics and extrinsics
profiles = pipe.get_active_profile()
streams = {
    LEFT: profiles.get_stream(rs.stream.fisheye,
                              LEFT).as_video_stream_profile(),
    RIGHT: profiles.get_stream(rs.stream.fisheye,
                               RIGHT).as_video_stream_profile(),
}
intrinsics = {
    LEFT: streams[LEFT].get_intrinsics(),
    RIGHT: streams[RIGHT].get_intrinsics(),
}

# Print information about both cameras
print(f"Left camera: {intrinsics[LEFT]}")
print(f"Right camera: {intrinsics[RIGHT]}")

# Translate the intrinsics from librealsense into OpenCV
K_l = calculate_intrinsics(intrinsics[LEFT])
D_l = fisheye_distortion(intrinsics[LEFT])
K_r = calculate_intrinsics(intrinsics[RIGHT])
D_r = fisheye_distortion(intrinsics[RIGHT])
w, h = intrinsics[LEFT].width, intrinsics[LEFT].height

# Get the relative extrinsics between the left and right camera
(R, T) = calculate_extrinsics(streams[LEFT], streams[RIGHT])

# Size of undistorted image (keep original size)
size = (w, h)

# Not sure why fisheye.stereoRectify doesn't work here
R_l, R_r, P_l, P_r, *_ = cv.stereoRectify(
    K_l,
    D_l,
    K_r,
    D_r,
    size,
    R,
    T,
)

print(P_l.shape, P_r.shape)
# P1, and P2 Should only differ by translation for right camera (P2)
print(f"P_l: {P_l}")
print(f"P_r: {P_r}")
# print(R1, R2)

(lm_x, lm_y) = cv.fisheye.initUndistortRectifyMap(K_l, D_l, R_l, P_l, size,
                                                  cv.CV_16SC2)
(rm_x, rm_y) = cv.fisheye.initUndistortRectifyMap(K_r, D_r, R_r, P_r, size,
                                                  cv.CV_16SC2)

# sys.exit()

save = True

while True:
    # TODO: run remapping in separate thread if real-time depth estimation desired
    frames = pipe.wait_for_frames()

    print(frames)

    frameset = frames.as_frameset()

    fl = frameset.get_fisheye_frame(LEFT).as_video_frame().get_data()
    fr = frameset.get_fisheye_frame(RIGHT).as_video_frame().get_data()

    fl = onp.array(fl)
    fr = onp.array(fr)

    print(fl.shape)
    print(fr.shape)

    undistorted_fl = cv.remap(src=fl,
                              map1=lm_x,
                              map2=lm_y,
                              interpolation=cv.INTER_LINEAR)
    undistorted_fr = cv.remap(src=fr,
                              map1=rm_x,
                              map2=rm_y,
                              interpolation=cv.INTER_LINEAR)

    if save:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(fl)
        axs[0, 0].set_title("Left Fisheye")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(fr)
        axs[0, 1].set_title("Right Fisheye")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(undistorted_fl)
        axs[1, 0].set_title("Left Undistorted")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(undistorted_fr)
        axs[1, 1].set_title("Right Undistorted")
        axs[1, 1].axis("off")

        fig.savefig("images/undistort.png", bbox_inches="tight")

        plt.figure()
        plt.imshow(onp.hstack((undistorted_fl, undistorted_fr)))
        plt.hlines(onp.arange(0, h, 120), 0, 2 * w, colors="white")
        plt.axis("off")
        plt.savefig("images/rectify.png")
        break

    cv.imshow(WINDOW_TITLE, onp.hstack((fl, undistorted_fl)))
    key = cv.waitKey(1)
    if key == ord("q"):
        break
