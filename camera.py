import numpy as onp
import pyrealsense2 as rs
from IPython import embed


def calculate_intrinsics(intrinsics):
    print(intrinsics)
    return onp.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ])


def calculate_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = onp.array(extrinsics.rotation).reshape((3, 3)).T
    T = onp.array(extrinsics.translation)
    return (R, T)


def fisheye_distortion(intrinsics):
    # Kannala-Brandt distortion coefficientse
    return onp.array(intrinsics.coeffs[:4])


if __name__ == "__main__":
    context = rs.context()
    sensor = context.query_all_sensors()[0]

    print(sensor.profiles)

    streams = {
        "left": sensor.profiles[0].as_video_stream_profile(),
        "right": sensor.profiles[1].as_video_stream_profile(),
    }

    intrinsics = {
        "left": calculate_intrinsics(streams["left"].get_intrinsics()),
        "right": calculate_intrinsics(streams["right"].get_intrinsics()),
    }

    # Calculate transformation from left to right camera
    extrinsics = calculate_extrinsics(streams["left"], streams["right"])

    print(intrinsics["left"])
    print(intrinsics["right"])
    print(extrinsics)
    """
    Intrinsics for left camera
    [[287.9636   0.     434.9343]
    [  0.     286.3158 395.5179]
    [  0.       0.       1.    ]]
    Intrinsics for right camera
    [[287.9015   0.     434.9749]
    [  0.     286.2658 399.7415]
    [  0.       0.       1.    ]]
    Rotation to second camera
    (DeviceArray([[ 0.99998623,  0.00260651, -0.00456264],
                [-0.00259843,  0.99999505,  0.00177418],
                [ 0.00456724, -0.0017623 ,  0.99998796]], dtype=float32), 
    Translation to second camera
    DeviceArray([-6.3469492e-02, -2.9780273e-05, -2.3131282e-04], dtype=float32))
    """