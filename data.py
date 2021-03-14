import os

import jax
import jax.numpy as jnp
import numpy as np
from IPython import embed
from skimage import io
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset


class StereoDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 max_raw_depth,
                 max_disp,
                 target_size,
                 transform=None):
        self.root_dir = root_dir
        self.target_size = target_size
        self.max_raw_depth = max_raw_depth
        self.max_disp = max_disp

        files = os.listdir(root_dir)
        self.length = len(files) // 3

        self.depth = []
        self.left_rgb = []
        self.right_rgb = []
        for i in range(self.length):
            frame_num = f"{i:06d}"
            _depth = os.path.join(self.root_dir, f"depth{frame_num}.png")
            _left = os.path.join(self.root_dir, f"frame{frame_num}_left.jpg")
            _right = os.path.join(self.root_dir, f"frame{frame_num}_right.jpg")
            self.depth.append(_depth)
            self.left_rgb.append(_left)
            self.right_rgb.append(_right)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(idx)
        if idx > len(self):
            raise ValueError("Index out of bounds.")

        _left = self.left_rgb[idx]
        _right = self.right_rgb[idx]
        _depth = self.depth[idx]

        left_img = self.load_image(_left)
        right_img = self.load_image(_right)
        depth_img = self.load_image(_depth)

        # Scale depth values
        # TODO: fix
        # depth_img = depth_img * (255 * self.max_disp / self.max_raw_depth)

        return {
            'frames': {
                'left': left_img,
                'right': right_img
            },
            'depth': depth_img
        }

    def crop(self, img):
        return img[:, 24:824]

    def load_image(self, img_path):
        img = np.array(io.imread(img_path)) / 255.
        # Crop image to target size
        img = self.crop(img)

        # Resize to target size
        img = resize(img, (self.target_size, self.target_size))
        return img


if __name__ == "__main__":
    root_dir = '/home/bryan/Dropbox/School/W21/CS 231A/hotel_0'
    ds = StereoDepthDataset(root_dir=root_dir, target_size=432)
    embed()
