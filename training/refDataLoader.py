from cv2 import cvtColor
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import cv2
import numpy as np

class refDataset(Dataset):
    def __init__(self, source_root, ref_root, target_root, opts, source_transform=None, ref_transform=None, target_transform=None):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.ref_paths = sorted(data_utils.make_dataset(ref_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.ref_transform = ref_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    # 将img转换成到lab空间，source取L通道，ref取ab通道，生成的图像与target分别在L和ab通道上做loss
    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        from_im = from_im.convert('L')
        # 修改为使用cv2读取图片
        # from_im = cv2.imread(from_path)
        # from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
        # # from_im_rgb = from_im
        # from_im = cv2.cvtColor(from_im, cv2.COLOR_RGB2LAB)
        # # from_im_lab = from_im
        # from_im = from_im[:, :, 0]
        # # from_im = (from_im * 255).astype(np.uint8)
        # from_im = Image.fromarray(from_im)
        # from_im_rgb = Image.fromarray(from_im_rgb)
        # from_im_lab = Image.fromarray(from_im_lab)

        ref_path = self.ref_paths[index]
        ref_im = Image.open(ref_path).convert('RGB')
        # 将ref转到ab通道
        # ref_im = cv2.imread(ref_path)
        # ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2RGB)
        # ref_im_rgb = ref_im
        # ref_im = cv2.cvtColor(ref_im, cv2.COLOR_RGB2LAB)
        # ref_im = ref_im[:, :, 1:3]
        # # ref_im = (ref_im * 255).astype(np.uint8)
        # # ref_im_rgb = (ref_im_rgb * 255).astype(np.uint8)
        # ref_im = Image.fromarray(ref_im)
        # ref_im_rgb = Image.fromarray(ref_im_rgb)


        to_path = self.target_paths[index]
        to_im = Image.open(to_path).convert('RGB')
        # 将target转换到lab
        # to_im = cv2.imread(to_path)
        # to_im = cv2.cvtColor(to_im, cv2.COLOR_BGR2RGB)
        # to_im = cv2.cvtColor(to_im, cv2.COLOR_RGB2LAB)
        # # to_im = (to_im * 255).astype(np.uint8)
        # to_im = to_im[:, :, 1:3]
        # to_im = Image.fromarray(to_im)

        if self.source_transform:
            from_im = self.source_transform(from_im)

        if self.ref_transform:
            ref_im = self.ref_transform(ref_im)
            # ref_im_rgb = self.ref_transform(ref_im_rgb)

        if self.target_transform:
            to_im = self.target_transform(to_im)

        # return from_im, ref_im, ref_im_rgb, to_im
        return from_im, ref_im, to_im

