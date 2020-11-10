import random
import pickle
import numpy as np
import cv2
import os
from os import path as osp
import itertools
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image

from utils.pixel_wise_mapping import remap_using_flow_fields
from utils.image_transforms import ArrayToTensor
from models.models_compared import GLU_Net


class DTUEvaluationDataset(Dataset):
    def __init__(self, scene_path, image_transform, image_size=(240, 240)):
        self.scene_path = scene_path
        self.image_transform = image_transform
        self.image_size = image_size

        fnames = ['clean_{:03d}_max.png'.format(k) for k in range(1, 50)]
        self.pairs = [pair for pair in itertools.combinations(fnames, 2)]

    def __getitem__(self, idx):
        fname1, fname2 = self.pairs[idx]

        img1 = cv2.imread(osp.join(self.scene_path, fname1))
        img2 = cv2.imread(osp.join(self.scene_path, fname2))

        if self.image_size is not None:
            h_scale, w_scale = self.image_size[0], self.image_size[1]
            img1 = cv2.cvtColor(cv2.resize(img1, (w_scale, h_scale)), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.resize(img2, (w_scale, h_scale)), cv2.COLOR_BGR2RGB)

        inputs = [img1, img2]
        inputs[0] = self.image_transform(inputs[0])
        inputs[1] = self.image_transform(inputs[1])

        return {'source_image_tensor': inputs[0],
                'target_image_tensor': inputs[1],
                'source_image_fname': fname1,
                'target_image_fname': fname2
                }

    def __len__(self):
        return len(self.pairs)


if __name__ == '__main__':
    DATA_PATH = '/data/datasets/Cleaned'
    PROJECT_PATH = '/data/projects/GLU-Net'
    PRECOMPUTED_RES_PATH = osp.join(PROJECT_PATH, 'uncertainty_output')
    SNAPSHOT = osp.join(PROJECT_PATH, 'snapshots')
    MODELS = ['42', '777', '1984', '2020', '4224']
    SCENES = ['scan1', 'scan3', 'scan17']
    IMAGE_SIZE = (240, 240)

    input_images_transform = transforms.Compose([ArrayToTensor(get_float=False)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_id = 0
    while True:
        scene = random.sample(SCENES, 1)[0]
        dataset = DTUEvaluationDataset(osp.join(DATA_PATH, scene),
                                       input_images_transform,
                                       image_size=IMAGE_SIZE)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8)
        data = next(iter(dataloader))

        keys = ["flow4", "flow3", "flow2", "flow_final"]
        u_over_models = defaultdict(list)
        v_over_models = defaultdict(list)
        var_total = defaultdict(np.array)
        # predict flows by networks
        with torch.no_grad():
            for model_id in MODELS:
                print('Starting model_' + model_id)
                snapshot_fname = osp.join(SNAPSHOT, "GLUNet_train_" + model_id, "model_best.pth")
                network = GLU_Net(path_pre_trained_models=snapshot_fname,
                                  consensus_network=False,
                                  cyclic_consistency=True,
                                  iterative_refinement=True,
                                  apply_flipping_condition=False)
                flow4, flow3, flow2, flow_final = network.estimate_flow(data['source_image_tensor'],
                                                                        data['target_image_tensor'],
                                                                        device)
                for key, est in zip(keys, [flow4, flow3, flow2, flow_final]):
                    u_over_models[key].append(est[0, 0].cpu().numpy())
                    v_over_models[key].append(est[0, 1].cpu().numpy())

        for key in keys:
            var_total[key] = np.sqrt(np.var(u_over_models[key], axis=0) + np.var(v_over_models[key], axis=0)) / 2.

        res_axis_ids = [0, 2, 4, 6]
        img_axis_ids = [1, 3, 5, 7]
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
        fig.suptitle("scene: " + scene)
        axes = axes.flatten()

        for key, ax_id in zip(keys, res_axis_ids):
            im = axes[ax_id].imshow(var_total[key])
        cbar_ax = fig.add_axes([0.50, 0.25, 0.02, 0.4])
        fig.colorbar(im, cax=cbar_ax)

        imgs = []
        # let's draw images
        for i, (ax_id, fn) in enumerate(zip(img_axis_ids[:-2], [data['source_image_fname'], data['target_image_fname']])):
            img = Image.open(osp.join(DATA_PATH, scene, fn[0]))
            if IMAGE_SIZE is not None:
                img = img.resize(IMAGE_SIZE)

            im = axes[ax_id].imshow(img)
            imgs.append(img)

        # let's warp an image
        disp_x, disp_y = np.mean(u_over_models["flow_final"], axis=0), np.mean(v_over_models["flow_final"], axis=0)
        img_warp = remap_using_flow_fields(np.array(imgs[0]), disp_x, disp_y)

        im = axes[img_axis_ids[-2]].imshow(img_warp)
        im = axes[img_axis_ids[-1]].imshow(np.abs(imgs[1].convert('L') - cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)))
        plt.savefig(str(img_id) + '_dtu.png')
        img_id += 1
