import random
import pickle
import numpy as np
import cv2
import os
from os import path as osp
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from utils.pixel_wise_mapping import remap_using_flow_fields


if __name__ == '__main__':
    DATA_PATH = '/data/datasets/hpatches-sequences-release/'
    PROJECT_PATH = '/data/projects/GLU-Net'
    PRECOMPUTED_RES_PATH = osp.join(PROJECT_PATH, 'uncertainty_output')
    MODELS = ['42', '777', '1984', '2020', '4224']
    scenes = [scene for scene in os.listdir(DATA_PATH) if scene[0] == 'v']
    viewpoints = ['12', '13', '14', '15', '16']

    while True:
        viewpoint = random.sample(viewpoints, 1)[0]
        scene = random.sample(scenes, 1)[0]
        keys = ["flow4", "flow3", "flow2", "flow_final"]
        u_over_models = defaultdict(list)
        v_over_models = defaultdict(list)
        scene_uncertainty_u = defaultdict(np.array)
        scene_uncertainty_v = defaultdict(np.array)
        var_total = defaultdict(np.array)

        for model in MODELS:
            fname = osp.join(PRECOMPUTED_RES_PATH, 'model_' + model, scene + '_' + viewpoint + '_flows.pkl')
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            '''
            for key in keys:
                u_est = data[key][0, 0].view(1, -1).cpu().numpy().tolist()[0]
                v_est = data[key][0, 1].view(1, -1).cpu().numpy().tolist()[0]
                u_over_models[key].append(u_est)
                v_over_models[key].append(v_est)
            '''
            for key in keys:
                u_est = data[key][0, 0].cpu().numpy()
                v_est = data[key][0, 1].cpu().numpy()
                u_over_models[key].append(u_est)
                v_over_models[key].append(v_est)

        '''
        for key in keys:
            mean_u, var_u = np.mean(u_over_models[key], axis=0), np.var(u_over_models[key], axis=0)
            mean_v, var_v = np.mean(v_over_models[key], axis=0), np.var(v_over_models[key], axis=0)
            scene_uncertainty_u[key] = np.vstack((mean_u, var_u))
            scene_uncertainty_v[key] = np.vstack((mean_v, var_v))
        '''
        for key in keys:
            var_total[key] = np.sqrt(np.var(u_over_models[key], axis=0) + np.var(v_over_models[key], axis=0)) / 2.

        '''
        steps = [1, 10, 100, 10000]
        ax_id = 0

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
        fig.suptitle("scene: " + scene + ', vp: ' + viewpoint)
        axes = axes.flatten()

        for i, (key, step) in enumerate(zip(keys, steps)):
            x_arr = range(0, len(scene_uncertainty_u[key][0, :]), step)
            u_conf95 = scene_uncertainty_u[key][0, 0::step] - 2 * scene_uncertainty_u[key][1, 0::step] ** 0.5
            u_conf105 = scene_uncertainty_u[key][0, 0::step] + 2 * scene_uncertainty_u[key][1, 0::step] ** 0.5

            v_conf95 = scene_uncertainty_v[key][0, 0::step] - 2 * scene_uncertainty_v[key][1, 0::step] ** 0.5
            v_conf105 = scene_uncertainty_v[key][0, 0::step] + 2 * scene_uncertainty_v[key][1, 0::step] ** 0.5

            axes[ax_id].plot(x_arr, scene_uncertainty_u[key][0, 0::step], marker='.', linestyle='none')
            axes[ax_id].fill_between(x_arr, u_conf95, u_conf105, alpha=0.2)
            axes[ax_id].set(title="u_disp: layer: " + key)
            axes[ax_id + 1].plot(x_arr, scene_uncertainty_v[key][0, 0::step], marker='.', linestyle='none')
            axes[ax_id + 1].fill_between(x_arr, v_conf95, v_conf105, alpha=0.2)
            axes[ax_id + 1].set(title="v_disp: layer: " + key)
            ax_id += 2

        plt.savefig('1.png')
        '''
        res_axis_ids = [0, 2, 4, 6]
        img_axis_ids = [1, 3, 5, 7]
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
        fig.suptitle("scene: " + scene + ', vp: ' + viewpoint)
        axes = axes.flatten()

        for key, ax_id in zip(keys, res_axis_ids):
            im = axes[ax_id].imshow(var_total[key])
        cbar_ax = fig.add_axes([0.50, 0.25, 0.02, 0.4])
        fig.colorbar(im, cax=cbar_ax)

        imgs = []
        # let's draw images
        for i, ax_id in enumerate(img_axis_ids[:-2]):
            img = Image.open(osp.join(DATA_PATH, scene, viewpoint[i] + '.ppm'))
            im = axes[ax_id].imshow(img)
            imgs.append(img)

        # let's warp an image
        disp_x, disp_y = np.mean(u_over_models["flow_final"], axis=0), np.mean(v_over_models["flow_final"], axis=0)
        img_warp = remap_using_flow_fields(np.array(imgs[0]), disp_x, disp_y)

        im = axes[img_axis_ids[-2]].imshow(img_warp)
        im = axes[img_axis_ids[-1]].imshow(np.abs(imgs[1].convert('L') - cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)))
        plt.savefig('1.png')
