import random
import pickle
import numpy as np
import os
from os import path as osp
import matplotlib.pyplot as plt
from collections import defaultdict


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

        for model in MODELS:
            fname = osp.join(PRECOMPUTED_RES_PATH, 'model_' + model, scene + '_' + viewpoint + '_flows.pkl')
            with open(fname, 'rb') as f:
                data = pickle.load(f)

            for key in keys:
                u_est = data[key][0, 0].view(1, -1).cpu().numpy().tolist()[0]
                v_est = data[key][0, 1].view(1, -1).cpu().numpy().tolist()[0]
                u_over_models[key].append(u_est)
                v_over_models[key].append(v_est)

        for key in keys:
            mean_u, var_u = np.mean(u_over_models[key], axis=0), np.var(u_over_models[key], axis=0)
            mean_v, var_v = np.mean(v_over_models[key], axis=0), np.var(v_over_models[key], axis=0)
            scene_uncertainty_u[key] = np.vstack((mean_u, var_u))
            scene_uncertainty_v[key] = np.vstack((mean_v, var_v))

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
    while True:
        viewpoint = random.sample(viewpoints, 1)[0]
        uncertainty_res_u = {}
        uncertainty_res_v = {}

        keys = ["flow4", "flow3", "flow2", "flow_final"]
        u_over_models = defaultdict(list)
        v_over_models = defaultdict(list)

        u_mean, u_var = defaultdict(list), defaultdict(list)
        v_mean, v_var = defaultdict(list), defaultdict(list)

        for model in MODELS:
            u_over_scenes = defaultdict(float)  # {"flow4": 0., "flow3": 0., "flow2": 0., "flow_final": 0.}
            v_over_scenes = defaultdict(float)  # {"flow4": 0., "flow3": 0., "flow2": 0., "flow_final": 0.}
            for i, scene in enumerate(scenes):
                fname = osp.join(PRECOMPUTED_RES_PATH, 'model_' + model, scene + '_' + viewpoint + '_flows.pkl')
                with open(fname, 'rb') as f:
                    data = pickle.load(f)

                for key in keys:
                    u_over_scenes[key] += data[key][0, 0].view(1, -1).cpu().numpy()
                    v_over_scenes[key] += data[key][0, 1].view(1, -1).cpu().numpy()

            for key in keys:
                u_over_scenes[key] /= len(scenes)
                v_over_scenes[key] /= len(scenes)

                u_over_models[key].append(u_over_scenes[key])
                v_over_models[key].append(v_over_scenes[key])

        for key in keys:
            mean_arr_u, var_arr_u = np.mean(u_over_models[key], axis=0), np.var(u_over_models[key], axis=0)
            mean_arr_v, var_arr_v = np.mean(v_over_models[key], axis=0), np.var(v_over_models[key], axis=0)
            u_mean[key], u_var[key] = mean_arr_u, var_arr_u
            v_mean[key], v_var[key] = mean_arr_v, var_arr_v

        key = keys[0]
        fig = plt.figure()
        plt.plot(range(len(u_mean[key])), u_mean[key])
        plt.fill_between(range(len(u_mean[key])),
                         np.array(u_mean[key]) - 2 * np.array(u_var[key]) ** 0.5,
                         np.array(u_mean[key]) + 2 * np.array(u_var[key]) ** 0.5,
                         alpha=0.1)
        plt.savefig('1.png')
    '''