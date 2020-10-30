import os
from os import path as osp
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle

from models.models_compared import GLU_Net
from datasets.hpatches import HPatchesdataset
from utils.image_transforms import ArrayToTensor


if __name__ == '__main__':
    DATA_PATH = '/data/datasets/hpatches-sequences-release/'
    PROJECT_PATH = '/data/projects/GLU-Net'
    MODELS = ['42', '777', '1984', '2020', '4224']
    SNAPSHOT = osp.join(PROJECT_PATH, 'snapshots')
    CSV_PATH = osp.join(PROJECT_PATH, 'datasets', 'csv_files')
    OUTPUT_PATH = osp.join(PROJECT_PATH, 'uncertainty_output')
    N_VIEWS = 5  # 1-2, 1-3, 1-4, 1-5, 1-6
    ORIG_IMG_SIZE = True

    if not osp.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    input_images_transform = transforms.Compose([ArrayToTensor(get_float=False)])
    gt_flow_transform = transforms.Compose([ArrayToTensor()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for model_id in MODELS:
            print('Starting model_' + model_id)
            network = GLU_Net(path_pre_trained_models=osp.join(SNAPSHOT, "GLUNet_train_" + model_id, "model_best.pth"),
                              consensus_network=False,
                              cyclic_consistency=True,
                              iterative_refinement=True,
                              apply_flipping_condition=False)

            out_path = osp.join(OUTPUT_PATH, 'model_' + model_id)
            if not osp.isdir(out_path):
                os.makedirs(out_path)

            for k in range(2, N_VIEWS + 2):
                test_set = HPatchesdataset(DATA_PATH,
                                           osp.join(CSV_PATH, 'hpatches_1_{}.csv'.format(k)),
                                           input_images_transform,
                                           gt_flow_transform,
                                           None,
                                           original_size=ORIG_IMG_SIZE)
                test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

                for _, mini_batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    source_img = mini_batch['source_image']
                    target_img = mini_batch['target_image']
                    scene_id = mini_batch['scene'][0]
                    flow4, flow3, flow2, flow1 = network.estimate_flow(source_img, target_img, device)

                    res = {"flow4": flow4, "flow3": flow3, "flow2": flow2, "flow_final": flow1}
                    with open(osp.join(out_path, scene_id + '_' + '1' + str(k) + "_flows.pkl"), 'wb') as f:
                        pickle.dump(res, f)
        print('Done')