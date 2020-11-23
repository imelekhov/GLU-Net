from os import path as osp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from datasets.training_dataset import HomoAffTpsNoizyDataset
from utils.image_transforms import ArrayToTensor, TensorToArray


if __name__ == "__main__":
    DATA_DIR = '/data/datasets'
    N_WARPS = 5
    STD = 0.0001

    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    pyramid_param = [520]

    train_dataset = HomoAffTpsNoizyDataset(n_warps=N_WARPS,
                                           std=STD,
                                           image_path=DATA_DIR,
                                           csv_file=osp.join('datasets',
                                                             'csv_files',
                                                             'train_ade.csv'),
                                           transforms=source_img_transforms,
                                           transforms_target=target_img_transforms,
                                           pyramid_param=pyramid_param,
                                           get_flow=True,
                                           output_size=(520, 520))

    dataloader = DataLoader(train_dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=8)

    for i, mini_batch in enumerate(dataloader):
        if i == 5:
            break
        fig, axes = plt.subplots(nrows=1, ncols=N_WARPS + 2, figsize=(35, 10))
        axes = axes.flatten()
        ax_id = 0
        for key in ['source_image', 'target_image']:
            img = TensorToArray(mini_batch[key].squeeze(0), type='int')
            axes[ax_id].imshow(img)
            ax_id += 1

        for img_tensor in mini_batch['target_image_nz'].squeeze(0):
            img = TensorToArray(img_tensor, type='int')
            axes[ax_id].imshow(img)
            ax_id += 1

        plt.savefig('img_jitter_' + str(i) + '.png')
