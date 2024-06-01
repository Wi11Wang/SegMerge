import sys
sys.path.append("..")
from utils import ParticlePackDataset

import os

import numpy as np
import scipy.ndimage as ndi
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from comet_ml import Experiment
from sklearn.metrics import confusion_matrix
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR


class SiameseNetwork(nn.Module):
    """
        Siamese network for image similarity estimation.
    """

    def __init__(self, backbone: str = 'resnet34', fc_size: int = 256):
        super(SiameseNetwork, self).__init__()
        self.backbone = backbone
        self.fc_size = fc_size
        # Set backbone resnet model
        if self.backbone.lower() == 'resnet18':
            self.resnet = torchvision.models.resnet18(weights=None)
        elif self.backbone.lower() == 'resnet34':
            self.resnet = torchvision.models.resnet34(weights=None)
        elif self.backbone.lower() == 'resnet50':
            self.resnet = torchvision.models.resnet50(weights=None)
        elif self.backbone.lower() == 'resnet101':
            self.resnet = torchvision.models.resnet101(weights=None)
        elif self.backbone.lower() == 'resnet152':
            self.resnet = torchvision.models.resnet152(weights=None)
        else:
            raise ValueError(f'Unknown backbone: {self.backbone}')
        # over-write the first conv layer to be able to read gray scale images
        # as resnet reads (3,x,x) where 3 is RGB channels
        # whereas we have (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        # remove the last layer of resnet18 (linear layer before average pooling layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, fc_size),
            nn.ReLU(inplace=True),
            nn.Linear(fc_size, 1),
            nn.Sigmoid()
        )
        # initialize the weights
        self.resnet.apply(SiameseNetwork.init_weights)
        self.fc.apply(SiameseNetwork.init_weights)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get the features of two images
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        # find the difference between two images' features
        output = torch.abs(torch.subtract(output1, output2))
        output = self.fc(output)
        return output

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class SiameseDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.reference_path = os.path.join(path, 'reference')
        self.compare_path = os.path.join(path, 'compare')
        self.references = [name for name in os.listdir(self.reference_path) if name.endswith('.png')]
        self.compares = [name for name in os.listdir(self.compare_path) if name.endswith('.png')]
        self.len = len(self.compares) * 2
        self.transforms = lambda x: x if transforms is None else transforms(x)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_idx = idx // 2
        cmp_name = self.compares[img_idx]
        # load compare image
        cmp_img = Image.open(os.path.join(self.compare_path, cmp_name))
        # if index is even number: load different pair
        if idx % 2 == 0:
            np.random.seed(idx)
            ref_name = np.random.choice(self.compares)
            for _ in range(10):
                ref_name = np.random.choice(self.compares)
                if ref_name[6:10] != cmp_name[6:10]:
                    break
            ref_img = Image.open(os.path.join(self.compare_path, ref_name))
            return self.transforms(ref_img), self.transforms(cmp_img), np.float32(0.0)
        # if index is odd number: load same pair
        else:
            ref_name = cmp_name[:11] + '000.png'
            ref_img = Image.open(os.path.join(self.reference_path, ref_name))
            return self.transforms(ref_img), self.transforms(cmp_img), np.float32(1.0)

    def get_metadata(self, train=True):
        return {
            'train' if train else 'valid' + 'path': self.path,
            'train' if train else 'valid' + 'dataset size': self.len
        }

    @staticmethod
    def prepare(mask_path, save_path, size, z_range=slice(None), preparation_step=2, n_workers=8):
        os.makedirs(os.path.join(save_path, 'reference'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'compare'), exist_ok=True)
        thresh = 4
        mask = ParticlePackDataset.load_memmap(mask_path)[z_range]
        n_slices = len(mask)
        helper = partial(SiameseDataset._prepare, mask_path, save_path, size, z_range, thresh)
        with Pool(n_workers) as p:
            list(tqdm(p.imap(helper, range(0, n_slices, preparation_step)), total=n_slices // preparation_step))

    @staticmethod
    def _prepare(mask_path, save_path, size, z_range, thresh, slice_idx):
        mask = ParticlePackDataset.load_memmap(mask_path)[z_range]
        fst_slice = mask[slice_idx]
        snd_slice = mask[slice_idx + 1]
        fst_slice_unique_labels = np.unique(fst_slice)[1:]
        snd_slice_unique_labels = np.unique(snd_slice)[1:]
        shared_labels = np.intersect1d(fst_slice_unique_labels[1:], snd_slice_unique_labels[1:])
        if len(shared_labels) <= 3:
            return
        for label_idx in range(len(shared_labels)):
            save_fmt = lambda x: f'{slice_idx:05d}_{label_idx:04d}_{x:03d}.png'
            label = shared_labels[label_idx]
            # reference image
            ref_img = SiameseDataset.extract_resize(fst_slice, label, size)
            # compare image
            cmp_img_extract = SiameseDataset.extract(snd_slice, label)
            cmp_img_extract_resized = SiameseDataset.resize(SiameseDataset.extract(snd_slice, label), size)
            # if the area is too small, not add to the dataset
            if np.sum(np.asarray(ref_img)) < thresh * 2 or np.sum(np.asarray(cmp_img_extract_resized)) < thresh * 2:
                continue
            # Save reference image
            ref_img.save(os.path.join(save_path, 'reference', save_fmt(0)))
            cmp_img_extract_resized.save(os.path.join(save_path, 'compare', save_fmt(1)))
            split_masks, n_splits = ndi.label(cmp_img_extract)
            # if n splits > 1: we have split particles
            if n_splits == 1:
                continue
            split_idx = 2
            for j in range(1, n_splits + 1):
                split_img = SiameseDataset.extract_resize(split_masks, j, size)
                if np.sum(np.asarray(split_img)) >= thresh:
                    split_img.save(os.path.join(save_path, 'compare', save_fmt(split_idx)))
                    split_idx += 1

    @staticmethod
    def crop_resize(mask, label, size):
        indices = np.nonzero(mask == label)
        x_indices = indices[0]
        y_indices = indices[1]
        x_start, x_end = x_indices.min(), x_indices.max() + 1
        y_start, y_end = y_indices.min(), y_indices.max() + 1
        cropped_mask = mask[x_start: x_end, y_start: y_end].copy()
        cropped_mask[cropped_mask != label] = 0
        cropped_mask[cropped_mask == label] = 1
        return Image.fromarray(cropped_mask.astype(np.uint8)).resize((size, size))

    @staticmethod
    def extract_resize(mask, label, size):
        extracted = SiameseDataset.extract(mask, label)
        return SiameseDataset.resize(extracted, size)

    @staticmethod
    def extract(mask_2d, label):
        return np.array(mask_2d == label).astype(np.uint8)

    @staticmethod
    def resize(mask, size):
        return Image.fromarray(mask).resize((size, size))


class Siamese:
    @staticmethod
    def train(train_dataset: SiameseDataset, valid_dataset: SiameseDataset,
              backbone: str = 'resnet18', fc_size: int = 256,
              loss_fn: str = 'bce', optimiser: str = 'adadelta', lr: float = 1.0, gamma: float = 0.7,
              batch_size: int = 64, num_workers: int = 0, epochs: int = 20, seed: int = None,
              checkpoint_path: str = None, exp_name: str = None):
        # Setup comet info
        experiment = None
        if exp_name is not None:
            experiment = Experiment(
                api_key="khMypDldEXE1xhxqD6P2lbCMh",
                project_name="siamese",
                workspace="u7018753",
                log_git_patch=False,
                auto_metric_logging=False
            )
            experiment.log_parameters(
                {
                    # data related
                    **train_dataset.get_metadata(train=True),
                    **valid_dataset.get_metadata(train=False),
                    # model related
                    "backbone": backbone,
                    "fc_size": fc_size,
                    "loss function": loss_fn,
                    "optimiser": optimiser,
                    "learning rate": lr,
                    "gamma": gamma,
                    "batch size": batch_size,
                    "num workers": num_workers,
                    "epochs": epochs,
                    "seed": seed,
                }
            )
            experiment.set_name(exp_name)
            if checkpoint_path is not None:
                checkpoint_path = os.path.join(checkpoint_path, exp_name)

        if seed is not None:
            torch.manual_seed(seed)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                                   pin_memory=True, num_workers=num_workers)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SiameseNetwork(backbone=backbone, fc_size=fc_size).to(device)
        if loss_fn.lower() == 'bce':
            loss_fn = nn.BCELoss()
        else:
            raise ValueError(f'Loss function {loss_fn} not supported')
        if optimiser == 'adadelta':
            optimiser = torch.optim.Adadelta(model.parameters(), lr=lr)
        elif optimiser == 'adam':
            optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        valid_results = {}
        best_valid_res = 0
        # Training loop
        scheduler = StepLR(optimiser, step_size=1, gamma=gamma)
        for epoch in range(1, epochs + 1):
            Siamese._train_one_epoch(train_loader, model, loss_fn, optimiser, experiment, epoch, device,
                                     checkpoint_path)
            valid_res = Siamese._valid(valid_loader, model, loss_fn, experiment, epoch, device)
            valid_results[epoch] = valid_res
            if experiment is not None:
                experiment.log_metric(name='Learning Rate', value=scheduler.get_last_lr(), step=epoch)
            # Save model after validation
            if checkpoint_path is not None:
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_path, 'last.pt'))
                if valid_res[0] > best_valid_res:
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'best.pt'))
                    best_valid_res = valid_res[0]
            scheduler.step()
        if experiment is not None:
            experiment.end()
        return valid_results

    @staticmethod
    def _train_one_epoch(train_loader, model, loss_fn, optimiser, experiment, epoch, device, checkpoint_path):
        model.train()
        batch_losses = []
        # init progress bar
        n_batches = len(train_loader)
        update_freq = max(int(n_batches * 0.01), 1)
        pbar = tqdm(total=n_batches, desc=f'Epoch {epoch}', ncols=120, leave=True)
        # training loop
        for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            optimiser.zero_grad()
            outputs = model(images_1, images_2).squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimiser.step()
            batch_losses.append(loss.item())
            # update progress bar
            if batch_idx > 0 and batch_idx % update_freq == 0:
                pbar.n = batch_idx
                pbar.set_description(f'Epoch {epoch}: Step loss: {loss.item():.6f}')
                # update to comet
                if experiment is not None:
                    experiment.log_metric(name='Training loss', value=loss.item(), step=batch_idx, epoch=epoch)
        # update to comet
        epoch_loss = np.mean(batch_losses)
        if experiment is not None:
            experiment.log_metric(name='Training loss', value=epoch_loss, step=epoch)

        # update progress bar
        pbar.n = pbar.total - 1
        pbar.update()
        pbar.set_description(f'Epoch {epoch} training finished. Epoch loss: {epoch_loss:.6f}')
        pbar.close()

    @staticmethod
    def _valid(valid_loader, model, loss_fn, experiment, epoch, device):
        model.eval()
        sum_valid_loss, n_corrects = 0, 0
        ground_truths = []
        predictions = []
        confidences = []

        # Lists for wrongly predicted images and their predictions
        wrongly_predicted_images_1 = []
        wrongly_predicted_images_2 = []
        wrong_predictions = []
        gt_predictions = []
        mean_pred_confidence = 0

        # init progress bar
        n_batches = len(valid_loader)
        update_freq = max(int(n_batches * 0.01), 1)
        pbar = tqdm(total=n_batches, desc=f'Epoch {epoch}', ncols=120, leave=True)
        with torch.no_grad():
            # validation loop
            for batch_idx, (images_1, images_2, targets) in enumerate(valid_loader):
                images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
                outputs = model(images_1, images_2).squeeze()
                sum_valid_loss += loss_fn(outputs, targets).sum().item()  # sum up batch loss
                pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
                corrects = pred.eq(targets.view_as(pred)).sum().item()
                n_corrects += corrects
                mean_pred_confidence += torch.mean(outputs[outputs > 0.5])
                confidences.extend(list(outputs[outputs > 0.5].cpu()))
                # Update for wrong predictions
                wrong_idxs = (pred != targets.view_as(pred)).nonzero(as_tuple=True)[0]
                if len(wrong_idxs) > 0 and len(wrongly_predicted_images_1) < 200:
                    wrongly_predicted_images_1.extend(images_1[wrong_idxs].cpu())
                    wrongly_predicted_images_2.extend(images_2[wrong_idxs].cpu())
                    wrong_predictions.extend(outputs[wrong_idxs].cpu().numpy())
                    gt_predictions.extend(targets[wrong_idxs].cpu().numpy())
                ground_truths.append(targets.cpu().numpy())
                predictions.append(pred.cpu().numpy())
                # update progress bar
                if batch_idx > 0 and batch_idx % update_freq == 0:
                    pbar.n = batch_idx
                    pbar.set_description(f'Epoch {epoch}: Batch acc: {corrects / len(outputs) * 100:.2f}%: Conf: {mean_pred_confidence:.2f}')
        # calculate validation accuracy and loss
        ground_truths = np.concatenate(ground_truths, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        accuracy = np.mean(ground_truths == predictions)
        sum_valid_loss /= len(valid_loader.dataset)
        # update to comet
        if experiment is not None:
            experiment.log_confusion_matrix(matrix=confusion_matrix(ground_truths, predictions),
                                            title="Validation Confusion Matrix")
            experiment.log_metric(name='Validation accuracy', value=accuracy, step=epoch)
            experiment.log_metric(name='Validation loss', value=sum_valid_loss, step=epoch)
            experiment.log_metric(name='Mean confidence', value=np.mean(confidences), step=epoch)
        # update progress bar
        pbar.n = pbar.total - 1
        pbar.update()
        pbar.set_description(f'Epoch {epoch} validation finished. '
                             f'Validation acc: {accuracy * 100:.2f}%')
        pbar.close()
        return accuracy, wrongly_predicted_images_1, wrongly_predicted_images_2, wrong_predictions, gt_predictions

    @staticmethod
    def inference(model_path, im1, im2):
        model = SiameseNetwork(backbone='resnet34', fc_size=256)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model.to('cuda')
        transforms = v2.Compose([
            v2.ToPILImage(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((56, 56),
                      interpolation=v2.InterpolationMode.BILINEAR,
                      antialias=False)
        ])
        im1[im1 > 0] = 1
        im2[im2 > 0] = 1
        # im1 = transforms(SiameseDataset.crop_2d(im1, 1)).unsqueeze(0).to('cuda')
        # im2 = transforms(SiameseDataset.crop_2d(im2, 1)).unsqueeze(0).to('cuda')
        # return model(im1, im2)

    @staticmethod
    def crop_around(mask1, mask2):
        merged_mask = torch.logical_or(mask1, mask2)
        rows = torch.any(merged_mask, dim=1)
        cols = torch.any(merged_mask, dim=0)

        # Find the boundaries
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]

        # Crop the binary mask to the bounding box
        return mask1[ymin:ymax + 1, xmin:xmax + 1], mask2[ymin:ymax + 1, xmin:xmax + 1]
