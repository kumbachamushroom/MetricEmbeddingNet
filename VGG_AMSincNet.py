from Lightning_Nets import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
import wandb
import argparse
import torch.utils.data as data
from Triplet_DataLoader import Spectrogram_Loader
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
sns.set_style("darkgrid")
import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg_utils import get_conv2d_output_shape
import numpy as np

class AdditiveMarginSoftmax(nn.Module):
    #AMSoftmax
    def __init__(self, margin=0.35, s=30):
        super().__init__()
        self.m = margin
        self.s = s
        self.epsilon = 1e-11

    def forward(self, predicted, target):
        predicted = predicted / (predicted.norm(p=2, dim=0) + self.epsilon)
        indexes = range(predicted.size(0))
        cost_theta_y = predicted[indexes, target]
        cost_theta_y_m = cost_theta_y - self.m
        exp_s = np.e ** (self.s * cost_theta_y_m)
        sum_cos_theta_j = (np.e ** (predicted * self.s)).sum(dim=1) - (np.e ** (predicted[indexes, target] * self.s))
        log = -torch.log(exp_s / (exp_s + sum_cos_theta_j + self.epsilon)).mean()
        return log

class VGG_EmbeddingNet(pl.LightningModule):
    def __init__(self, dimension=256, classes=29):
        super().__init__()

        self.dimension = dimension
        self.classes = classes

        h = 257  # 512 in VoxCeleb paper. 201 in practice.  # 128 for melspectrogram, 257 for normal spectrogram
        w = 301  # typically 3s with 10ms steps

        self.conv1_ = nn.Conv2d(1, 96, (7, 7), stride=(2, 2), padding=1, bias=True)
        # 254 x 148 when n_features = 512
        # 99 x 148 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (7, 7), stride=(2, 2), padding=1)

        self.bn1_ = nn.BatchNorm2d(96)
        self.mpool1_ = nn.MaxPool2d((3, 3), stride=(2, 2))
        # 126 x 73 when n_features = 512
        # 49 x 73 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))

        self.conv2_ = nn.Conv2d(96, 256, (5, 5), stride=(2, 2), padding=1, bias=True)
        # 62 x 36 when n_features = 512
        # 24 x 36 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (5, 5), stride=(2, 2), padding=1)

        self.bn2_ = nn.BatchNorm2d(256)
        self.mpool2_ = nn.MaxPool2d((3, 3), stride=(2, 2))
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))

        self.conv3_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1, bias=True)
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

        self.bn3_ = nn.BatchNorm2d(256)

        self.conv4_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1, bias=True)
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

        self.bn4_ = nn.BatchNorm2d(256)

        self.conv5_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1, bias=True)
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

        self.bn5_ = nn.BatchNorm2d(256)

        self.mpool5_ = nn.MaxPool2d((5, 3), stride=(3, 2))
        # 9 x 8 when n_features = 512
        # 3 x 8 when n_features = 201
        h, w = get_conv2d_output_shape((h, w), (5, 3), stride=(3, 2))

        self.fc6_ = nn.Conv2d(256, 4096, (h, 1), stride=(1, 1), bias=True)
        # 1 x 8
        h, w = get_conv2d_output_shape((h, w), (h, 1), stride=(1, 1))

        self.fc7_ = nn.Linear(4096, 1024, bias=True)
        self.fc8_ = nn.Linear(1024, self.dimension, bias=True)
        self.fc9_ = nn.Linear(self.dimension, self.classes, bias=True)


    def forward(self, sequences):
        """Embed sequences

        Parameters
        ----------
        sequences : torch.Tensor (batch_size, n_samples, n_features)
            Batch of sequences.

        Returns
        -------
        embeddings : torch.Tensor (batch_size, dimension)
            Batch of embeddings.
        """

        x = sequences
        # print(x)
        # print(x.size())
        # x = torch.transpose(sequences, 1, 2).view(
        # 30, 1, 3, 3)

        # conv1. shape => 254 x 148 => 126 x 73
        x = self.mpool1_(F.relu(self.bn1_(self.conv1_(x))))

        # conv2. shape =>
        x = self.mpool2_(F.relu(self.bn2_(self.conv2_(x))))

        # conv3. shape = 62 x 36
        x = F.relu(self.bn3_(self.conv3_(x)))

        # conv4. shape = 30 x 17
        x = F.relu(self.bn4_(self.conv4_(x)))

        # conv5. shape = 30 x 17
        x = self.mpool5_(F.relu(self.bn5_(self.conv5_(x))))

        # fc6. shape =
        x = F.dropout(F.relu(self.fc6_(x)))

        # (average) temporal pooling. shape =
        x = torch.mean(x, dim=-1)

        # fc7. shape =
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc7_(x)))

        # fc8. shape =
        x = self.fc8_(x)
        x = F.softmax(self.fc9_(x))
        return x

    def train_dataloader(self):
        return torch.utils.data.DataLoader(Spectrogram_Loader(filename=args.train_set,
                                                              mel=args.melspectrogram),
                                           batch_size=args.train_batch_size,
                                           shuffle=True,
                                           **{'num_workers':args.num_workers, 'pin_memory':True})

    def test_dataloader(self):
        return torch.utils.data.DataLoader(Spectrogram_Loader(filename=args.test_set,
                                                              mel=args.melspectrogram),
                                           batch_size=args.test_batch_size,
                                           shuffle=False,
                                           **{'num_workers':args.num_workers, 'pin_memory':True})

    def val_dataloader(self):
        return torch.utils.data.DataLoader(Spectrogram_Loader(filename=args.valid_set,
                                                              mel=args.melspectrogram),
                                           batch_size=args.test_batch_size, shuffle=False,
                                           **{'num_workers':args.num_workers, 'pin_memory':True})
    def loss_function(self, pout, labels):
        loss = loss_criterion(pout, labels.long())
        return loss

    def training_step(self, batch, batch_idx):
        x, labels,_ = batch
        pout = self(x)
        pred = torch.max(pout, dim=1)[1]
        loss = self.loss_function(pout, labels)
        return {'loss':loss}

    def validation_step(self, batch, batch_idx):
        x, labels, _ = batch
        pout = self(x)
        pred = torch.max(pout, labels)
        loss = self.loss_function(pout, labels)
        return {'val_loss':loss}

    def configure_optimizers(self):


def main():
    wandb.init()
    global args
    global loss_criterion
    loss_criterion = AdditiveMarginSoftmax(margin=0.35, s=30)
    parser = argparse.ArgumentParser(description="VGGVox trainined with the Additive Margin SincNet Loss")
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                       help='input batch size for training')
    parser.add_argument('--train-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed')
    parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--name', default='VGG_Spectogram_Triplet', type=str,
                        help='name of network')
    parser.add_argument('--train-set',
                        default='/home/lucvanwyk/Data/pyannote/Extracted_Speech/trimmed_pyannote_sample_list_train.txt',
                        type=str,
                        help='path to train samples')
    parser.add_argument('--test-set',
                        default='/home/lucvanwyk/Data/pyannote/Extracted_Speech/trimmed_pyannote_sample_list_test.txt',
                        type=str,
                        help='path to test samples')
    parser.add_argument('--valid-set',
                        default='/home/lucvanwyk/Data/pyannote/Extracted_Speech/trimmed_sample_list_full_valid_full.txt',
                        type=str,
                        help='path to validation samples')
    parser.add_argument('--model-path',
                        default='/home/lucvanwyk/MetricEmbeddingNet/models/VGG_Spectrogram_Triplet',
                        type=str,
                        help='path to where models are saved/loaded')
    parser.add_argument('--save-model', type=bool, default=True,
                        help='save model?')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='load model?')
    parser.add_argument('--melspectrogram', type=bool, default=False,
                        help='use melspectrogram?')
    parser.add_argument('--num-workers', type=bool, default=False,
                        help='number of workers to use for dataloaders')
    args = parser.parse_args()
    wandb.config.update(args)


    ground_truth_labels = list(set([int(line.split()[2]) for line in open(args.train_set)]))
    model = VGG_EmbeddingNet(dimension=512, classes=len(ground_truth_labels))

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, precision=32, distributed_backend='dp', auto_lr_find=False, num_sanity_val_steps=0)
    trainer.fit(model=model)
    #path = '/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt'


if __name__ == '__main__':
    main()