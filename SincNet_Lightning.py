from Lightning_Nets import *
from Triplet_DataLoader import Window_Loader
import wandb
import torch.utils.data as data
import torch.optim as optim
import time
from Metric_losses_lightning import batch_hard_triplet_loss
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
from SincNet_dataio import ReadList, read_conf, str_to_bool
from statistics import mean
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from random import randint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
sns.set_style("darkgrid")


class Full_SincNet(pl.LightningModule):
    def __init__(self, SincNet_model, MLP_model, args):
        super(Full_SincNet,self).__init__()
        self.SincNet_model = SincNet_model
        self.MLP_model = MLP_model
        self.loss_criterion = batch_hard_triplet_loss(8, True)
        self.args = args
        self.kwargs = {'num_workers':8, 'pin_memory':True}

    def train_dataloader(self):
        return torch.utils.data.DataLoader(Window_Loader(filename=args.train_set,
                                                         windowed=True,
                                                         window_length=0.2,
                                                         overlap=0.01),
                                           batch_size=args.train_batch_size,
                                           shuffle=True,
                                           **self.kwargs)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(Window_Loader(filename=args.test_set,
                                                         windowed=True,
                                                         window_length=0.2,
                                                         overlap=0.01),
                                           batch_size=args.test_batch_size,
                                           shuffle=False,
                                           **self.kwargs)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(Window_Loader(filename=args.valid_set,
                                                         windowed=True,
                                                         window_length=0.2,
                                                         overlap=0.01),
                                           batch_size=args.test_batch_size,
                                           shuffle=False,
                                           **self.kwargs)
    def forward(self, x):
        batch_size, windows, samples = list(x.size())[0], list(x.size())[1], list(x.size())[2]
        x = x.view(-1, batch_size*windows, samples).squeeze()
        embeddings = self.MLP_model(self.SincNet_model(x)).view(batch_size, windows, 258)
        embeddings = torch.mean(embeddings, dim=1, keepdim=True).squeeze()
        return embeddings

    def loss_function(self,embeddings, int_labels):
        loss = self.loss_criterion(int_labels, embeddings)
        return loss

    def training_step(self, batch, batch_idx):
        x, int_labels, _ = batch
        x = self(x)
        loss, correct_negative,total= self.loss_function(x, int_labels)
        logs = {'train_loss':loss, 'train_accuracy':(correct_negative/total)*100}
        if self.current_epoch % 20 == 0:
            self.plot_embeddings()
        return {'loss':loss, 'log':logs}


    def plot_embeddings(self):
        self.eval()
        embeddingList = []
        with torch.no_grad():
            for batch_idx, (x, int_labels, string_labels) in enumerate(self.val_dataloader()):
                x, int_labels = x.cuda(), int_labels.cuda()
                embeddings = self(x)
                embeddingList.append((embeddings, string_labels))
        embeddings = [embedding[0] for embedding in embeddingList]
        embeddings = torch.cat(embeddings, dim=0)
        labels = []
        for listing in [embedding[1] for embedding in embeddingList]:
            for item in listing:
                labels.append(item)
        unique_labels = list(set(labels))
        colors = []
        for i in range(len(unique_labels)):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        color_list = []
        for label in labels:
            color_list.append(colors[unique_labels.index(label)])
        embeddings = embeddings.cpu().numpy()
        coords = TSNE
        coords = TSNE(n_components=2, random_state=47, perplexity=20, n_iter=4000).fit_transform(embeddings)
        print(coords)
        coords = pd.DataFrame({'x': [x for x in coords[:, 0]],
                               'y': [y for y in coords[:, 1]],
                               'labels': labels,
                               'color': color_list})

        fig, _ = plt.subplots()
        fig.set_size_inches(22, 12)
        p1 = sns.scatterplot(x="x",
                             y="y",
                             hue="color",
                             s=1,
                             legend=None,
                             # scatter_kws={'s': 1},
                             data=coords)
        for line in range(0, coords.shape[0]):
            p1.text(coords["x"][line],
                    coords['y'][line],
                    '  ' + coords["labels"][line].title(),
                    horizontalalignment='left',
                    verticalalignment='bottom', size='small',
                    color=coords['color'][line],
                    fontsize=1,
                    weight='normal'
                    ).set_size(10)
        plt.title('t-SNE plot for speaker embeddings trained with triplet loss on VGG-Vox (256 dimensional embedding)',
                  fontsize=24)

        plt.tick_params(labelsize=20)
        plt.xlabel('tsne-one', fontsize=20)
        plt.ylabel('tsne-two', fontsize=20)
        plt.savefig('tsne-plot-{}.png'.format(self.current_epoch))
        wandb.log({"Validation set at epoch {}".format(self.current_epoch): plt})


    def validation_step(self, batch, batch_idx):
        x, int_labels, _ = batch
        x = self(x)
        loss, correct_negative, total = self.loss_function(x, int_labels)
        logs = {'valid_loss':loss, 'valid_accuracy':(correct_negative/total)*100}
        return {'loss':loss, 'logs':logs}

    #def validation_step(self):
    #    if self.current_epoch % 5 == 0:
    #        print('Doing validation')

    def configure_optimizers(self):
        return torch.optim.RMSprop(list(self.SincNet_model.parameters())+list(self.MLP_model.parameters()), lr=0.0001, alpha=0.8, momentum=0.5)

def main():
    global args


    parser = argparse.ArgumentParser(description="SincNet Speaker Recognition from Raw Waveform")
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed')
    parser.add_argument('--margin', type=float, default=8, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--name', default='VGG_Spectogram_Triplet', type=str,
                        help='name of network')
    parser.add_argument('--train-set',
                        default='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt',
                        type=str,
                        help='path to train samples')
    parser.add_argument('--test-set',
                        default='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_test.txt',
                        type=str,
                        help='path to test samples')
    parser.add_argument('--valid-set',
                        default='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_valid.txt',
                        type=str,
                        help='path to validation samples')
    parser.add_argument('--model-path-sincnet',
                        default='/home/lucvanwyk/MetricEmbeddingNet/models/SincNet_Triplet',
                        type=str,
                        help='path to where sincnet models are saved/loaded')
    parser.add_argument('--model-path-mlp',
                        default='/home/lucvanwyk/MetricEmbeddingNet/models/MLP_Triplet',
                        type=str,
                        help='path to where mlp models are saved/loaded')
    parser.add_argument('--save-model', type=bool, default=True,
                        help='save model?')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='load model?')
    parser.add_argument('--cfg', type=str, default='SincNet_options_Teapot.cfg',
                        help='configuration file')
    args = parser.parse_args()

    options = read_conf()
    wandb_logger = WandbLogger(name='testingwandblogger_validandall', project='sincnet_triplet')



    # get parameters for SincNet and MLP
    # [cnn]
    # [cnn]
    cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
    cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
    cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
    cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
    cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
    cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
    cnn_act = list(map(str, options.cnn_act.split(',')))
    cnn_drop = list(map(float, options.cnn_drop.split(',')))

    # [dnn]
    fc_lay = list(map(int, options.fc_lay.split(',')))
    fc_drop = list(map(float, options.fc_drop.split(',')))
    fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
    fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
    fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
    fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
    fc_act = list(map(str, options.fc_act.split(',')))

    SincNet_args = {'input_dim': 3200,  # 3 seconds at 16000Hz
                    'fs': 16000,
                    'cnn_N_filt': cnn_N_filt,
                    'cnn_len_filt': cnn_len_filt,
                    'cnn_max_pool_len': cnn_max_pool_len,
                    'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
                    'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
                    'cnn_use_laynorm': cnn_use_laynorm,
                    'cnn_use_batchnorm': cnn_use_batchnorm,
                    'cnn_act': cnn_act,
                    'cnn_drop': cnn_drop
                    }

    SincNet_model = SincNet(SincNet_args)
    DNN1_args = {'input_dim': SincNet_model.out_dim,
                 'fc_lay': fc_lay,
                 'fc_drop': fc_drop,
                 'fc_use_batchnorm': fc_use_batchnorm,
                 'fc_use_laynorm': fc_use_laynorm,
                 'fc_use_laynorm_inp': fc_use_laynorm_inp,
                 'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
                 'fc_act': fc_act}
    MLP_model = MLP(DNN1_args)
    model = Full_SincNet(SincNet_model, MLP_model, args)
    trainer =pl.Trainer( max_epochs=10,gpus=1, precision=16, distributed_backend='dp', logger=wandb_logger)
    trainer.fit(model)


if __name__ == '__main__':
    main()
