from Lightning_Nets import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
import argparse
import torch.utils.data as data
from Triplet_DataLoader import Spectrogram_Loader
from Metric_losses_lightning import batch_hard_triplet_loss
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
sns.set_style("darkgrid")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class VGG_EmbeddingNet(pl.LightningModule):
    def __init__(self, loss_criterion, args, kwargs, learning_rate):
        super().__init__()
        self.dimension = 512

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

        self.loss_criterion = loss_criterion
        self.args = args
        self.kwargs = kwargs
        self.learning_rate = learning_rate

    def train_dataloader(self):
        return torch.utils.data.DataLoader(Spectrogram_Loader(filename=self.args.train_set
                                                              , mel=self.args.melspectrogram),
                                   batch_size=self.args.train_batch_size, shuffle=True, **self.kwargs)
    #def test_dataloader(self):
    #    return torch.utils.data.DataLoader(Spectrogram_Loader(filename=self.args.test_set,
    #                                                          mel=self.args.melspectrogram),
    #                                       batch_size = self.args.test_batch_size, shuffle=False, **self.kwargs)
    def val_dataloader(self):
        return torch.utils.data.DataLoader(Spectrogram_Loader(filename=self.args.test_set,
                                                              mel=self.args.melspectrogram),
                                           batch_size=self.args.test_batch_size, shuffle=False, **self.kwargs)
    def forward(self, x):
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
        #print('This is x',x.size())
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
        embeddings = F.normalize(x, dim=1)
        return embeddings

    def loss_function(self, embeddings, int_labels):
        loss = self.loss_criterion(int_labels, embeddings)
        return loss

    def training_step(self, batch, batch_idx):
        x, int_labels, _ = batch
        embedding = self(x)
        #print('This is the embedding', embedding.size())
        loss, hardest_positive, hardest_negative = self.loss_function(embedding, int_labels)
        #correct_negative, total = torch.tensor(correct_negative).float(), torch.tensor(total).float()
        self.train_losses.append(loss)
        self.positive_dist.append(hardest_positive)
        self.negatives_dist.append(hardest_negative)
        return {'loss':loss}

    def on_train_epoch_start(self):
        self.train_losses = []
        self.positive_dist = []
        self.negatives_dist = []


    def on_train_epoch_end(self):
        avg_losses = torch.stack(self.train_losses).mean()
        avg_positive = torch.stack(self.positive_dist).mean()
        avg_negative = torch.stack(self.negatives_dist).mean()
        wandb.log({'train_loss':avg_losses, 'postive_distances_train':avg_positive, 'negative_distances_train':avg_negative}, step=self.current_epoch)
        return({'loss':avg_losses})

    def validation_step(self, batch, batch_idx):
        x, int_labels,_ = batch
        x = F.normalize(self(x), dim=1)
        loss, harders_positive, hardest_negative = self.loss_function(x, int_labels)
        #correct_negative = torch.tensor(correct_negative)
        #total = torch.tensor(total)
        return {'val_loss':loss, 'positive_distances':harders_positive, 'negative_distances':hardest_negative}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_positive = torch.stack([x['positive_distances'] for x in outputs]).mean()
        avg_negative = torch.stack([x['negative_distances'] for x in outputs]).mean()
        #total = torch.stack([x['total'] for x in outputs]).sum().float()
        wandb.log({'val_loss':avg_loss, 'positive_distances_test':avg_positive, 'negative_distances_test':avg_negative}, step=self.current_epoch)
        if (self.current_epoch % 20) == 0:
            torch.save(self.state_dict(), self.args.model_path+'_margin:{}_full'.format(self.args.margin))
            #self.plot_embeddings()
            #b.log({'coverage': coverage, 'purity': purity}, step=self.current_epoch)

        return {'val_loss':avg_loss}

    def plot_embeddings(self):
        #self.model.eval()
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
        print('There are {} unique labels'.format(len(unique_labels)))
        colors = []
        for i in range(len(unique_labels)):
            colors.append('#%06X' % randint(0, 0xFFFFFF))
        color_list = []
        for label in labels:
            color_list.append(colors[unique_labels.index(label)])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        #This is just a test, is it working?
        embeddings = embeddings.cpu().numpy()
        embeddings = StandardScaler().fit_transform(embeddings)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        print('----------------')
        print(coords)
        print('---------------')
        #kmeans = KMeans(n_clusters=len(unique_labels))
        #kmeans.fit(embeddings)
        #y_kmeans = kmeans.predict(embeddings)

        cluster_df = pd.DataFrame({'x': [x for x in coords[:, 0]],
                                   'y': [y for y in coords[:, 1]],
                                   'labels': labels,
                                   'color': color_list})
                                   #'cluster' : y_kmeans})

        fig, _ = plt.subplots()
        fig.set_size_inches(22, 12)
        p1 = sns.scatterplot(x="x",
                             y="y",
                             hue="color",
                             s=1,
                             legend=None,
                             # scatter_kws={'s': 1},
                             data=cluster_df)
        for line in range(0, coords.shape[0]):
            p1.text(cluster_df["x"][line],
                    cluster_df['y'][line],
                    '  ' + cluster_df["labels"][line].title(),
                    horizontalalignment='left',
                    verticalalignment='bottom', size='small',
                    color=cluster_df['color'][line],
                    fontsize=1,
                    weight='normal'
                    ).set_size(10)
        plt.title('PCA plot for speaker embeddings trained with triplet loss on SincNet (256 dimensional embedding)',
                  fontsize=24)

        plt.tick_params(labelsize=20)
        plt.xlabel('pca-one', fontsize=20)
        plt.ylabel('pca-two', fontsize=20)
        plt.savefig('./figures/pca-plot-VGG-epoch:{}-margin:{}.png'.format(self.current_epoch,self.args.margin))

        '''purity = 0
        coverage = 0
        for i in list(set(y_kmeans)):
            labels = cluster_df[cluster_df['cluster'] == i]
            cluster_count = len(labels)
            correct_clusters = labels['labels'].value_counts()[0]
            cluster_label = labels['labels'].value_counts().index[0]
            label_count = len(cluster_df[cluster_df.labels == cluster_label])
            purity = purity + (correct_clusters/cluster_count)
            coverage = coverage + (correct_clusters/label_count)
        coverage = coverage/len(list(set(y_kmeans)))
        purity = purity/len(list(set(y_kmeans)))'''



    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), self.learning_rate, alpha=0.8, momentum=0.5, weight_decay=5e-4)

def train(trainer, model):
    trainer.fit(model)

class Custom_EarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        self._run_early_stopping_check(trainer=trainer, pl_module=pl_module)




def main():
    wandb.init()
    parser = argparse.ArgumentParser(description="VGGVox CNN with Spectrograms for Speaker Verification")
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing')

    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed')
    parser.add_argument('--margin', type=float, default=0.5, metavar='M',
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
    args = parser.parse_args()
    wandb.config.update(args)



    kwargs = {'num_workers': 8, 'pin_memory': True}
    model = VGG_EmbeddingNet(loss_criterion=batch_hard_triplet_loss(margin_negative=args.margin, squared=True), args=args, kwargs=kwargs, learning_rate=args.lr)
    wandb.watch(models=model)


    early_stop_callback = Custom_EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=30,
        verbose=True,
        mode='max'
    )

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, precision=32, distributed_backend='dp', auto_lr_find=False, num_sanity_val_steps=0)#, early_stop_callback=early_stop_callback)
    train(trainer=trainer, model=model)

if __name__ == '__main__':
    main()


