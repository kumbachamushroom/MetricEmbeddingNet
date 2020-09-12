import torch
import os
import wandb
from Triplet_Net import VGGVox
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from Metric_Losses import batch_hard_triplet_loss
from Triplet_DataLoader import Spectrogram_Loader
import time
import random
import numpy as np
#import torch.nn as nn
from statistics import mean
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from matplotlib.lines import Line2D
sns.set_style("darkgrid")

def plot_validation_set(valid_loader, model, epoch):
    model.eval()
    embeddingList = []
    with torch.no_grad():
        for batch_idx, (spectrograms, int_labels, string_labels) in enumerate(valid_loader):
            spectrograms = spectrograms.cuda()
            embeddings = model(spectrograms)
            embeddingList.append((embeddings, string_labels))
            print("Labelling : {}/{} ".format(batch_idx, args.test_batch_size), flush=True, end='\r')
    tensor = torch.empty(sum([len(embedding[0]) for embedding in embeddingList]), 256)
    print(tensor.size())
    embeddings = [embedding[0] for embedding in embeddingList]
    embeddings = torch.cat(embeddings, dim=0)
    labels = []
    for listing in [embedding[1] for embedding in embeddingList]:
        for item in listing:
            labels.append(item)

    unique_labels = list(set(labels))
    print(unique_labels)
    colors = []
    for i in range(len(unique_labels)):
        colors.append('#%06X' % randint(0,0xFFFFFF))
    color_list = []
    for label in labels:
        color_list.append(colors[unique_labels.index(label)])



    embeddings = embeddings.cpu().numpy()
    coords = TSNE(n_components=2, random_state=47, perplexity=20, n_iter=4000).fit_transform(embeddings)
    print(coords)
    coords = pd.DataFrame({'x' : [x for x in coords[:,0]],
                           'y' : [y for y in coords[:,1]],
                           'labels' : labels,
                           'color' : color_list})

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
    plt.savefig('tsne-plot.png')
    wandb.log({"Validation set at epoch {}".format(epoch):plt})






def train(train_loader, model, optimizer, device, epoch):
    model.train()
    losses = []
    accuracy = []
    total_samples = []
    for batch_idx, (spectrograms, int_labels, _) in enumerate(train_loader):
        spectrograms, int_labels = spectrograms.to(device), int_labels.to(device)
        #print(spectrograms.size())
        embeddings = model(spectrograms)
        loss, correct_negative, total = batch_hard_triplet_loss(int_labels, embeddings,
                                                                margin_positive=8, margin_negative=8, device='cuda',
                                                                squared=True)
        total_samples.append(total)
        losses.append(loss.item())
        acc = (correct_negative/total)*100
        accuracy.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(' Training Epoch {} : [{}/{}] \t Loss: {:.4f} \t Accuracy: {:.2f} \t '
              .format(epoch, batch_idx, int(len(train_loader.dataset) / args.train_batch_size),loss, acc),
              flush=True, end='\r')
    return mean(losses), mean(accuracy)

def test(data_loader, model, device, epoch):
    model.eval()
    losses = []
    accuracy = []
    total_samples = []
    with torch.no_grad():
        for batch_idx, (spectrograms, int_labels,_) in enumerate(data_loader):
            spectrograms, int_labels = spectrograms.to(device), int_labels.to(device)
            embeddings = model(spectrograms)
            loss, correct_negative, total = batch_hard_triplet_loss(int_labels, embeddings,
                                                                    margin_negative=8, margin_positive=8, device='cuda',
                                                                squared=True)
            total_samples.append(total)
            losses.append(loss.item())
            #print(loss)
            acc = (correct_negative/total)*100
            accuracy.append(acc)
            #print(' Testing/Validating Epoch {}: \t Loss: {:.4f} \t AccuracyL {:.2f} \t'.format(epoch, loss, acc), flush=True, end='\r')

    print(' Test/Validate Epoch {}: \t Loss: {:.4f}, Accuracy: {:.2f} \t'.format(epoch, mean(losses), mean(accuracy)))
    return mean(losses), mean(accuracy)



def main():
    global args
    #wandb.login()
    wandb.init(project="vgg_triplet")
    config = wandb.config
    parser = argparse.ArgumentParser(description="VGGVox CNN with Spectrograms for Speaker Verification")
    parser.add_argument('--test-batch-size', type=int, default=40, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--train-batch-size', type=int, default=40, metavar='N',
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
    parser.add_argument('--train-set', default='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt', type=str,
                        help='path to train samples')
    parser.add_argument('--test-set', default='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_test.txt', type=str,
                        help='path to test samples')
    parser.add_argument('--valid-set', default='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_valid.txt', type=str,
                        help='path to validation samples')
    parser.add_argument('--model-path', default='/home/lucas/PycharmProjects/MetricEmbeddingNet/models/VGG_Spectrogram_Triplet.pt', type=str,
                        help='path to where models are saved/loaded')
    parser.add_argument('--save-model', type=bool, default=True,
                        help='save model?')
    parser.add_argument('--load-model', type=bool, default=False,
                        help='load model?')
    parser.add_argument('--melspectrogram', type=bool, default=False,
                        help='use melspectrogram?')
    args = parser.parse_args()

    wandb.config.update(args)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')

    kwargs = {'num_workers': 6, 'pin_memory': True}
    train_loader = data.DataLoader(Spectrogram_Loader(filename=args.train_set, mel=False),
                                   batch_size=config.train_batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(Spectrogram_Loader(filename=args.test_set, mel=False),
                                  batch_size=config.test_batch_size, shuffle=True, **kwargs)
    valid_loader = data.DataLoader(Spectrogram_Loader(filename=args.valid_set, mel=False),
                                   batch_size=config.test_batch_size, shuffle=True, **kwargs)

    model = VGGVox()
    model.to(device)

    if args.load_model:
        try:
            model.load_state_dict(torch.load(args.model_path))
        except:
            print("Could not load model {} not found".format(args.model_path))
            #nn.init.xavier_uniform(model.parameters())

    #optimizer = optim.Adam(model.parameters(), lr = config.lr)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.8, momentum=0.5)
    #optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-3, amsgrad=True)
    wandb.watch(model)

    for epoch in range(1, config.epochs+1):
        start_time = time.time()
        train_loss, train_acc = train(train_loader=train_loader, model=model, optimizer=optimizer, device=device,
                                      epoch=epoch)
        test_loss, test_acc = test(data_loader=test_loader, model=model, device=device, epoch=epoch)
        #valid_loss, valid_acc = test(data_loader=valid_loader, model=model, device=device, epoch=epoch)
        print('Finished epoch {} in {:.2f} '.format(epoch, (time.time()-start_time)))
        wandb.log({'Train Loss': train_loss, 'Train Accuracy': train_acc, 'Test Loss': test_loss,
                   'Test Accuracy': test_acc}) # 'Validation Loss': valid_loss, 'Validation Accuracy': valid_acc})
        if config.save_model and (epoch % 20 == 0):
            torch.save(model.state_dict(), config.model_path)
            print("Model saved after {} epochs".format(epoch))
            plot_validation_set(valid_loader=valid_loader, model=model, epoch=epoch)
            print("Validation plot saved")


if __name__ == '__main__':
    main()