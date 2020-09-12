from Triplet_Net import *
from Triplet_DataLoader import Window_Loader
import wandb
import torch.utils.data as data
import torch.optim as optim
import time
from Metric_Losses import batch_hard_triplet_loss
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
from SincNet_dataio import ReadList, read_conf, str_to_bool
from statistics import mean

def train_windowed(SincNet_model, MLP_model, optimizer_SincNet, optimizer_MLP, device, epoch, train_loader):
    SincNet_model.train()
    MLP_model.train()
    losses = []
    accuracy = []
    total_samples = []
    for batch_idx, (tracks, int_labels,_) in enumerate(train_loader):
        tracks, int_labels = tracks.to(device), int_labels.to(device)
        batch_size, windows, samples = list(tracks.size())[0], list(tracks.size())[1], list(tracks.size())[2]
        tracks = tracks.view(-1, batch_size*windows, samples).squeeze()
        embeddings = MLP_model(SincNet_model(tracks)).view(batch_size, windows, samples)
        mean_embeddings = torch.mean(embeddings, dim=1, keepdim=True).squeeze() #64,258
        loss, correct_negative, total = batch_hard_triplet_loss(int_labels, mean_embeddings, margin_positive=8,
                                                                margin_negative=8, device=device, squared=True)
        total_samples.append(total)
        losses.append(loss.detch())
        acc = (correct_negative/total)*100
        accuracy.append(acc)
        optimizer_MLP.zero_grad()
        optimizer_SincNet.zero_grad()
        loss.backward()
        optimizer_MLP.step()
        optimizer_SincNet.step()
        print(' Training Epoch {} : [{}/{}] \t Loss: {:.4f} \t Accuracy: {:.2f} \t '
              .format(epoch, batch_idx, int(len(train_loader.dataset) / args.train_batch_size), loss, acc),
              flush=True, end='\r')
    return mean(losses), mean(accuracy)

def train_snippets(SincNet_model, MLP_model,optimizer_SincNet, optimizer_MLP,device, epoch, train_loader):
    SincNet_model.train()
    MLP_model.train()
    losses = []
    accuracy = []
    total_samples = []
    for batch_idx, (tracks, int_labels, _) in enumerate(train_loader):
        print(batch_idx)
        tracks, int_labels = tracks.to(device), int_labels.to(device)
        #batch_size, windows, samples = list(tracks.size())[0], list(tracks.size())[1], list(tracks.size())[2]
        embeddings = MLP_model(SincNet_model(tracks))
        loss, correct_negative, total = batch_hard_triplet_loss(int_labels, embeddings, margin_positive=8,
                                                               margin_negative=8, device=device, squared=True)
        total_samples.append(total)

        losses.append(loss.item())
        acc = (correct_negative / total) * 100
        accuracy.append(acc)
        optimizer_MLP.zero_grad()
        optimizer_SincNet.zero_grad()
        loss.backward()
        optimizer_MLP.step()
        optimizer_SincNet.step()

        return mean(losses), mean(accuracy)



def test_windowed(SincNet_model, MLP_model, device, epoch, test_loader):
    SincNet_model.eval()
    MLP_model.eval()
    total_samples = []
    losses = []
    accuracy = []
    with torch.no_grad():
        for batch_idx, (tracks, int_labels, _) in enumerate(test_loader):
            #print(batch_idx)
            tracks, int_labels = tracks.to(device), int_labels.to(device)
            batch_size, windows, samples = list(tracks.size())[0], list(tracks.size())[1], list(tracks.size())[2]
            tracks = tracks.view(-1, batch_size*windows, samples).squeeze()
            embeddings = MLP_model(SincNet_model(tracks))
            mean_embeddings = torch.empty(batch_size, list(embeddings.size())[1])
            for i in range(batch_size):
                mean_embeddings[i][:] = torch.mean(embeddings[(i)*windows:(i)*windows+windows][:], dim=0, keepdim=True)
            loss, correct_negative, total = batch_hard_triplet_loss(int_labels, mean_embeddings, margin_positive=8,
                                                                    margin_negative=8, device=device, squared=True)
            total_samples.append(total)
            losses.append(loss.item())
            acc = (correct_negative/total)*100
            accuracy.append(acc)
    print(' Test/Validate Epoch {}: \t Loss: {:.4f}, Accuracy: {:.2f} \t'.format(epoch, mean(losses),
                                                                                         mean(accuracy)))
    return mean(losses), mean(accuracy)


def main():
    global args
    options = read_conf()


    wandb.init(project="sincnet_triplet")
    config = wandb.config

    parser = argparse.ArgumentParser(description="SincNet Speaker Recognition from Raw Waveform")
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
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
                        default='/home/lucvanwyk/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt',
                        type=str,
                        help='path to train samples')
    parser.add_argument('--test-set',
                        default='/home/lucvanwyk/Data/pyannote/Extracted_Speech/trimmed_sample_list_test.txt',
                        type=str,
                        help='path to test samples')
    parser.add_argument('--valid-set',
                        default='/home/lucvanwyk/Data/pyannote/Extracted_Speech/trimmed_sample_list_valid.txt',
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
    wandb.config.update(args)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:0')
    kwargs = {'num_workers': 8, 'pin_memory': True}
    #train_loader = data.DataLoader(Window_Loader(filename=args.train_set, window_length=0.2, overlap=0.01),
    #                               batch_size=args.train_batch_size, shuffle=True)
    train_loader = data.DataLoader(Window_Loader(filename='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt',windowed=True, window_length=0.2, overlap=0.01),
                                   batch_size=16, shuffle=True)
    #test_loader = data.DataLoader(Window_Loader(filename=args.train_set,windowed=True, window_length=0.2, overlap=0.01),
    #                              batch_size=args.test_batch_size, shuffle=True )
    #valid_loader = data.DataLoader(Window_Loader(filename=args.valid_set,windowed=True, window_length=0.2, overlap=0.01),
    #                               batch_size=args.test_batch_size, shuffle=True)


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
    SincNet_model.to(device)

    DNN1_args = {'input_dim': SincNet_model.out_dim,
                 'fc_lay': fc_lay,
                 'fc_drop': fc_drop,
                 'fc_use_batchnorm': fc_use_batchnorm,
                 'fc_use_laynorm': fc_use_laynorm,
                 'fc_use_laynorm_inp': fc_use_laynorm_inp,
                 'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
                 'fc_act': fc_act}

    MLP_net = MLP(DNN1_args)
    MLP_net.to(device)

    print('----')
    print(SincNet_model.out_dim)

    wandb.watch(models=SincNet_model)
    wandb.watch(models=MLP_net)

    if args.load_model:
        try:
            SincNet_model.load_state_dict(torch.load(args.model_path_sincnet))
            MLP_net.load_state_dict(torch.load(args.model_path_mlp))
        except:
            print('Could not load models')

    optimizer_SincNet = optim.RMSprop(params=SincNet_model.parameters(), lr=args.lr, momentum=0.5, alpha=0.8)
    optimizer_MLP = optim.RMSprop(params=MLP_net.parameters(), lr=args.lr, momentum=0.5, alpha=0.8)

    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train_loss, train_acc = train_windowed(SincNet_model=SincNet_model,
                                               MLP_model=MLP_net,
                                               optimizer_SincNet=optimizer_SincNet,
                                               optimizer_MLP=optimizer_MLP,
                                               device=device,epoch=epoch,
                                               train_loader=train_loader)
        print('Finished training epoch {} loss {:.4f} accuracy {:.2f}'.format(epoch, train_loss, train_acc))
        test_loss, test_acc = test_windowed(SincNet_model=SincNet_model,
                                            MLP_model=MLP_net,
                                            epoch=epoch, device=device,
                                            test_loader=test_loader)
        wandb.log({'Train Loss': train_loss, 'Train Accuracy': train_acc, 'Test Loss': test_loss,
                  'Test Accuracy': test_acc})
        print('Finished epoch {} in {:.2f}'.format(epoch, (time.time() - start_time)))
        if args.save_model and (epoch % 20 == 0):
            torch.save(SincNet_model.state_dict(), args.model_path_sincnet)
            torch.save(MLP_net.state_dict(), args.model_path_mlp)
            print('Model saved after {} epochs'.format(epoch))
















if __name__ == '__main__':
    main()