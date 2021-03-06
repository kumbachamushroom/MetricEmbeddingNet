
from Triplet_Net import *
from Triplet_DataLoader import Triplet_Tensor_Loader, Triplet_Time_Loader, Window_Loader
from SincNet_dataio import ReadList, read_conf,str_to_bool
import wandb
import torch.utils.data as data
import torch.optim as optim
import time
from Metric_Losses import batch_hard_triplet_loss
import torch.backends.cudnn as cudnn

class AverageMeter:
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.valu = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val *n
        self.count += n
        self.avg = self.sum / self.count

def train_windowed(train_loader, SincNet_model, MLP_model, optimizer_SincNet, optimizer_MLP, epoch, device):
    MLP_model.train()
    SincNet_model.train()
    losses = AverageMeter()
    accuracy = AverageMeter()
    total_samples = 0
    for batch_idx, (tracks, int_labels) in enumerate(train_loader):
        tracks, int_labels = tracks.cuda(), int_labels.cuda()
        embeddings = torch.zeros(list(tracks.size())[0], 258, device=device)
        for i in range(list(tracks.size())[0]):
            track = tracks[i,:,:] #15, 3600
            embedding =  MLP_model(SincNet_model(track))
            embedding = torch.mean(embedding, dim=0, keepdim=True)
            embeddings[i, :] = embedding
        loss, correct_negative, total = batch_hard_triplet_loss(int_labels, embeddings, margin_positive=2,
                                                                margin_negative=2, device='cuda',
                                                                squared=True)
        #total_samples = total + total_samples
        #print(correct_negative, total, acc)
        acc = (correct_negative / total) * 100
        #print(correct_negative, total, acc)
        accuracy.update(acc, 1)
        optimizer_SincNet.zero_grad()
        optimizer_MLP.zero_grad()
        loss.backward()
        optimizer_SincNet.step()
        optimizer_MLP.step()
        losses.update(loss.detach(), 1)
        print(' Train epoch: {} [{}/{}]\t Loss {:.4f} Acc {:.4f} \t '.format(epoch, batch_idx,
                                                                             int(len(train_loader.dataset) / 64), loss,
                                                                             acc), flush=True, end='\r')
    return losses.avg, accuracy.avg

def test_windowed(test_loader, SincNet_model, MLP_model, epoch, device):
    MLP_model.eval()
    SincNet_model.eval()
    total_samples=0
    losses = AverageMeter()
    accuracy = AverageMeter()
    with torch.no_grad():
        for batch_idx, (tracks, int_labels) in enumerate(test_loader):
            tracks, int_labels = tracks.cuda(), int_labels.cuda()
            embeddings = torch.zeros(list(tracks.size())[0], 258, device=device)
            for i in range(list(tracks.size())[0]):
                track = tracks[i,:,:]
                embedding = MLP_model(SincNet_model(track))
                embedding = torch.mean(embedding, dim=0, keepdim=True)
                embeddings[i, :] = embedding
            loss, correct_negative, total = batch_hard_triplet_loss(int_labels, embeddings, margin_positive=2,
                                                            margin_negative=2, device='cuda',
                                                            squared=True)

            total_samples = total_samples + total
            losses.update(loss.detach(), 1)
            accuracy.update((correct_negative / total) * 100, 1)
    print('Test Epoch {}: Loss: {:.4f}, Accuracy {:.2f} \t'.format(epoch, losses.avg, accuracy.avg))
    return losses.avg, accuracy.avg




def train(train_loader, SincNet_model, MLP_model, optimizer_SincNet, optimizer_MLP, epoch):
    MLP_model.train()
    SincNet_model.train()
    losses = AverageMeter()
    accuracy = AverageMeter()
    total_samples = 0
    for batch_idx, (tracks, int_labels, string_labels) in enumerate(train_loader):
        tracks, int_labels = tracks.cuda(), int_labels.cuda()
        embeddings = SincNet_model(tracks)
        embeddings = MLP_model(embeddings)
        loss, correct_negative, total = batch_hard_triplet_loss(int_labels, embeddings, margin_positive=2,
                                                                margin_negative=2, device='cuda',
                                                                squared=True)
        total_samples = total_samples + total
        acc = (correct_negative/total)*100
        accuracy.update(acc, 1)
        optimizer_SincNet.zero_grad()
        optimizer_MLP.zero_grad()
        #print(loss)
        loss.backward()
        optimizer_SincNet.step()
        optimizer_MLP.step()
        losses.update(loss, 1)
        print(' Train epoch: {} [{}/{}]\t  Loss {:.4f} Acc {:.2f} \t '.format(epoch, batch_idx, int(len(train_loader.dataset)/64), loss, acc), flush=True, end='\r')
        #wandb.log(
        #    {"Train Accuracy": acc, "Train Loss": loss})#, "Test Accuracy": test_accuracy_avg,
        #    # "Test Loss": test_losses_avg})

    return losses.avg, accuracy.avg

def test(test_loader, SincNet_model, MLP_model,epoch):
    MLP_model.eval()
    SincNet_model.eval()
    total_samples = 0
    losses = AverageMeter()
    accuracy = AverageMeter()
    with torch.no_grad():
        for batch_idx, (tracks, int_labels, string_labels) in enumerate(test_loader):
            tracks, int_labels = tracks.cuda(), int_labels.cuda()
            embeddings = SincNet_model(tracks)
            embeddings = MLP_model(embeddings)
            #print(embeddings)
            loss, correct_negative, total = batch_hard_triplet_loss(int_labels, embeddings, margin_negative=2, margin_positive=2,
                                                                    device='cuda', squared=True)
            total_samples = total_samples + total
            losses.update(loss, 1)
            accuracy.update((correct_negative/total)*100, 1)
    print('Test Epoch {}: Loss: {:.4f}, Accuracy {:.2f} \t'.format(epoch, losses.avg, accuracy.avg))
    return losses.avg, accuracy.avg


def testing_loader(loader):
    i = 1
    start = time.time()
    for tracks, int_labels, string_labels in loader:
        tracks = tracks.cuda()
        int_labels = int_labels.cuda()
        print(time.time()-start)
        start=time.time()



def main():
    #READ CONFIG FILE
    options = read_conf()

    #LOG ON WANDB?
    log = options.wandb
    project_name = options.project

    if log:
        wandb.init(project='SincNet_Triplet')
        wandb.run.name = project_name

    device = torch.device("cuda:0")

    kwargs = {'num_workers' : 4, 'pin_memory':True}

    #Get data path
    data_PATH = options.path
    sincnet_path = options.sincnet_path
    mlp_path = options.mlp_path
    load = options.load

    #train_loader = data.DataLoader(Triplet_Time_Loader(path=data_PATH, spectrogram=False, train=True), batch_size=64, shuffle=True, **kwargs)
    #test_loader = data.DataLoader(Triplet_Time_Loader(path=data_PATH, spectrogram=False, train=False), batch_size=64, shuffle=True, **kwargs)


    #get parameters for SincNet and MLP
    #[cnn]
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

    # [optimization]
    lr = float(options.lr)
    batch_size = int(options.batch_size)
    N_epochs = int(options.N_epochs)
    N_batches = int(options.N_batches)
    N_eval_epoch = int(options.N_eval_epoch)
    seed = int(options.seed)
    torch.manual_seed(120)

    train_loader = data.DataLoader(Window_Loader(path=data_PATH, spectrogram=False, train=True), batch_size=batch_size,
                                   shuffle=True, **kwargs)
    test_loader = data.DataLoader(Window_Loader(path=data_PATH, spectrogram=False, train=False), batch_size=batch_size,
                                  shuffle=True, **kwargs)

    SincNet_args = {'input_dim': 3200, #3 seconds at 16000Hz
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

    if load:
        try:
            SincNet_model.load_state_dict(torch.load(sincnet_path))
            MLP_net.load_state_dict(torch.load(mlp_path))
        except:
            print('Could not load models')

    if log:
        wandb.watch(models=SincNet_model)
        wandb.watch(models=MLP_net)

    #optimizer_SincNet = optim.RMSprop(params=SincNet_model.parameters(), lr=lr,
    #                                  alpha=0.8, momentum=0.5)
    #optimizer_MLP = optim.RMSprop(params=MLP_net.parameters(), lr=lr, alpha=0.8, momentum=0.5)
    optimizer_SincNet = optim.Adam(params=SincNet_model.parameters(), lr=lr)
    optimizer_MLP = optim.Adam(params=MLP_net.parameters(), lr=lr)

    #cudnn.benchmark = True



    for epoch in range(1, N_epochs+1):
        start_time = time.time()
        train_losses_avg, train_accuracy_avg = train_windowed(epoch=epoch, train_loader=train_loader, SincNet_model=SincNet_model, MLP_model=MLP_net,
                                                     optimizer_SincNet=optimizer_SincNet, optimizer_MLP=optimizer_MLP, device=device)
        duration = time.time() - start_time
        print("Done training epoch {} in {:.4f} \t Accuracy {:.2f} Loss {:.4f}".format(epoch, duration, train_accuracy_avg, train_losses_avg))
        test_losses_avg, test_accuracy_avg = test_windowed(test_loader=test_loader, SincNet_model=SincNet_model, MLP_model=MLP_net, epoch=epoch, device=device)
        if log:
            wandb.log({"Train Accuracy":train_accuracy_avg, "Train Loss":train_losses_avg, "Test Accuracy":test_accuracy_avg, "Test Loss":test_losses_avg})
        if (epoch % 5) == 0:
            torch.save(SincNet_model.state_dict(), sincnet_path)
            torch.save(MLP_net.state_dict(), mlp_path)
            print("Model saved after {} epochs".format(epoch))







if __name__ == '__main__':
    main()
