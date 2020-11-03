from Lightning_Nets import *
import torch
import wandb
import argparse
import torch.utils.data as data
import seaborn as sns
import torchaudio
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
sns.set_style("darkgrid")
from Prepare_Track import Prepare_Track
from Triplet_DataLoader import Spectrogram_FrameWise_Loader, Single_Speaker_Loader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt

def get_embeddings(dataloader, device, model, label='convo1'):
    EmbeddingList = []
    model.train()
    #model.eval()
    with torch.no_grad():
        for batch_idx, (spectrograms) in enumerate(dataloader):
            spectrograms = spectrograms.to(device)
            embeddings = model(spectrograms)
            EmbeddingList.append(torch.nn.functional.normalize(embeddings, p=2, dim=1))
            #EmbeddingList.append(embeddings.detach())
        return EmbeddingList

def Compare_Embeddings(df_convo, target_embeddings, convo_embeddings, target_speaker, threshold):
    target_labels = df_convo.loc[target_speaker, :].values
    print(target_labels)
    second_speaker = [x for x in list(df_convo.index.values) if target_speaker not in x][0]
    second_speaker_labels = df_convo.loc[second_speaker, :].values

    outputs_target = np.zeros_like(target_labels)
    outputs_second = np.zeros_like(target_labels)

    #total_embeddings = torch.cat((convo_embeddings, target_embeddings), dim=0)
    #total_embeddings = StandardScaler().fit_transform(total_embeddings.cpu().numpy())
    #pca = PCA(n_components=2)
    #coords = pca.fit_transform(total_embeddings)
    #total_embeddings = torch.from_numpy(total_embeddings)
    #convo_embeddings = total_embeddings[0:len(total_embeddings)-1,:]
    #target_embeddings = total_embeddings[-1,:]



    for i in range(len(target_labels)):
        if target_labels[i] == 1:
            print('correct distance', -torch.norm(target_embeddings-convo_embeddings[i], p=2))
            #print('correct distance', torch.dist(target_embeddings, convo_embeddings[i], p=2))
        else:
            print('WRONG distance', -torch.norm(target_embeddings - convo_embeddings[i], p=2))
            #print('WRONG distance', torch.dist(target_embeddings, convo_embeddings[i], p=2))
        if torch.dist(target_embeddings, convo_embeddings[i]) < 0.3:
            outputs_target[i] = 1
            outputs_second[i] = 0
        else:
            outputs_target[i] = 0
            outputs_second[i] = 1

    acc_target = np.count_nonzero(np.multiply(outputs_target, target_labels) == 1)
    acc_second = np.count_nonzero(np.multiply(outputs_second, second_speaker_labels) == 1)
    print(acc_target, acc_second)
    return acc_target, acc_second

def plot_embedding(speaker1_embeddings, speaker2_embeddings, speaker3_embeddings):
    embeddings = torch.cat((speaker1_embeddings, speaker2_embeddings, speaker3_embeddings), dim=0)
    embeddings = Normalizer().fit_transform(embeddings.cpu().numpy())
    embeddings = StandardScaler().fit_transform(embeddings)
    colors = ['blue' for i in range(len(speaker1_embeddings))] + ['red' for i in range(len(speaker2_embeddings))] + ['green' for i in range(len(speaker3_embeddings))]
    speaker1 = ['speaker1' for i in range(len(speaker1_embeddings))]
    speaker2 = ['speaker2' for i in range(len(speaker2_embeddings))]
    speaker3 = ['speaker3' for i in range(len(speaker3_embeddings))]
    labels = speaker1+speaker2+speaker3
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    cluster_df = pd.DataFrame({'x': [x for x in coords[:, 0]],
                               'y': [y for y in coords[:, 1]],
                               'labels': labels,
                               'color': colors})
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
    plt.title('Test',
              fontsize=24)

    plt.tick_params(labelsize=20)
    plt.xlabel('pca-one', fontsize=20)
    plt.ylabel('pca-two', fontsize=20)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="VGGVox CNN with Spectrograms for Speaker Verification")
    parser.add_argument('--model-path', type=str,
                        default='/home/lucas/PycharmProjects/MetricEmbeddingNet/models/VGG_Spectrogram_Triplet_margin:0.2_full',
                        help='path to model')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='number of frames to load at each iteration')
   # parser.add_argument()
    args = parser.parse_args()

    device = torch.device('cuda:0')
    model = VGGVox()
    model.to(device)
    model.load_state_dict(torch.load(args.model_path))

    kwargs = {'num_workers': 12, 'pin_memory': True}



    Convo1 = Prepare_Track(path_to_track='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MTD020ME_FEE083_20_3_5.wav', path_to_rttm='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MTD020ME_FEE083_20_3_5.rttm')
    df_convo1_frame_labels1, convo1_frame_list, speaker_dict_conv1 = Convo1.label_frames(window_size=3, step_size=0.1)

    #Convo2 = Prepare_Track(
    #    path_to_track='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE005_20_3_5.wav',
    #    path_to_rttm='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE005_20_3_5.rttm')
    #df_convo2_frame_labels2, convo2_frame_list, speaker_dict_conv2 = Convo2.label_frames(window_size=3, step_size=0.1)


    #Convo1_Loader = data.DataLoader(Spectrogram_FrameWise_Loader(filename='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MTD020ME_FEE083_20_3_5.wav', frame_list=convo1_frame_list), batch_size=args.batch_size, shuffle=False, **kwargs)
    #Convo2_Loader = data.DataLoader(Spectrogram_FrameWise_Loader(filename='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE005_20_3_5.wav', frame_list=convo2_frame_list), batch_size=args.batch_size, shuffle=False, **kwargs)
    Target_Speaker_Loader = data.DataLoader(Single_Speaker_Loader(path='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt',speaker='FIE081',mel=False), batch_size=args.batch_size, shuffle=False, **kwargs)
    Target_Speaker_Loader_2 = data.DataLoader(Single_Speaker_Loader(path='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt',speaker='FEE083',mel=False), batch_size=args.batch_size, shuffle=False, **kwargs)
    Target_Speaker_Loader_3 = data.DataLoader(Single_Speaker_Loader(path='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list_train.txt',speaker='MTD020ME',mel=False), batch_size=args.batch_size, shuffle=False, **kwargs)


    EmbeddingList1 = get_embeddings(Target_Speaker_Loader_2, device, model)
    EmbeddingList1 = torch.cat(EmbeddingList1, dim=0)[0:30]

    #EmbeddingList2 = get_embeddings(Convo2_Loader, device, model)
    #EmbeddingList2 = torch.cat(EmbeddingList2, dim=0)

    TargetEmbeddingList = get_embeddings(Target_Speaker_Loader, device, model)[0:30]
    TargetEmbeddingList = torch.cat(TargetEmbeddingList, dim=0)

    Spk3 = get_embeddings(Target_Speaker_Loader_3, device, model)
    Spk3 = torch.cat(Spk3, dim=0)

    diag = torch.diag(torch.matmul(TargetEmbeddingList[0:9], TargetEmbeddingList[0:9].t()))
    distances = diag.unsqueeze(0) - 2.0*diag + diag.unsqueeze(1)
    print(distances)
    print('--------')

    Embeddings = torch.cat((TargetEmbeddingList[0:9], EmbeddingList1[0:9]), dim=0)

    diag = torch.diag(torch.matmul(Embeddings, Embeddings.t()))
    distances = diag.unsqueeze(0) - 2.0*diag + diag.unsqueeze(1)
    print(distances)

    #Embeddings = torch.cat((TargetEmbeddingList, EmbeddingList1), dim=0)
    #dot_product = torch.matmul(Embeddings, Embeddings.t())
    #square_norm = torch.diag(dot_product)
    #distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    #print(distances)
    #plot_embedding(EmbeddingList1, TargetEmbeddingList, Spk3)



if __name__ == '__main__':
    main()