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
from Triplet_DataLoader import Spectrogram_FrameWise_Loader

def get_embeddings(dataloader, device, model, label='convo1'):
    EmbeddingList = []
    model.eval()
    for batch_idx, (spectrograms) in enumerate(dataloader):
        spectrograms = spectrograms.to(device)
        embeddings = model(spectrograms)
        EmbeddingList.append(embeddings.detach())
    return EmbeddingList

def cluster_embeddings(Convo1_Embeddings, Convo2_Embeddings, df_labels1, df_labels2, common_speaker):
    Agg_EmbeddingList = torch.cat([Convo1_Embeddings, Convo2_Embeddings], dim=0).cpu().numpy()

    kmeans = KMeans(n_clusters=2,n_init=20, algorithm="elkan")
    ac = AgglomerativeClustering(n_clusters=2)

    pca = PCA(n_components=2)
    Convo2_Embeddings = StandardScaler().fit_transform(Convo2_Embeddings.cpu().numpy())
    principalComponents = pca.fit_transform(Convo2_Embeddings)

    #kmeans.fit(principalComponents)_pre
    #ac.fit(principalComponents)
    ac = ac.fit_predict(principalComponents)
    #y_kmeans = kmeans.predict(principalComponents)

    #cluster_labels_1 = y_kmeans#[0:df_labels1.shape[1]]ta
    cluster_labels_2 = ac#y_kmeans#[df_labels1.shape[1]+1:]
    print(cluster_labels_2)

    #common_speaker_1 = df_labels1.loc[common_speaker,:].values.tolist()
    common_speaker_2 = df_labels2.loc[common_speaker,:].values.tolist()


    speaker_labels = list(df_labels1.index.values) + list(df_labels2.index.values)
    #second_speaker_1_label = [x for x in list(df_labels1.index.values) if common_speaker not in x][0]
    second_speaker_2_label = [x for x in list(df_labels2.index.values) if common_speaker not in x][0]


    #second_speaker_1 = df_labels1.loc[second_speaker_1_label,:].values.tolist()
    second_speaker_2 = df_labels2.loc[second_speaker_2_label,:].values.tolist()


    #df_conv_1 = pd.DataFrame(list(zip(cluster_labels_1, common_speaker_1, second_speaker_1)), columns=['cluster', common_speaker, second_speaker_1_label])
    df_conv_1 = pd.DataFrame()
    df_conv_2 = pd.DataFrame(list(zip(cluster_labels_2, common_speaker_2, second_speaker_2)), columns=['cluster', common_speaker, second_speaker_2_label])


    return df_conv_1, df_conv_2

def main():
    parser = argparse.ArgumentParser(description="VGGVox CNN with Spectrograms for Speaker Verification")
    parser.add_argument('--model-path', type=str,
                        default='/home/lucas/PycharmProjects/MetricEmbeddingNet/models/VGG_Spectrogram_Triplet',
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



    Convo1 = Prepare_Track(path_to_track='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE087_20_3_5.wav', path_to_rttm='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE087_20_3_5.rttm')
    df_convo1_frame_labels1, convo1_frame_list, speaker_dict_conv1 = Convo1.label_frames(window_size=3, step_size=0.1)

    Convo2 = Prepare_Track(
        path_to_track='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE005_20_3_5.wav',
        path_to_rttm='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE005_20_3_5.rttm')
    df_convo2_frame_labels2, convo2_frame_list, speaker_dict_conv2 = Convo2.label_frames(window_size=3, step_size=0.1)


    Convo1_Loader = data.DataLoader(Spectrogram_FrameWise_Loader(filename='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE087_20_3_5.wav', frame_list=convo1_frame_list), batch_size=args.batch_size, shuffle=False, **kwargs)
    Convo2_Loader = data.DataLoader(Spectrogram_FrameWise_Loader(filename='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations/MEO069_FEE005_20_3_5.wav', frame_list=convo2_frame_list), batch_size=args.batch_size, shuffle=False, **kwargs)

    EmbeddingList1 = get_embeddings(Convo1_Loader, device, model)
    EmbeddingList1 = torch.cat(EmbeddingList1, dim=0)


    EmbeddingList2 = get_embeddings(Convo2_Loader, device, model)
    EmbeddingList2 = torch.cat(EmbeddingList2, dim=0)




    df_1, df_2 = cluster_embeddings(EmbeddingList1, EmbeddingList2, df_convo1_frame_labels1, df_convo2_frame_labels2, 'MEO069')
    print(df_2)


if __name__ == '__main__':
    main()

