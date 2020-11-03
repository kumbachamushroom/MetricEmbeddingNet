import torch
import torchaudio
from pyannote.core import Segment
import numpy as np
import torch.utils.data as data
import seaborn as sns
import pandas as pd
import numpy as np
from New_Prepare_Track import Prepare_Track_Multi_Label
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
sns.set_style("darkgrid")
from math import floor
import hydra
from omegaconf import DictConfig
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from pyannote.database.util import load_rttm
import glob
from pyannote.core import Annotation, Timeline
from random import randint
from pyannote.metrics.diarization import DiarizationErrorRate
import csv

def get_embeddings(model, starts, stops, duration, cfg):
    excerpt = Segment(start=starts, end=stops)
    embedding = model.crop({'audio':cfg.audio.verification_path, 'duration':duration}, segment=excerpt)
    #print(np.mean(embedding, axis=0, keepdim=True).shape())
    return np.mean(embedding, axis=0, keepdims=True)

def get_track_embeddings(model, frame_list, path, duration):
    embeddings = []
    for i in range(len(frame_list)):
        start, stop = round(frame_list[i][0],1), round(frame_list[i][1],1)
        try:
            excerpt = Segment(start=start, end=stop)
            embedding = np.mean(model.crop({'audio': path, 'duration': duration}, segment=excerpt), axis=0, keepdims=True)
            embeddings.append(embedding)
        except:
            embeddings.append(np.zeros(shape=(1,512), dtype=float))
            print('could not embed  ',start, stop)
    return np.concatenate(embeddings, axis=0)

def speaker_verification(track_embedding, df_labels, df_embeddings_verification, threshold):
    speaker_list = df_labels.columns.tolist()
    df_output = pd.DataFrame()
    for speaker in speaker_list:
        speaker_emb = np.array(df_embeddings_verification[speaker].values).reshape(1,512)
        output_frames = np.zeros_like(df_labels[speaker].values)
        for i in range(len(df_labels[speaker].values)):
            distance = cdist(track_embedding[i].reshape(1, -1),speaker_emb,metric='cosine')[0][0]
            if distance < threshold:
                output_frames[i] = 1
        df_output[speaker] = output_frames
    return df_output

def multi_speaker_verification(track_embedding, df_labels, df_embeddings_verification, threshold):
    speaker_list = df_labels.columns.tolist()
    df_output = pd.DataFrame()
    speaker_emb = {}
    for speaker in speaker_list:
        output_frames = np.zeros_like(df_labels[speaker].values)
        df_output[speaker] = output_frames
        speaker_emb[speaker] = np.array(df_embeddings_verification[speaker].values).reshape(1,512)
    for i in range(len(output_frames)):
        distances = []
        for speaker in speaker_list:
            distances.append(cdist(track_embedding[i].reshape(1, -1),speaker_emb[speaker],metric='cosine')[0][0])
        if min(distances) <= threshold:
            print('MINIMUM IS')
            print(distances.index(min(distances)))
            print('---------------')
            # Upload
            df_output.iloc[i, speaker[distances.index(min(distances))]] = 1
    return df_output

def FAR_FRR(y_true, y_pred):
    false_acceptance = 0
    false_rejection = 0
    for i in range(len(y_true)):
        if (y_true[i] == 1) and (y_pred[i] == 0):
            false_rejection = false_rejection + 1
        elif (y_true[i] == 0) and (y_pred[i] == 1):
            false_acceptance = false_acceptance + 1
    FAR = false_acceptance/len(y_true[y_true == 0])
    FRR = false_rejection/len(y_true[y_true == 1])
    return FAR, FRR


def plot_PR_ROC_per_spk(x, y, title, x_label, y_label):
    speaker_list = y.columns.tolist()
    plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for speaker in speaker_list:
        try:
            plt.plot(x[speaker], y[speaker])
        except:
            plt.plot(x, y[speaker])
    plt.legend(speaker_list)
    plt.savefig(title+'.png')

def plot_FRR_FAR_per_spk(FAR, FRR, threshold, title):
    speaker_list = FAR.columns.tolist()
    colors = []

    for i in range(len(speaker_list)):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    plt.figure()
    plt.xlabel('Threshold')
    plt.ylabel('FAR (Left), FRR(Right) (%)')
    for i, speaker in enumerate(speaker_list):
        plt.plot(threshold, FAR[speaker], color=colors[i])
        plt.plot(threshold, FRR[speaker], color=colors[i])
    plt.legend(speaker_list)
    plt.savefig(title+'.png')


def fDER(df_labels, df_outputs):
    speaker_list = df_labels.columns.tolist()
    num_frames = len(df_labels.iloc[0, :])
    E_FA = 0
    E_MISS = 0
    E_Spk = 0
    for i in range(num_frames):
        true_frame = df_labels.iloc[i, :].to_numpy()
        output_frame = df_outputs.iloc[i,:].to_numpy()
        if 1 not in true_frame:
            if 1 in output_frame:
                E_FA = E_FA + 1
        else:
            if 1 not in output_frame:
                E_MISS  = E_MISS + 1
            elif true_frame != output_frame:
                E_Spk = E_Spk + 1
    print(E_FA/num_frames)
    print(E_MISS/num_frames)
    print(E_Spk/num_frames)
    DER = (E_FA + E_MISS + E_Spk)/num_frames
    return DER

def save_plot(x, y, x_label, y_label, title):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(title+'.png')


def performance_metrics(df_labels,df_embeddings_verification, track_embedding, cfg, frame_list, iteration):
    speaker_list = df_labels.columns.tolist()
    df_precision = pd.DataFrame(columns=speaker_list, index=cfg.audio.threshold)
    df_roc = pd.DataFrame(columns=speaker_list, index=cfg.audio.threshold)
    df_recall = pd.DataFrame(columns=speaker_list, index=cfg.audio.threshold)
    df_far = pd.DataFrame(columns=speaker_list, index=cfg.audio.threshold)
    df_frr = pd.DataFrame(columns=speaker_list, index=cfg.audio.threshold)
    der = []
    metric = DiarizationErrorRate(skip_overlap=True, collar=cfg.audio.collar)
    groundtruth = load_rttm(cfg.audio.rttm_path)[cfg.audio.uri[iteration]]
    for threshold in cfg.audio.threshold:
        df_output = multi_speaker_verification(track_embedding=track_embedding, df_labels=df_labels, df_embeddings_verification=df_embeddings_verification, threshold=threshold)
        for speaker in speaker_list:
            try:
                df_precision.loc[threshold, speaker] = precision_score(df_labels[speaker], df_output[speaker], average='binary')
            except:
                df_precision.loc[threshold, speaker] = 0
            try:
                df_recall.loc[threshold, speaker] = recall_score(df_labels[speaker], df_output[speaker], average='binary')
            except:
                df_recall.loc[threshold, speaker] = 0
            try:
                df_roc.loc[threshold, speaker] = roc_auc_score(df_labels[speaker], df_output[speaker], average=None)
            except:
                df_roc.loc[threshold, speaker] = 0
            try:
                far, frr = FAR_FRR(y_true=df_labels[speaker], y_pred=df_output[speaker])
                df_far.loc[threshold, speaker] = far
                df_frr.loc[threshold, speaker] = frr
            except:
                df_far.loc[threshold, speaker] = 0
                df_frr.loc[threshold, speaker] = 0

        components = metric(groundtruth, merge_frames(df_outputs=df_output, frame_list=frame_list,
                                                      filename=cfg.audio.uri[iteration] + '_' + str(threshold)),
                            detailed=True)
        components = metric[:]
        der.append(components)
    return df_precision, df_recall, df_roc, df_far, df_frr, der

def DER(df_labels, df_outputs, frame_list, cfg, collar):
    speaker_list = df_labels.columns.tolist()
    rttm_segment = load_rttm(cfg.audio.rttm_path)[cfg.audio.uri[0]]
    E_MISS = 0
    E_FA = 0
    E_Spk = 0
    reference_length = 0
    length = (len(frame_list))
    for i, frame in enumerate(frame_list):
        frame_start, frame_end = float(frame[0]), float(frame[1])
        segments = []

        for segment in rttm_segment.get_timeline():

            if list(rttm_segment.get_labels(segment))[0] in speaker_list:
                intersection = max(0, min(float(frame[1]), segment.end) - max(float(frame[0]), segment.start))
                if intersection > collar:
                    segments.append(segment)
                    #print('start', segment.start)
                    #print('end', segment.end)
                    reference_length = reference_length + intersection
        if len(segments) == 0:
            if 1 in df_outputs.iloc[i,:].to_numpy():
                E_FA = E_FA + (float(frame[1]) - float(frame[0]))
        if len(segments) > 0:
            if 1 not in (df_outputs.iloc[i, :].to_numpy()):
                E_MISS = E_MISS + (float(frame[1]) - float(frame[0]))
            else:
                active_speakers = []
                for interval in segments:
                    intersection = max(0, min(float(frame[1]), interval.end) - max(float(frame[0]), interval.start))
                    active_speakers.append(list(rttm_segment.get_labels(interval))[0])
                for active_spk in active_speakers:
                    if (df_outputs.loc[i, active_spk] == 0):
                        E_Spk = E_Spk + (segments[active_speakers.index(active_spk)].end - segments[active_speakers.index(active_spk)].start)
                inactive_speakers = list(set(speaker_list)-set(active_speakers))
                for spk in inactive_speakers:
                    if (df_outputs.loc[i, spk] == 1):
                        E_Spk = E_Spk + (float(frame[1])-float(frame[0]))

    print(reference_length)
    print(E_MISS)
    print(E_FA)
    print(E_Spk)
    return (E_MISS + E_Spk + E_FA)/reference_length
#WIDI!!
def merge_frames(df_outputs, frame_list, filename):
    speaker_list = df_outputs.columns.tolist()
    annotation = Annotation()
    for speaker in speaker_list:
        seg_start = 0
        seg_end = 0
        for i, label in enumerate(df_outputs[speaker]):
            if (label == 1) and (seg_start == 0):
                seg_start = float(frame_list[i][0])
            elif (label == 0) and (seg_start > 0):
                seg_end = float(frame_list[i][1])
                annotation[Segment(start=seg_start, end=seg_end)] = speaker
                seg_start = 0
            else:
                seg_end = float(frame_list[i][1])

    #with open('/home/lucas/PycharmProjects/MetricEmbeddingNet/rttm_out/'+filename+'.rttm', 'w') as f:
    #    annotation.write_rttm(f)
    return annotation

@hydra.main(config_path="Pyannote_Emb_Config_2.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    model = torch.hub.load(cfg.pretrained_model.path, cfg.pretrained_model.type)
    excerpt = Segment(start=60.0, end=120.0)
    speakers = []
    for k, path in enumerate(cfg.audio.verification_path):
        print(k)
        print(path)
        list = glob.glob(cfg.audio.target_path[k]+'/*', recursive=True)
        print(cfg.audio.target_path[k])
        for i in range(len(list)):
            list[i] = list[i][list[i].rfind('/')+1:list[i].rfind('.')]
        speakers.append(tuple(list))
    print(speakers)
    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            df_DER = pd.DataFrame()
            for i, track in enumerate(cfg.audio.verification_path):
                df_frames = pd.DataFrame()
                df_embedding_track = pd.DataFrame()
                df_embeddings_verification = pd.DataFrame()
                frame_list, df_frames= Prepare_Track_Multi_Label(Audio_path=track, RTTM_path=cfg.audio.rttm_path,
                                                                    window_size=window_length,
                                                                    step_size=float(window_length * step_length))
                for j in range(len(speakers[i])):
                    label = speakers[i][j]
                    target_embedding = np.mean(model.crop({'audio':cfg.audio.target_path[i]+'/'+label+'.wav','duration':1000},segment=excerpt), axis=0, keepdims=True)[0]
                    df_embeddings_verification[label] = target_embedding
                duration = int(frame_list[-1][1])
                track_embedding = get_track_embeddings(model=model, frame_list=frame_list, path=track, duration=duration)
                df_precision, df_recall, df_roc, df_far, df_frr, der = performance_metrics(df_labels=df_frames, df_embeddings_verification=df_embeddings_verification, cfg=cfg, track_embedding=track_embedding, frame_list=frame_list, iteration=i)


                #Plot and save precision-recall data
                plot_PR_ROC_per_spk(x=df_recall, y=df_precision, title='Precision-Recall Curve (Per-Speaker) ('+cfg.audio.uri[i]+')'+'WL:'+str(window_length)+'SL'+str(step_length), x_label='Recall', y_label='Precision')
                # Plot and save FRR-FAR data
                plot_PR_ROC_per_spk(x=cfg.audio.threshold, y=df_roc, title='AUC-ROC (per-speaker) ('+cfg.audio.uri[i]+')WL:'+str(window_length)+'SL'+str(step_length), x_label='threshold', y_label='AUC')
                plot_FRR_FAR_per_spk(FAR=df_far, FRR=df_frr,threshold=cfg.audio.threshold, title='FRR-FAR (per-speaker) ('+cfg.audio.uri[i]+')WL:' +str(window_length)+'SL'+str(step_length))
                with open(cfg.dataframes.save_path+'/PR_AUC_Single.csv', mode='a') as csv_file:
                    fieldnames = ['Window Length', 'Overlap', 'Track', 'Threshold', 'Speaker', 'Precision', 'Recall', 'AUC', 'FAR','FRR']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for t, threshold in enumerate(cfg.audio.threshold):
                        speaker_list = df_frames.columns.tolist()
                        for speaker in speaker_list:
                            writer.writerow({'Window Length': window_length,
                                             'Overlap': step_length,
                                             'Track': cfg.audio.uri[i],
                                             'Threshold':threshold,
                                             'Speaker': speaker,
                                             'Precision': df_precision.loc[threshold, speaker],
                                             'Recall': df_recall.loc[threshold, speaker],
                                             'AUC': df_roc.loc[threshold, speaker],
                                             'FAR': df_far.loc[threshold, speaker],
                                             'FRR':df_frr.loc[threshold, speaker]})

                der_error_rate = []
                der_fa = []
                der_conf = []
                der_md = []
                der_tot = []
                for metrics in der:
                    print(metrics)
                    error_rate = (metrics['false alarm'] + metrics['missed detection'] + metrics['confusion']) / \
                                 metrics['total']
                    if error_rate < 1.0:
                        der_error_rate.append(error_rate)
                    else:
                        der_error_rate.append(1.0)
                    der_fa.append(metrics['false alarm'])
                    der_conf.append(metrics['confusion'])
                    der_md.append(metrics['missed detection'])
                    der_tot.append(metrics['total'])
                plt.figure()
                plt.plot(cfg.audio.threshold, der_error_rate)
                plt.title(
                    'Diarization Error Rate vs. Threshold (' + cfg.audio.uri[i] + ')' + str(window_length) + 'SL' + str(
                        step_length))
                plt.xlabel('Threshold')
                plt.ylabel('DER')
                plt.savefig('DER_' + cfg.audio.uri[i] + '_WL:' + str(window_length) + 'SL' + str(step_length) + '.png')

                with open(cfg.dataframes.save_path + '/DER_Single.csv', mode='a') as csv_file:
                    fieldnames = ['Window Length', 'Overlap', 'Track', 'Threshold', 'DER', 'False Alarm',
                                  'Missed Detection', 'Confusion', 'Total']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for t, threshold in enumerate(cfg.audio.threshold):
                        writer.writerow({'Window Length': window_length,
                                         'Overlap': step_length,
                                         'Track': cfg.audio.uri[i],
                                         'Threshold': threshold,
                                         'DER': der_error_rate[t],
                                         'False Alarm': der_fa[t],
                                         'Missed Detection': der_md[t],
                                         'Confusion': der_conf[t],
                                         'Total': der_tot[t]})

if __name__ == '__main__':
    main()

