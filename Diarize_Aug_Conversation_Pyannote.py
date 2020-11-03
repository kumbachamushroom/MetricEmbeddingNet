import torch
import torchaudio
from pyannote.core import Segment
import numpy as np
import torch.utils.data as data
import seaborn as sns
import pandas as pd
import numpy as np
from New_Prepare_Track import Prepare_Track
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
from pyannote.core import Annotation
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
        df_output = speaker_verification(track_embedding=track_embedding, df_labels=df_labels, df_embeddings_verification=df_embeddings_verification, threshold=threshold)
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

        #der.append(metric(groundtruth, merge_frames(df_outputs=df_output, frame_list=frame_list, filename='try1_'+str(threshold))))
        components = metric(groundtruth, merge_frames(df_outputs=df_output, frame_list=frame_list, filename=cfg.audio.uri[iteration]+'_'+str(threshold)), detailed=True)
        components = metric[:]
       # print('False alarm: {}, Missed_Detection: {}, Confusion{}, Total {}'.format(DER['false alarm'], DER['missed detection'], DER['confusion'], DER['total']))
        #if DER <= 1:
        der.append(components)
        #else:
        #    der.append(1.0)
    return df_precision, df_recall, df_roc, df_far, df_frr, der


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

    with open('/home/lucvanwyk/MetricEmbeddingNet/rttm_out/'+filename+'.rttm', 'w') as f:
        annotation.write_rttm(f)
    return annotation







@hydra.main(config_path="Pyannote_Emb_Config.yaml")
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    model = torch.hub.load(cfg.pretrained_model.path, cfg.pretrained_model.type)
    excerpt = Segment(start=60.0, end=120.0)
    speakers = []
    for k, path in enumerate(cfg.audio.verification_path):
        print(k)
        print(path)
        list = glob.glob(cfg.audio.target_path[k]+'/*', recursive=True)
        for i in range(len(list)):
            list[i] = list[i][list[i].rfind('/')+1:list[i].rfind('.')]
        speakers.append(tuple(list))
    print(speakers)

    for window_length in cfg.audio.window_length:
        for step_length in cfg.audio.step_length:
            for i, track in enumerate(cfg.audio.verification_path):
                df_DER = pd.DataFrame(columns=['DER', 'false alarm', 'missed detection', 'confusion', 'total'],
                                      index=cfg.audio.threshold)
                df_frames = pd.DataFrame()
                df_embedding_track = pd.DataFrame()
                df_embeddings_verification = pd.DataFrame()
                for j in range(len(speakers[i])):
                    label = speakers[i][j]
                    label_array, frame_list = Prepare_Track(Audio_path=track, RTTM_path=cfg.audio.rttm_path, window_size=window_length, step_size=float(window_length*step_length), Speaker=label)
                    df_frames[label] = label_array
                    target_embedding = np.mean(model.crop({'audio':cfg.audio.target_path[i]+'/'+label+'.wav','duration':1000},segment=excerpt), axis=0, keepdims=True)[0]
                    df_embeddings_verification[label] = target_embedding
                    duration = int(frame_list[-1][1])
                track_embedding = get_track_embeddings(model=model, frame_list=frame_list, path=track, duration=duration)
                df_precision, df_recall, df_roc, df_far, df_frr, der = performance_metrics(df_labels=df_frames, df_embeddings_verification=df_embeddings_verification, cfg=cfg, track_embedding=track_embedding, frame_list=frame_list, iteration=i)


                #Plot and save precision-recall data
                plot_PR_ROC_per_spk(x=df_recall, y=df_precision, title='Precision-Recall Curve (Per-Speaker) ('+cfg.audio.uri[i]+')'+'WL:'+str(window_length)+'SL'+str(step_length), x_label='Recall', y_label='Precision')
                #df_recall.to_csv(cfg.dataframes.save_path+'/Recall_{}_WL:{}_SL:{}.csv'.format(cfg.audio.uri[i], window_length, step_length), index=True)
                #df_precision.to_csv(cfg.dataframes.save_path+'/Precision_{}_WL:{}_SL:{}.csv'.format(cfg.audio.uri[i], window_length, step_length), index=True)

                # Plot and save FRR-FAR data
                plot_PR_ROC_per_spk(x=cfg.audio.threshold, y=df_roc, title='AUC-ROC (per-speaker) ('+cfg.audio.uri[i]+')WL:'+str(window_length)+'SL'+str(step_length), x_label='threshold', y_label='AUC')
                plot_FRR_FAR_per_spk(FAR=df_far, FRR=df_frr,threshold=cfg.audio.threshold, title='FRR-FAR (per-speaker) ('+cfg.audio.uri[i]+')WL:' +str(window_length)+'SL'+str(step_length))
                #df_far.to_csv(
                #        cfg.dataframes.save_path + '/FAR_{}_WL:{}_SL:{}.csv'.format(cfg.audio.uri[i],window_length, step_length), index=True)
                #df_frr.to_csv(
                #        cfg.dataframes.save_path + '/FRR_{}_WL:{}_SL:{}.csv'.format(cfg.audio.uri[i], window_length, step_length),
                #        index=True)
                with open(cfg.dataframes.save_path+'/PR_AUC_Multi_2.csv', mode='a') as csv_file:
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
                    error_rate = (metrics['false alarm'] + metrics['missed detection'] + metrics['confusion'])/metrics['total']
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
                plt.title('Diarization Error Rate vs. Threshold ('+cfg.audio.uri[i]+')'+str(window_length)+'SL'+str(step_length))
                plt.xlabel('Threshold')
                plt.ylabel('DER')
                plt.savefig('DER_'+cfg.audio.uri[i]+'_WL:'+str(window_length)+'SL'+str(step_length)+'.png')
                df_DER['DER'], df_DER['false alarm'], df_DER['missed detection'], df_DER['confusion'], df_DER['total'] = der_error_rate, der_fa, der_md, der_conf, der_tot
                #df_DER.to_csv(cfg.dataframes.save_path+'/DER_{}_WL:{}_SL:{}.csv'.format(cfg.audio.uri[i],window_length, step_length), index=True)

                with open(cfg.dataframes.save_path+'/DER_Multi_2.csv', mode='a') as csv_file:
                    fieldnames = ['Window Length', 'Overlap', 'Track', 'Threshold', 'DER', 'False Alarm', 'Missed Detection', 'Confusion', 'Total']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writeheader()
                    for t, threshold in enumerate(cfg.audio.threshold):
                        writer.writerow({'Window Length': window_length,
                                         'Overlap':step_length,
                                         'Track':cfg.audio.uri[i],
                                         'Threshold':threshold,
                                         'DER':der_error_rate[t],
                                         'False Alarm':der_fa[t],
                                         'Missed Detection':der_md[t],
                                         'Confusion':der_conf[t],
                                         'Total':der_tot[t]})



#update
    #update





    '''
    emb_common = model.crop({'audio':cfg.audio.target_path,'duration':1000}, segment=excerpt)
    emb_common = np.mean(emb_common, axis=0, keepdims=True)
    track, sample_rate = torchaudio.load(cfg.audio.verification_path)
    duration = float(floor(track.size()[1]/sample_rate))

    #Convo = Prepare_Track(
    #    path_to_track='/home/lucvanwyk/pyannote-audio/tutorials/data/amicorpus/EN2002a/audio/EN2002a.Mix-Headset.wav',
    #    path_to_rttm='/home/lucvanwyk/Data/pyannote/AMI/MixHeadset.test.rttm')
    #df_convo_frame_labels1, convo_frame_list, speaker_dict_convo = Convo.label_frames(window_size=3, step_size=0.1)
    label_array, frame_list = Prepare_Track(Audio_path=cfg.audio.verification_path, RTTM_path=cfg.audio.rttm_path, window_size=cfg.audio.window_length, step_size=cfg.audio.step_length, Speaker=cfg.audio.name)

    precision, recall, roc, FAR, FRR, DER = plot_embeddings(model=model, frame_list=frame_list, frame_labels=label_array, duration=duration, target_speaker='FEO072', target_embedding=emb_common, cfg=cfg)#, threshold=threshold)
    print(precision)
    print(recall)
    print(roc)
    print(DER)
    print(duration)

    plt.figure()
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig('test_pr.png')

    plt.figure()
    plt.plot(cfg.audio.threshold, FAR)
    plt.plot(cfg.audio.threshold, FRR)
    plt.xlabel('Threshold')
    plt.ylabel('%')
    plt.savefig('test_FRR_FAR.png')
    #EER = np.argwhere(np.diff(np.sign(np.array(FRR)-np.array(FAR)))).flatten()
    #print('EER', EER)
    plt.figure()
    plt.plot(cfg.audio.threshold, DER)
    plt.xlabel('Threshold')
    plt.ylabel('DER (%)')
    plt.savefig('test_DER.png')

    '''



#main()


if __name__ == '__main__':
    main()

