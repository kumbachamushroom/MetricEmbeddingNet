import os
import torchaudio
from math import floor
import numpy as np
import pandas as pd
from math import fabs


def Prepare_Track(Audio_path, RTTM_path, Speaker, window_size, step_size):

    def get_frames():
        track, sample_rate = torchaudio.load(Audio_path)
        track_length = len(track.numpy()[0])  # Get number of samples (length) of track
        del track  # We only need the number of samples (length) of the track
        n_increments = floor(
            ((floor(track_length / sample_rate) - window_size) / step_size))  # Number of steps taken by sliding window
        frame_list = []
        for i in range(
                n_increments + 2):  # create list of size N_frames that contain the start and stop time of each frame (convert time to samples)
            start_time = i * step_size
            end_time = start_time + window_size
            frame_list.append((start_time, end_time))
        return frame_list



    rttm = open(RTTM_path)



    segments = [line.split() for line in rttm if (line.split()[7] == Speaker)]

    frame_list = get_frames()
    label_array = np.zeros(len(frame_list))
    for i, time in enumerate(frame_list):
        start_time, stop_time = time[0], time[1]
            #print(start_time, stop_time)
        utterances = [line for line in segments if (
                    (start_time <= float(line[3])) and (stop_time > float(line[3])) and (
                         (stop_time - float(line[3])) > window_size/2)) or (
                                    (start_time < (float(line[4]) + float(line[3]))) and (
                                        stop_time >= (float(line[4]) + float(line[3]))) and (
                                                (float(line[3]) + float(line[4]) - start_time) > window_size/2)) or (
                                    (start_time >= float(line[3])) and (
                                        stop_time <= (float(line[3]) + float(line[4]))))]

        for k, line in enumerate(utterances):
            label_array[i] = 1

    print("Frame labelling succesfull")
    label_df = pd.DataFrame(data=label_array)
    return label_array, frame_list



#label_df, frame_list = Prepare_Track(Audio_path='/home/lucas/PycharmProjects/Data/pyannote/amicorpus/IB4001.Mix-Headset.wav', RTTM_path='/home/lucas/PycharmProjects/Data/pyannote/AMI/MixHeadset.development.rttm', window_size=3, step_size=0.1, Speaker='FIE038')

#print(label_df)
#print('done')

def Prepare_Track_Multi_Label(Audio_path, RTTM_path, window_size, step_size):
    def get_intersection(start1, start2, end1, end2):
        intersection = max(0, min(end1, end2)-max(start1, start2))
        #if intersection > 0:
           # print(intersection)
        return intersection

    def get_frames():
        track, sample_rate = torchaudio.load(Audio_path)
        track_length = len(track.numpy()[0])  # Get number of samples (length) of track
        del track  # We only need the number of samples (length) of the track
        n_increments = floor(
            ((floor(track_length / sample_rate) - window_size) / step_size))  # Number of steps taken by sliding window
        frame_list = []
        for i in range(
                n_increments + 2):  # create list of size N_frames that contain the start and stop time of each frame (convert time to samples)
            start_time = i * step_size
            end_time = start_time + window_size
            frame_list.append((start_time, end_time))
        return frame_list

    def create_speaker_dict(speaker_list):
        speaker_dict = {}
        for i, speaker in enumerate(speaker_list):
            speaker_dict[speaker] = i+1
        return speaker_dict



    rttm = open(RTTM_path)
    segments = [line.split() for line in rttm if (line.split()[1] == Audio_path[Audio_path.rfind('/')+1:Audio_path.rfind('.')])]
    speaker_list = list(set([segment[7] for segment in segments]))
    frame_list = get_frames()
    label_array = np.zeros(len(frame_list))
    speaker_dict = create_speaker_dict(speaker_list)
    #print(speaker_dict)
    for i, time in enumerate(frame_list):
        start_time, stop_time = time[0], time[1]
            #print(start_time, stop_time)
        utterances = [line for line in segments if (get_intersection(start_time, float(line[3]), stop_time, (float(line[3]) + float(line[4]))) > (window_size/3))]
        #print(utterances)
        if len(utterances) > 0:
            max_length = 0
            for k, line in enumerate(utterances):
                intersection = get_intersection(start_time, float(line[3]), stop_time, (float(line[3]) + float(line[4])))
                if intersection > max_length:
                    label = speaker_list.index(line[7])+1
                    max_length = intersection
                #print(label)
                label_array[i] = label
    print("Frame labelling succesfull")

    speaker_df = pd.DataFrame(columns=speaker_list, data=np.zeros(shape=(len(frame_list), len(speaker_list)), dtype=int))
    for i, label in enumerate(label_array):
        if label > 0:
            speaker_df.loc[i, speaker_list[int(label)-1]] = 1

    return frame_list, speaker_df


#label_df, frame_list, test = Prepare_Track_Multi_Label(Audio_path='/home/lucas/PycharmProjects/Data/pyannote/amicorpus/IB4001.Mix-Headset.wav', RTTM_path='/home/lucas/PycharmProjects/Data/pyannote/AMI/MixHeadset.development.rttm', window_size=3, step_size=2)
#print('done')