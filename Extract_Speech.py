import os
import glob
import torchaudio
import numpy as np
import torch
from math import floor
from collections import Counter
from random import shuffle as shuffle
import pandas as pd

def Extract_Speech_From_RTTM(Audio_File_List, RTTM_File, Save_Path, Min_Seg_Dur):
    '''
    :param Audio_File_List: List of audio files to extract speech from
    :param RTTM_File: Path to RTTM annotations file
    :param Save_Path: Where to save the extracted speech file
    :param Min_Seg_Dur: Minimum Segment Duration
    :return: None
    '''
    def get_segments(filename):
        f = open(RTTM_File)
        segments = [line.split() for line in f if (line.split()[1][:line.split()[1].find('.')] == filename[filename.rfind('/')+1:]) and (float(line.split()[4]) >= Min_Seg_Dur)]
        return segments

    def get_speaker_list(segments):
        speakers = [line[7] for line in segments]
        speakers = list(set(speakers))
        return speakers

    def Extract_Speech(filename, speaker_list, segments):
        track, sample_rate = torchaudio.load(filename+'/audio/'+filename[filename.rfind('/'):]+'.Mix-Headset.wav')
        track = track.numpy()[0]
        for speaker in speaker_list:
            timestamps = [(int(float(segment[3]) * sample_rate), int(float(segment[4]) * sample_rate)) for segment in segments if (segment[7] == speaker)]
            extracted_speech = np.zeros_like(track)
            for start, duration in timestamps:
                extracted_speech[start:start+duration] = track[start:start+duration]
            extracted_speech = extracted_speech[extracted_speech != 0]
            print('Saving track ', filename[filename.rfind('/'):])
            if len(extracted_speech) > 0:
                if os.path.isdir(os.path.join(Save_Path,filename[filename.rfind('/')+1:])):
                    torchaudio.save(os.path.join(Save_Path, filename[filename.rfind('/') + 1:], speaker + '.wav'),
                                    torch.from_numpy(extracted_speech), sample_rate)
                else:
                    os.mkdir(os.path.join(Save_Path,filename[filename.rfind('/')+1:]))
                    torchaudio.save(os.path.join(Save_Path, filename[filename.rfind('/') + 1:], speaker + '.wav'),
                                    torch.from_numpy(extracted_speech), sample_rate)


    for file in Audio_File_List:
        print(file)
        segments = get_segments(filename = file)
        speakers = get_speaker_list(segments = segments)
        if len(speakers) > 0:
        #    print(file)
            Extract_Speech(filename=file, speaker_list=speakers, segments=segments)


file_list = glob.glob('/home/lucas/PycharmProjects/Data/pyannote/amicorpus/*', recursive=True)
Extract_Speech_From_RTTM(Audio_File_List= file_list, RTTM_File='/home/lucas/PycharmProjects/Data/pyannote/AMI/MixHeadset.development.rttm', Save_Path='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech', Min_Seg_Dur=6)

