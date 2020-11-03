import torchaudio
import os
import glob
import xml.etree.ElementTree as ET
from math import floor
from collections import Counter
from random import shuffle as shuffle
from random import randint
import torch
import numpy as np



class AMI_Conversation_Creator:
    def __init__(self, common_spk,perc_overlap, min_seg, max_seg, rttm_file, xml_file, max_length=20):
        self.common_spk = common_spk
        self.perc_overlap = perc_overlap
        self.min_seg = min_seg
        self.max_seg = max_seg
        self.rttm_file = rttm_file
        self.xml_file = xml_file

        # get speaker labels creates a dict containing the speaker-label-->headset information
        tree = ET.parse(xml_file)
        root = tree.getroot()
        labels = {}
        for element in root:
            for subelement in element:
                meeting = (subelement.attrib['{http://nite.sourceforge.net/}id'][:-2])
                channel = subelement.attrib['channel']
                filename = meeting + '.Headset-' + channel + '.wav'
                speaker_label = subelement.attrib['global_name']
                if filename not in labels.keys():
                    labels[filename] = speaker_label
        self.labels = labels

    def create_rttm(self, audio_path, save_path, max_length, second_speaker):
        audio_list = glob.glob(audio_path, recursive=True)
        target_spk_tracks = [k for k in audio_list if k[k.rfind('_')+1:k.rfind('.')] == self.common_spk]
        second_spk_tracks = [k for k in audio_list if k[k.rfind('_')+1:k.rfind('.')] == second_speaker]
        max_length = floor((max_length*60)/2)*16000

        target_speech = []
        for i in range(len(target_spk_tracks)):
            track, sample_rate = torchaudio.load(target_spk_tracks[i])
            track = track.numpy()[0]
            target_speech.append(track)
        target_speech = np.concatenate(target_speech)


        secondary_speech = []
        for i in range(len(second_spk_tracks)):
            track,_ = torchaudio.load(second_spk_tracks[i])
            track = track.numpy()[0]
            secondary_speech.append(track)
        secondary_speech = np.concatenate(secondary_speech)


        target_speech = target_speech[0:max_length]
        secondary_speech = secondary_speech[0:max_length]
        print(len(target_speech))


        convers_track = []
        pos =  rttm_pos = 0
        rttm_file = open(os.path.join(save_path, '{}_{}_{}_{}_{}.rttm'.format(self.common_spk, second_speaker, 20, 3, 5)), 'a')

        while pos <= (max_length):
            segment = randint(self.min_seg, self.max_seg)*16000
            speech = np.concatenate((target_speech[pos:pos+segment], secondary_speech[pos:pos+segment]))

            if len(speech) == 0:
                print(pos)
            convers_track.append(speech)

            rttm_file.write(
                'SPEAKER ' + '{}_{}_{}_{}_{}'.format(self.common_spk, second_speaker, 20, 3, 5) + ' 1 ' + str(
                    rttm_pos / 16000) + ' ' + str((segment) / 16000)
                + ' <NA> <NA> ' + self.common_spk + ' <NA> <NA> \n')

            pos = pos + segment
            rttm_pos = rttm_pos+segment

            rttm_file.write(
                'SPEAKER ' + '{}_{}_{}_{}_{}'.format(self.common_spk, second_speaker, 20, 3, 5) + ' 1 ' + str(
                    rttm_pos / 16000) + ' ' + str((segment) / 16000)
                + ' <NA> <NA> ' + second_speaker + ' <NA> <NA>\n')

            rttm_pos =rttm_pos + segment
        #rttm_file.close()


        convers_track = np.concatenate(convers_track)

        torchaudio.save(os.path.join(save_path,'{}_{}_{}_{}_{}.wav'.format(self.common_spk, second_speaker, 20, 3, 5)),torch.from_numpy(convers_track), 16000)



test = AMI_Conversation_Creator(common_spk='FTD019UID', perc_overlap=1, min_seg=6, max_seg=12, rttm_file='/home/lucas/PycharmProjects/Data/pyannote/AMI/MixHeadset.train.rttm', xml_file='/home/lucas/PycharmProjects/Data/corpusResources/meetings.xml')
#test.create_rttm(audio_path='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/*.wav', save_path='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations', max_length=20, second_speaker='FEE087')
test.create_rttm(audio_path='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/*.wav', save_path='/home/lucas/PycharmProjects/Data/pyannote/Aug_Conversations', max_length=5, second_speaker='MTD018ID')
