#This script is used to:
#1) Extract speech from given headset given rttm/xml annotations
#2) Create sample list with train/test/valid splits

import os
import glob
import xml.etree.ElementTree as ET
import torchaudio
import numpy as np
import torch
from math import floor
from collections import Counter
from random import shuffle as shuffle
import pandas as pd

class Generate_Sample_Lists:
    def __init__(self, audio_PATH, rttm_PATH, xml_PATH, save_PATH):
        '''
        :param audio_PATH: absolute path to individual headset mixes
        :param rttm_PATH: absolute path to rttm annotations (use pyannote's train set)
        :param xml_PATH: absolute path to xml file containing speaker-headset information (meetings.xml)
        :param save_PATH: absolute path to where to save files
        '''
        self.audio_PATH = audio_PATH
        self.rttm_PATH = rttm_PATH
        self.xml_PATH = xml_PATH
        self.save_PATH = save_PATH
        self.track_list = glob.glob(self.audio_PATH, recursive=True)

    def get_speaker_labels(self):
        #get speaker labels creates a dict containing the speaker-label-->headset information
        tree = ET.parse(self.xml_PATH)
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
        return labels

    def extract_speech(self, labels):
        """
        extracts speech from given file and saves that as a new .wav file which only contains speech from the desired
        speaker
        :param track: track to extract speech from
        :param labels: dictionary of speaker labels
        :return: None
        """
        for track in self.track_list:

            try:
                filename = track[track.rfind('/') + 1:]
                speaker_label = labels[filename]
                #track_array = torchaudio.backend.sox_backend.load_wav(track, normalization=False)
                track_array, sample_rate = torchaudio.load(track)
                track_array = track_array.numpy()[0]
                print(track_array)
                rttm = open(self.rttm_PATH)
            except:
                print("Could not extract speech")
            else:
                filename = filename[:filename.find('.')]
                lines = [line.split() for line in rttm if (filename in line.split()[1]) and (line.split()[7] == speaker_label)]
                timestamps = [(int(float(line[3]) * sample_rate), int(float(line[4])*sample_rate)) for line in lines]

                extracted_speech = np.empty_like(track_array)
                for start, duration in timestamps:
                    extracted_speech[start:start+duration] = track_array[start:start+duration]
                new_filename = filename+'_'+speaker_label+'.wav'
                extracted_speech = extracted_speech[extracted_speech != 0]

                if len(extracted_speech) > 0:
                    try:
                        #torchaudio.backend.sox_backend.save(filepath=os.path.join(self.save_PATH,new_filename), src=extracted_speech, sample_rate=sample_rate)
                        print(max(extracted_speech))
                        torchaudio.save(os.path.join(self.save_PATH, new_filename), torch.from_numpy(extracted_speech), sample_rate)
                    except:
                        print("Could not save while {}".format(filename))

    def create_sample_list(self, snippet_length=3, filename='sample_list.txt'):
        '''
        Just a loop that extracts speech for every track in the list
        :return: None
        '''

        self.track_list = glob.glob(self.save_PATH + '/*.wav', recursive=True)

        for track in self.track_list:
            self.label_speech(track=track, labels=self.get_speaker_labels,
                              snippet_length=snippet_length, list_name=filename)

    def label_speech(self,track, labels, snippet_length, list_name):
        filename = track[track.rfind('/') + 1:]
        path = track
        speaker_label = track[track.rfind('_')+1:track.rfind('.')]
        track, sample_rate = torchaudio.load(path)
        num_samples = floor(len(track[0])/(snippet_length*sample_rate))
        sample_length = snippet_length*sample_rate #sample length in num of samples
        print("the number of samples is ", num_samples)
        f = open(os.path.join(self.save_PATH, list_name), 'a')
        for i in range(num_samples):
            start_time = int(i*sample_length)
            end_time = int(start_time + sample_length)
            f.write(path+"\t"+speaker_label+"\t" + str(start_time)+"\t" + str(end_time)+"\n")

    def trim_samples(self,max_samples,filename, new_filename):
        """
        Takes the sample_list file and trims the file down so that it has the number of samples per speaker
        as specificed in max_samples
        :param max_samples: number of samples per speaker
        :return: None
        """
        samples = []
        trimmed_samples = []
        for line in open(os.path.join(self.save_PATH, filename)):
            samples.append((line.split()[0], line.split()[1], line.split()[2], line.split()[3]))
        speakers = [sample[1] for sample in samples]
        unique_speakers = Counter(speakers).keys()
        for i, speaker in enumerate(unique_speakers):
            sample_speaker = [sample for sample in samples if sample[1] == speaker]
            try:
                sample_speaker = sample_speaker[0:max_samples]
            finally:
                trimmed_samples.extend(sample_speaker)
        f = open(os.path.join(self.save_PATH,new_filename), 'a')
        for i in enumerate(trimmed_samples):
            #shuffle(trimmed_samples)
            f.write(
                trimmed_samples[i[0]][0] + "\t" + trimmed_samples[i[0]][1] + "\t" + str(list(unique_speakers).index(trimmed_samples[i[0]][1]))+ "\t" + trimmed_samples[i[0]][2] + "\t" +
                trimmed_samples[i[0]][3] + "\n")

    def dataset_split(self, filename, train=0.7, test=0.2):
        #get list of samples
        samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in open(os.path.join(self.save_PATH, filename))]
        shuffle(samples)
        train_set = samples[0:int(len(samples)*train)]
        print(len(train_set))
        test_set = samples[int(len(samples)*train):int(len(samples)*(test+train))]
        print(len(test_set))
        try:
            valid_set = samples[int(len(samples)*(test+train)):]
            for i in range(len(valid_set)):
                f = open(os.path.join(self.save_PATH, filename[:filename.find('.')] + '_valid.txt'), 'a')
                f.write(
                    valid_set[i][0] + "\t" + valid_set[i][1] + "\t" + valid_set[i][2] + "\t" + valid_set[i][3] + "\t" + valid_set[i][4] + "\n")
        except:
            print("Could not create validation set")
        f = open(os.path.join(self.save_PATH,filename[:filename.find('.')]+'_train.txt'),'a')
        for i in range(len(train_set)):
            f.write(train_set[i][0]+"\t"+train_set[i][1]+"\t"+train_set[i][2]+"\t"+train_set[i][3]+ "\t" + train_set[i][4] + "\n")
        f = open(os.path.join(self.save_PATH, filename[:filename.find('.')] + '_test.txt'), 'a')
        for i in range(len(test_set)):
            f.write(test_set[i][0]+"\t"+test_set[i][1]+"\t"+test_set[i][2]+"\t"+test_set[i][3]+"\t" + test_set[i][4] +"\n")

    def describe_dataset(self, filename):
        speakers = []
        samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in
                   open(os.path.join(self.save_PATH, filename))]
        speakers = [sample[1] for sample in samples]
        speakers = Counter(speakers).keys()
        pd_dataset = pd.DataFrame(data=None, columns=['Gender','Num_Samples'], index=speakers)

        for speaker in speakers:
            num_speaker = [sample for sample in samples if sample[1] == speaker]
       #     print(len(num_speaker), speaker)
            if speaker[0] == 'F':
                gender = 'Female'
            else:
                gender = 'Male'
            pd_dataset.loc[speaker, 'Num_Samples'] = len(num_speaker)
            pd_dataset.loc[speaker, 'Gender'] = gender
        print(pd_dataset)
        print(pd_dataset.describe())
        print(pd_dataset['Gender'].value_counts())
       # print(pd_dataset.describe())



if os.uname()[1] != 'lucas-FX503VD':
    test = Generate_Sample_Lists(audio_PATH='/home/lucvanwyk/Data/pyannote/amicorpus_individual/**/*.wav',
                                 save_PATH='/home/lucvanwyk/Data/pyannote/Extracted_Speech',
                                 xml_PATH='/home/lucvanwyk/Data/corpusResources/meetings.xml',
                                 rttm_PATH='/home/lucvanwyk/Data/pyannote/AMI/MixHeadset.train.rttm')
    #test.extract_speech(labels=test.get_speaker_labels())
    test.create_sample_list(snippet_length=3, filename='sample_test_list.txt')
    test.trim_samples(max_samples=100, filename='sample_test_list.txt', new_filename='trimmed_sample_list.txt')
    test.dataset_split(filename='trimmed_sample_list.txt', train=0.7, test=0.2)

else:
    test = Generate_Sample_Lists(audio_PATH='/home/lucas/PycharmProjects/Data/pyannote/amicorpus_individual/**/*.wav',
                             save_PATH='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech',
                             xml_PATH='/home/lucas/PycharmProjects/Data/corpusResources/meetings.xml',
                             rttm_PATH='/home/lucas/PycharmProjects/Data/pyannote/AMI/MixHeadset.train.rttm')
    #test.extract_speech(labels=test.get_speaker_labels())
    #test.create_sample_list(snippet_length=3,filename='sample_test_list.txt')
    #test.trim_samples(max_samples=200,filename='sample_test_list.txt',new_filename='trimmed_sample_list.txt')
    #test.dataset_split(filename='trimmed_sample_list.txt',train=0.7, test=0.2)
    test.describe_dataset(filename='trimmed_sample_list_test.txt')














