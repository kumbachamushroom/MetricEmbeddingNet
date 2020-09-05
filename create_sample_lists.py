import os
import glob
import xml.etree.ElementTree as ET
import torchaudio
import numpy as np
import torch
from math import floor
from collections import Counter
from random import shuffle

class Generate_Sample_List:
    def __init__(self,track_list, path_rttm, path_xml, save_path):
        #self.path_audio = path_audio
        self.track_list = track_list
        self.path_rttm = path_rttm
        self.path_xml = path_xml
        self.save_path = save_path

    def get_speaker_labels(self):
        """
        get_speaker_labels create dictionary containing the speaker label - headset relationships
        :param xml_path: path to the meetings.xml file
        :return: a dictionary with telling you which speaker was using which headset in which conversation
                example : {'MEE095', 'EN2009c.Headset-1.wav': 'FEE083', 'EN2009c.Headset-2.wav': 'MEE094',
                'EN2009c.Headset-0.wav': 'MEE095',
                'EN2009d.Headset-0.wav': 'FEE083', 'EN2009d.Headset-2.wav': 'MEE094', 'EN2009d.Headset-3.wav': 'MEE095',
                'EN2009d.Headset-1.wav': 'FEE096'}
        """
        tree = ET.parse(self.path_xml)
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

    def create_sample_list(self, snippet_length=3, list_name='sample_list.txt'):
        '''
               Just a loop that extracts speech for every track in the list
               :return: None
               '''

        self.track_list = glob.glob(self.save_path + '/*', recursive=True)

        for track in self.track_list:
            self.label_speech(track=track, labels=self.get_speaker_labels,
                              snippet_length=snippet_length, list_name = list_name)

    def extract_speech(self):
        """
        extracts speech from given file and saves that as a new .wav file which only contains speech from the desired
        speaker
        :param track: track to extract speech from
        :param labels: dictionary of speaker labels
        :return: None
        """
        labels = self.get_speaker_labels()
        for track in self.track_list:

            try:
                filename = track[track.rfind('/') + 1:]
                print(filename)
                speaker_label = labels[filename]
                track_sox = torchaudio.backend.sox_backend.load(track, normalization=True)
                #print(track_sox)
                track_array, sample_rate = track_sox[0], track_sox[1]
                track_array = track_array[0]
                rttm = open(self.path_rttm)
            except:
                print("Could not extract speech")
            else:
                filename = filename[:filename.find('.')]
                lines = [line.split() for line in rttm if (filename in line.split()[1]) and (line.split()[7] == speaker_label)]
                timestamps = [(int(float(line[3]) * sample_rate), int(float(line[4])*sample_rate)) for line in lines]
                extracted_speech = torch.empty_like(track_array)
                for start, duration in timestamps:
                    extracted_speech[start:start+duration] = track_array[start:start+duration]
                new_filename = filename+'_'+speaker_label+'.pt'
                extracted_speech = extracted_speech[extracted_speech != 0]
                #print(np.isnan(np.sum(extracted_speech)))
                if torch.isnan(extracted_speech).any() == False:
                    try:
                        torch.save(extracted_speech, os.path.join(self.save_path, new_filename))
                    except:
                        print("Could not save while {}".format(new_filename))
                else:
                    print("{} contained NAN values ".format(new_filename))

    def label_speech(self,track, labels, snippet_length, list_name):
        filename = track[track.rfind('/') + 1:]
        path = track
        speaker_label = track[track.rfind('_')+1:track.rfind('.')]
        print(track)
        track = torch.load(track)
        sample_rate = 16000
        sample_length = snippet_length * sample_rate #sample length in num of samples
        num_samples = floor(len(track)/(sample_length))
        print("the number of samples is ", num_samples)
        f = open(os.path.join(self.save_path,list_name), 'a')
        for i in range(num_samples):
            start_time = int(i*sample_length)
            end_time = int(start_time + sample_length)
            f.write(path+"\t"+speaker_label+"\t" + str(start_time)+"\t" + str(end_time)+"\n")

    def trim_samples(self,max_samples, sample_list_name, new_list_name):
        """
        Takes the sample_list file and trims the file down so that it has the number of samples per speaker
        as specificed in max_samples
        :param max_samples: number of samples per speaker
        :return: None
        """
        samples = []
        trimmed_samples = []
        for line in open(os.path.join(self.save_path,sample_list_name)):
            samples.append((line.split()[0], line.split()[1], line.split()[2], line.split()[3]))
        shuffle(samples)
        speakers = [sample[1] for sample in samples]
        unique_speakers = Counter(speakers).keys()
        for i, speaker in enumerate(unique_speakers):
            sample_speaker = [sample for sample in samples if sample[1] == speaker]
            try:
                sample_speaker = sample_speaker[0:max_samples]
            finally:
                trimmed_samples.extend(sample_speaker)
        f = open(os.path.join(self.save_path, new_list_name), 'a')
        for i in enumerate(trimmed_samples):
            #shuffle(trimmed_samples)
            f.write(
                trimmed_samples[i[0]][0] + "\t" + trimmed_samples[i[0]][1] + "\t" + str(list(unique_speakers).index(trimmed_samples[i[0]][1]))+ "\t" + trimmed_samples[i[0]][2] + "\t" +
                trimmed_samples[i[0]][3] + "\n")


if os.uname()[1] != 'lucas-FX503VD':
    #Get a list of all the track from which you would like to extract speech
    track_list = glob.glob('/home/lucvanwyk/Data/pyannote/amicorpus_individual/**/*.wav', recursive=True)
    #print(track_list)
    #create Generate_Triplet_List object
    obj = Generate_Sample_List(path_rttm='/home/lucvanwyk/Data/pyannote/AMI/MixHeadset.train.rttm',save_path='/home/lucvanwyk/Data/pyannote/Extracted_Tensor', path_xml='/home/lucvanwyk/Data/corpusResources/meetings.xml',track_list=track_list)
    #obj.extract_speech()
    obj.create_sample_list(snippet_length=0.5, list_name='sample_list_tensors_3.txt')
    obj.trim_samples(max_samples=10000,sample_list_name='sample_list_tensors_3.txt', new_list_name='trimmed_samples_tensor_3.txt')
else:
    # Get a list of all the track from which you would like to extract speech
    track_list = glob.glob('/home/lucas/PycharmProjects/Data/pyannote/amicorpus_individual/**/*.wav', recursive=True)
    # print(track_list)
    # create Generate_Triplet_List object
    obj = Generate_Sample_List(path_rttm='/home/lucas/PycharmProjects/Data/pyannote/AMI/MixHeadset.train.rttm',
                               save_path='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech',
                               path_xml='/home/lucas/PycharmProjects/Data/corpusResources/meetings.xml', track_list=track_list)
    #obj.extract_speech()
    obj.create_sample_list(snippet_length=1, list_name='sample_list_tensors_0.25.txt')
    obj.trim_samples(max_samples=600, sample_list_name='sample_list_tensors_0.25.txt',
                     new_list_name='trimmed_samples_tensor_0.25.txt')

#test = torch.load('/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/EN2001e_MEE067.pt')
#print(test.size())
#torchaudio.save('/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/testfile.wav', test, 16000)

#test = torchaudio.backend.sox_backend.load('/home/lucvanwyk/Data/pyannote/amicorpus_individual/EN2001d/audio/EN2001d.Headset-0.wav', normalization=True, num_frames=100)
#print(test[0])
#cpnnection?