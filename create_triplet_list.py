import os
import glob
import numpy as np
from pydub import AudioSegment
import xml.etree.ElementTree as ET
import torchaudio
import numpy as np
import torch
from math import floor
from collections import Counter
from random import shuffle

class Generate_Triplet_List:

    def __init__(self,path_audio,path_rttm, save_path, path_xml,track_list):
        """
        :param path_audio: Path to audio files
        :param path_rttm: Path to rttm annotations
        :param track_list: List of audio files
        :param save_path: Path to save sample list to
        """
        self.path_audio = path_audio
        self.path_rttm = path_rttm
        self.save_path = save_path
        self.path_xml = path_xml
        self.track_list = track_list

    def create_sample_list(self, snippet_length=3):
        '''
        Just a loop that extracts speech for every track in the list
        :return: None
        '''

        self.track_list = glob.glob(self.save_path + '/*', recursive=True)

        for track in self.track_list:
            self.label_speech(track=track, labels=self.get_speaker_labels,
                              snippet_length=snippet_length)

    def get_speaker_labels(self):
        """
        get_speaker_labels create dictionary containing the speaker label - headset relationships
        :param xml_path: path to the meetings.xml file
        :return: a dictionary with telling you which speaker was using which headset in which conversation
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
                print(filename)
                speaker_label = labels[filename]
                track_array, sample_rate = torchaudio.load(track)
                track_array = track_array.numpy()[0]
                rttm = open(self.path_rttm)
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
                try:
                    torchaudio.save(self.save_path+'/'+new_filename, torch.from_numpy(extracted_speech), sample_rate)
                except:
                    print("Could not save while {}".format(filename))

    def label_speech(self,track, labels, snippet_length):
        print(track)
        filename = track[track.rfind('/') + 1:]
        path = track
        speaker_label = track[track.rfind('_')+1:track.rfind('.')]
        print(speaker_label)
        track, sample_rate = torchaudio.load(path)
        num_samples = floor(len(track[0])/(snippet_length*sample_rate))
        f = open(self.save_path+'/sample_list.txt', 'a')
        for i in range(num_samples):
            start_time = i*3
            end_time = start_time + 3
            f.write(path+"\t"+speaker_label+"\t" + str(start_time)+"\t" + str(end_time)+"\n")

    def trim_samples(self,max_samples):
        """
        Takes the sample_list file and trims the file down so that it has the number of samples per speaker
        as specificed in max_samples
        :param max_samples: number of samples per speaker
        :return: None
        """
        samples = []
        trimmed_samples = []
        for line in open(os.path.join(self.save_path,'sample_list.txt')):
            samples.append((line.split()[0], line.split()[1], line.split()[2], line.split()[3]))
        print(samples)
        speakers = [sample[1] for sample in samples]
        unique_speakers = Counter(speakers).keys()
        for i, speaker in enumerate(unique_speakers):
            shuffle(samples)
            sample_speaker = [sample for sample in samples if sample[1] == speaker]
            try:
                sample_speaker = sample_speaker[0:max_samples]
            finally:
                trimmed_samples.extend(sample_speaker)
        f = open(self.save_path+'/trimmed_sample_list.txt', 'a')
        for i in enumerate(trimmed_samples):
            shuffle(trimmed_samples)
            f.write(
                trimmed_samples[i[0]][0] + "\t" + trimmed_samples[i[0]][1] + "\t" + str(list(unique_speakers).index(trimmed_samples[i[0]][1]))+ "\t" + trimmed_samples[i[0]][2] + "\t" +
                trimmed_samples[i[0]][3] + "\n")








#Get a list of all the track from which you would like to extract speech
track_list = glob.glob('/home/lucvanwyk/MetricEmbeddingNet/pyannote/amicorpus_individual/**/*.wav', recursive=True)
print(track_list)


#create Generate_Triplet_List object
obj = Generate_Triplet_List(path_rttm='/home/lucvanwyk/MetricEmbeddingNet/pyannote/AMI/MixHeadset.train.rttm',path_audio='/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/EN2001a/audio',save_path='/home/lucvanwyk/MetricEmbeddingNet/pyannote/Extracted_Speech', path_xml='/home/lucvanwyk/MetricEmbeddingNet/corpusResources/meetings.xml',track_list=track_list, snippet_length=0.2)



#Create sampl_list and trimmed_sample_list
obj.extract_speech(labels=obj.get_speaker_labels())
obj.create_sample_list(snippet_length=0.2)
obj.trim_samples(max_samples=10000)




