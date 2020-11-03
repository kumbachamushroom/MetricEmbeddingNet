import torch.utils.data.dataloader as dataloader
import os
import torch
import torchaudio
import numpy as np
from random import sample as Sampler
from random import shuffle as shuffle
import matplotlib.pyplot as plt
import numpy as np
import math
import time

PATH = "/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits"

class TripletLoader(torch.utils.data.Dataset):
    def __init__(self,base_path, anchor_positive_pairs,sample_list,train=True):
        self.base_path = base_path
        self.anchor_positive_pairs = []
        self.sample_list = []
        self.train = train
        for line in open(os.path.join(self.base_path, sample_list)):
            self.sample_list.append((line.split()[0], line.split()[1]))
        for line in open(os.path.join(self.base_path,anchor_positive_pairs)):
            negative = Sampler([sample[0] for sample in self.sample_list if sample[1] != line.split()[2]],1)
            self.anchor_positive_pairs.append((line.split()[0],line.split()[1], negative[0]))
            #print(line.split()[0], line.split()[1], negative[0])
        shuffle(self.anchor_positive_pairs)
        if self.train:
            self.anchor_positive_pairs = self.anchor_positive_pairs[0:int(0.8*len(self.anchor_positive_pairs))]
        else:
            self.anchor_positive_pairs = self.anchor_positive_pairs[int(0.8*len(self.anchor_positive_pairs)):]



    def __getitem__(self,index):
        anchor, positive, negative = str(self.anchor_positive_pairs[index][0]), str(self.anchor_positive_pairs[index][1]), str(self.anchor_positive_pairs[index][2])
        anchor, sample_rate = torchaudio.load(os.path.join(self.base_path,anchor))
        positive, sample_rate = torchaudio.load(os.path.join(self.base_path,positive))
        negative, sample_rate = torchaudio.load(os.path.join(self.base_path,negative))
        anchor_specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(anchor)
        positive_specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(positive)
        #negatives = [sample[0] for sample in self.sample_list if sample[1] != label]
        #random_state = np.random.RandomState(29)
        #negative = Sampler(negatives,1)
        #negative, sample_rate = torchaudio.load(os.path.join(self.base_path,negative[0]))
        negative_specgram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(negative)
        return anchor_specgram,positive_specgram,negative_specgram

    def __len__(self):
        return len(self.anchor_positive_pairs)


class Selective_Loader(torch.utils.data.Dataset):
    def __init__(self, base_path, sample_list, label, train=True, negative=True):
        self.base_path = base_path
        self.sample_list = sample_list
        self.train = train
        self.label = label
        self.samples = []
        self.negative = negative
        for line in open(os.path.join(self.base_path, sample_list)):
            self.samples.append((line.split()[0], line.split()[1]))
        if self.negative:
            self.samples = [sample[0] for sample in self.samples if sample[1] != self.label]
            print(self.samples)
            shuffle(self.samples)
        else:
            self.samples = [sample[0] for sample in self.samples if sample[1] == self.label]
            print(self.samples)
            shuffle(self.samples)

    def __getitem__(self, index):
        sample_name = str(self.samples[index])
        track , sample_rate = torchaudio.load(os.path.join(self.base_path, sample_name))
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track)
        return spectrogram

#test_loader = Negative_Loader(base_path="/home/lucas/PycharmProjects/Papers_with_code/data/AMI/triplet_splits", sample_list="sample_list.txt",label='5', train=True, negative=False)

class Frame_Loader:
    def __init__(self,path, frame_list):
        self.path = path
        self.frame_list = frame_list
        self.track, self.sample_rate = torchaudio.load(self.path)

    def __getitem__(self, index):
        start_sample = self.frame_list[index][0]*self.sample_rate
        end_sample = self.frame_list[index][1]*self.sample_rate
        spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(self.track)
        return spectrogram

    def __len__(self):
        return len(self.frame_list)

class Triplet_Time_Loader:
    def __init__(self, path, spectrogram=True, train=True):
        self.path = path
        self.as_spectrogram = spectrogram
        self.samples = []
        self.samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in open(self.path)]
        #shuffle(self.samples)
        if train:
            self.samples = self.samples[0:int(0.8 * len(self.samples))]
            print("TRAIN LENGTH", len(self.samples))
        else:
            self.samples = self.samples[int(0.8 * len(self.samples)):]
            print("TEST LENGTH", len(self.samples))

    def __getitem__(self, index):
        sample, string_label, int_label, start_time, stop_time = self.samples[index][0], self.samples[index][1], int(self.samples[index][2]), int(self.samples[index][3]), int(self.samples[index][4])
        track, sample_rate = torchaudio.load(sample)
        print(torch.mean(track))
        print(sample_rate)
        track = track[0][(start_time):(stop_time)]
        if self.as_spectrogram:
            track = track.view(1, -1)
            print(track)
            spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track)
            return spectrogram, torch.tensor(int_label), string_label
        else:
            return track, torch.tensor(int_label), string_label

    def __len__(self):
        return len(self.samples)




class Spectrogram_Loader(torch.utils.data.Dataset):
    def __init__(self, filename, mel=False, one_hot=False):
        """
        Load spectrograms from given input audio snippet
        :param filename: .txt file containing snippet information
        :param mel: bool --> return melspectrogram
        """
        self.path = filename
        self.mel = mel
        self.one_hot = one_hot
        self.samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line
                        in open(self.path)]
        #print(self.samples[2])
        self.classes = len(list(set([int(sample[2]) for sample in self.samples])))
        #print("The length is: {}".format(len(self.samples)))
    def __getitem__(self, index):
        sample, string_label, int_label, start_time, stop_time = self.samples[index][0], self.samples[index][1], int(
            self.samples[index][2]), int(self.samples[index][3]), int(self.samples[index][4])
        #track, sample_rate = torchaudio.backend.sox_backend.load_wav(sample, normalization=False)
        track, sample_rate = torchaudio.load(sample)
        track = track[0][(start_time):(stop_time)]
        track = track.view(1, -1)
        wkwargs = {'requires_grad': False}
        if self.mel == False:
            #print(track.size())
            spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, window_fn=torch.hamming_window,win_length=int(sample_rate*25e-3), hop_length=int(sample_rate*10e-3), normalized=True, power=1,wkwargs=wkwargs)(track)
            #Spectrogram size --> [1, 257, 301]
            if self.one_hot:
                one_hot_vector = [0 for _ in range(self.classes)]
                one_hot_vector[int_label] = 1
                return spectrogram, one_hot_vector, string_label
            else:
                return spectrogram, torch.tensor(int_label), string_label
        else:
            spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=512,sample_rate=16000, win_length=int(sample_rate*25e-3),hop_length=int(sample_rate*10e-3), window_fn=torch.hamming_window, normalized=True, power=1, wkwargs=wkwargs)(track)
            # Spectrogram size --> [1, 128, 301]
            return spectrogram, torch.tensor(int_label), string_label

    def __len__(self):
        return len(self.samples)

#test = Spectrogram_Loader(filename='/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/trimmed_sample_list.txt', mel=False, one_hot=True)

class Spectrogram_FrameWise_Loader(torch.utils.data.Dataset):
    '''
    This class is the same as Spectrogram_Loader but it returns a tensor of overlapping frames
    '''
    def __init__(self, filename, frame_list, mel=False):
        self.path = filename
        self.mel = mel
        self.frame_list = frame_list
        self.track, self.sample_rate = torchaudio.load(self.path)

    def __getitem__(self, index):
        start_sample = math.ceil(self.frame_list[index][0] * self.sample_rate)
        #print('start ', start_sample)
        end_sample = math.ceil(self.frame_list[index][1] * self.sample_rate)
        #print('end ', end_sample)
        wkwargs = {'requires_grad': False}
        if self.mel == False:
            spectrogram = torchaudio.transforms.Spectrogram(n_fft=512,
                                                            window_fn=torch.hamming_window,
                                                            win_length=int(self.sample_rate*25e-3),
                                                            hop_length=int(self.sample_rate*10e-3),
                                                            normalized=True,
                                                            power=1,wkwargs=wkwargs)(self.track[0][start_sample: end_sample].view(1,-1))
            if (spectrogram.size()[2] < 301):
                spectrogram = torch.nn.functional.pad(spectrogram, (0,(301 - spectrogram.size()[2])), "constant", 0)

            #Spectrogram size --> [1, 257, 301]
            #print(self.frame_list[index][1] - self.frame_list[index][0])
            return spectrogram

        else:
            spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=512,
                                                               sample_rate=self.sample_rate,
                                                               win_length=int(self.sample_rate*25e-3)
                                                               ,hop_length=int(self.sample_rate*10e-3)
                                                               ,window_fn=torch.hamming_window,
                                                               normalized=True, power=1, wkwargs=wkwargs)(self.track[0][start_sample, end_sample].view(1,-1))
            # Spectrogram size --> [1, 128, 301]
            return spectrogram, self.frame_list[index]

    def __len__(self):
        return len(self.frame_list)



'''
class Triplet_Tensor_Loader:
    def __init__(self, path,spectrogram=True, train=True):
        self.path = path
        self.as_spectrogram = spectrogram
        self.samples = []
        self.samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in open(self.path)]
        #shuffle(self.samples)
        if train:
            self.samples = self.samples[0:int(0.8 * len(self.samples))]
            print("TRAIN LENGTH", len(self.samples))
        else:
            self.samples = self.samples[int(0.8 * len(self.samples)):]
            print("TEST LENGTH", len(self.samples))

    def __getitem__(self, index):
        sample, string_label, int_label, start_time, stop_time = self.samples[index][0], self.samples[index][1], int(self.samples[index][2]), int(self.samples[index][3]), int(self.samples[index][4])
        track = torch.load(sample)
        track = track[(start_time):(stop_time)]
        if self.as_spectrogram:
            track = track.view(1, -1)
            spectrogram = torchaudio.transforms.Spectrogram(normalized=True, power=1, n_fft=400, hop_length=100)(track)
            return spectrogram, torch.tensor(int_label), string_label
        else:
            return track, torch.tensor(int_label), string_label

    def __len__(self):
        return len(self.samples)
'''


class Single_Speaker_Loader:
    """
        The Single_Speaker_Loader class is used to load samples from a SINGLE SPECIFIED SPEAKER
        path: Path to the sample list (txt file)
        speaker: Label of desired speaker
    """
    def __init__(self,path, speaker, mel):
        self.path = path
        self.samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in open(self.path)]
        self.samples = [sample for sample in self.samples if (sample[1] == speaker)]
        self.mel = mel

    def __getitem__(self, index):
        sample, string_label, int_label, start_sample, end_sample = self.samples[index][0], self.samples[index][1], int(
            self.samples[index][2]), int(self.samples[index][3]), int(self.samples[index][4])
        track, sample_rate = torchaudio.load(sample)
        track = track[0][start_sample:end_sample]
        track = track.view(1, -1)
        wkwargs = {'requires_grad': False}
        if self.mel:
            spectrogram = torchaudio.transforms.MelSpectrogram(n_fft=512,
                                                               sample_rate=sample_rate,
                                                               win_length=int(sample_rate*25e-3)
                                                               ,hop_length=int(sample_rate*10e-3)
                                                               ,window_fn=torch.hamming_window,
                                                               normalized=True, power=1, wkwargs=wkwargs)(track)
            return spectrogram, torch.tensor(int_label), string_label
        else:
            spectrogram = torchaudio.transforms.Spectrogram(n_fft=512,
                                                            window_fn=torch.hamming_window,
                                                            win_length=int(sample_rate * 25e-3),
                                                            hop_length=int(sample_rate * 10e-3),
                                                            normalized=True,
                                                            power=1, wkwargs=wkwargs)(track)
            return spectrogram #, torch.tensor(int_label), string_label

    def __len__(self):
        return len(self.samples)

class Window_Loader(torch.utils.data.Dataset):
    def __init__(self, filename,windowed=True, window_length=0.2, overlap=0.01):
        self.path = filename
        self.samples = []
        self.samples = [(line.split()[0], line.split()[1], line.split()[2], line.split()[3], line.split()[4]) for line in open(self.path)]
        self.window_length = int(window_length*16000)
        self.overlap = int(overlap*16000)
        self.windowed = windowed

    def __getitem__(self, index):
        sample, string_label, int_label, start_time, stop_time = self.samples[index][0], self.samples[index][1], int(self.samples[index][2]), int(self.samples[index][3]), int(self.samples[index][4])
        track = torchaudio.backend.sox_backend.load(sample, normalization=True)
        track, sample_rate = track[0], track[1]
        track = track[0][(start_time):(stop_time)]
        if self.windowed:
            n_windows = 1 + math.floor((len(track)-self.window_length)/(self.window_length-self.overlap))
            n_windows = int(n_windows)
            window_tensor = torch.zeros(n_windows, self.window_length)
            for i in range(n_windows):
                offset = self.overlap*i
                window_tensor[i, :] = track[offset:offset+self.window_length]
            return window_tensor, torch.tensor(int_label), string_label
        else:
            return track, torch.tensor(int_label), string_label

    def __len__(self):
        return len(self.samples)











#test = Triplet_Time_Loader(path=os.path.join('/home/lucas/PycharmProjects/Papers_with_code/data/AMI/amicorpus_individual/Extracted_Speech','trimmed_sample_list.txt'))
#print(test.__getitem__(1))


