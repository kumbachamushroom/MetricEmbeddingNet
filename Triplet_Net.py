import torch
import torch.nn as nn
import torch.nn.functional as F
from vgg_utils import get_conv2d_output_shape
import numpy as np
import sys
from torch.autograd import Variable
import math

class VGGVox(nn.Module):
     """VGGVox implementation

     Reference
     ---------
     Arsha Nagrani, Joon Son Chung, Andrew Zisserman. "VoxCeleb: a large-scale
     speaker identification dataset."

     """

     def __init__(self, dimension=256):

         super().__init__()


         self.dimension = dimension

         h = 201  # 512 in VoxCeleb paper. 201 in practice.
         w = 481 # typically 3s with 10ms steps

         self.conv1_ = nn.Conv2d(1, 96, (7, 7), stride=(2, 2), padding=1)
         # 254 x 148 when n_features = 512
         # 99 x 148 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (7, 7), stride=(2, 2), padding=1)

         self.bn1_ = nn.BatchNorm2d(96)
         self.mpool1_ = nn.MaxPool2d((3, 3), stride=(2, 2))
         # 126 x 73 when n_features = 512
         # 49 x 73 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))

         self.conv2_ = nn.Conv2d(96, 256, (5, 5), stride=(2, 2), padding=1)
         # 62 x 36 when n_features = 512
         # 24 x 36 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (5, 5), stride=(2, 2), padding=1)

         self.bn2_ = nn.BatchNorm2d(256)
         self.mpool2_ = nn.MaxPool2d((3, 3), stride=(2, 2))
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(2, 2))

         self.conv3_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

         self.bn3_ = nn.BatchNorm2d(256)

         self.conv4_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

         self.bn4_ = nn.BatchNorm2d(256)

         self.conv5_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
         # 30 x 17 when n_features = 512
         # 11 x 17 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

         self.bn5_ = nn.BatchNorm2d(256)

         self.mpool5_ = nn.MaxPool2d((5, 3), stride=(3, 2))
         # 9 x 8 when n_features = 512
         # 3 x 8 when n_features = 201
         h, w = get_conv2d_output_shape((h, w), (5, 3), stride=(3, 2))

         self.fc6_ = nn.Conv2d(256, 4096, (h, 1), stride=(1, 1))
         # 1 x 8
         h, w = get_conv2d_output_shape((h, w), (h, 1), stride=(1, 1))

         self.fc7_ = nn.Linear(4096, 1024)
         self.fc8_ = nn.Linear(1024, self.dimension)


     def forward(self, sequences):
         """Embed sequences

         Parameters
         ----------
         sequences : torch.Tensor (batch_size, n_samples, n_features)
             Batch of sequences.

         Returns
         -------
         embeddings : torch.Tensor (batch_size, dimension)
             Batch of embeddings.
         """


         x = sequences
         #print(x)
         #print(x.size())
        # x = torch.transpose(sequences, 1, 2).view(
             #30, 1, 3, 3)

         # conv1. shape => 254 x 148 => 126 x 73
         x = self.mpool1_(F.relu(self.bn1_(self.conv1_(x))))

         # conv2. shape =>
         x = self.mpool2_(F.relu(self.bn2_(self.conv2_(x))))

         # conv3. shape = 62 x 36
         x = F.relu(self.bn3_(self.conv3_(x)))

         # conv4. shape = 30 x 17
         x = F.relu(self.bn4_(self.conv4_(x)))

         # conv5. shape = 30 x 17
         x = self.mpool5_(F.relu(self.bn5_(self.conv5_(x))))

         # fc6. shape =
         x = F.dropout(F.relu(self.fc6_(x)))

         # (average) temporal pooling. shape =
         x = torch.mean(x, dim=-1)

         # fc7. shape =
         x = x.view(x.size(0), -1)
         x = F.dropout(F.relu(self.fc7_(x)))

         # fc8. shape =
         x = self.fc8_(x)

         return x

class TripletNet(nn.Module):
    def __init__(self, embeddingnet):
        super(TripletNet,self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(embedded_x,embedded_y,2)
        dist_b = F.pairwise_distance(embedded_x,embedded_z,2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
    y_left = flip(y_right, 0)

    y = torch.cat([y_left, Variable(torch.ones(1)).cuda(), y_right])

    return y


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1,
                               steps=int((self.kernel_size / 2)))  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1,
                                                         -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (
                    self.n_ / 2)) * self.window_  # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET).
        # I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class sinc_conv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs):
        super(sinc_conv, self).__init__()

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (fs / 2) - 100

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs).cuda()

        min_freq = 50.0;
        min_band = 50.0;

        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)

        n = torch.linspace(0, N, steps=N)

        # Filter window (hamming)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N);
        window = Variable(window.float().cuda())

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)

            band_pass = band_pass / torch.max(band_pass)

            filters[i, :] = band_pass.cuda() * window

        out = F.conv1d(x, filters.view(self.N_filt, 1, self.Filt_dim))

        return out

#An activation function that does literally nothing
class no_activation(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self,x):
        return x

def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!

    if act_type == "none":
        return no_activation()


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        #print(mean)
        #mean[torch.isnan(mean)] == 0
        #print(mean)
        std = x.std(-1, keepdim=True)
        #print(std)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MLP(nn.Module):
    def __init__(self, options):
        super(MLP, self).__init__()

        self.input_dim = int(options['input_dim'])
        self.fc_lay = options['fc_lay']
        self.fc_drop = options['fc_drop']
        self.fc_use_batchnorm = options['fc_use_batchnorm']
        self.fc_use_laynorm = options['fc_use_laynorm']
        self.fc_use_laynorm_inp = options['fc_use_laynorm_inp']
        self.fc_use_batchnorm_inp = options['fc_use_batchnorm_inp']
        self.fc_act = options['fc_act']

        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        # input layer normalization
        if self.fc_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        # input batch normalization
        if self.fc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        self.N_fc_lay = len(self.fc_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_fc_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.fc_drop[i]))

            # activation
            self.act.append(act_fun(self.fc_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.fc_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.fc_lay[i], momentum=0.05))

            if self.fc_use_laynorm[i] or self.fc_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.fc_lay[i], bias=add_bias))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.fc_lay[i], current_input).uniform_(-np.sqrt(0.01 / (current_input + self.fc_lay[i])),
                                                                     np.sqrt(0.01 / (current_input + self.fc_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))

            current_input = self.fc_lay[i]

    def forward(self, x):

        # Applying Layer/Batch Norm
        if bool(self.fc_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.fc_use_batchnorm_inp):
            x = self.bn0((x))

        for i in range(self.N_fc_lay):

            if self.fc_act[i] != 'linear':

                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

                if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
                    x = self.drop[i](self.act[i](self.wx[i](x)))

            else:
                if self.fc_use_laynorm[i]:
                    x = self.drop[i](self.ln[i](self.wx[i](x)))

                if self.fc_use_batchnorm[i]:
                    x = self.drop[i](self.bn[i](self.wx[i](x)))

                if self.fc_use_batchnorm[i] == False and self.fc_use_laynorm[i] == False:
                    x = self.drop[i](self.wx[i](x))

        return x


class SincNet(nn.Module):

    def __init__(self, options):
        super(SincNet, self).__init__()

        self.cnn_N_filt = options['cnn_N_filt']
        self.cnn_len_filt = options['cnn_len_filt']
        self.cnn_max_pool_len = options['cnn_max_pool_len']

        self.cnn_act = options['cnn_act']
        self.cnn_drop = options['cnn_drop']

        self.cnn_use_laynorm = options['cnn_use_laynorm']
        self.cnn_use_batchnorm = options['cnn_use_batchnorm']
        self.cnn_use_laynorm_inp = options['cnn_use_laynorm_inp']
        self.cnn_use_batchnorm_inp = options['cnn_use_batchnorm_inp']

        self.input_dim = int(options['input_dim'])

        self.fs = options['fs']

        self.N_cnn_lay = len(options['cnn_N_filt'])
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)

        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)

        current_input = self.input_dim

        for i in range(self.N_cnn_lay):

            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])

            # dropout
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act.append(act_fun(self.cnn_act[i]))

            # layer norm initialization
            self.ln.append(
                LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))

            self.bn.append(
                nn.BatchNorm1d(N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]),
                               momentum=0.05))

            if i == 0:
                self.conv.append(SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs))

            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])

        self.out_dim = current_input * N_filt

    def forward(self, x):



        batch = x.shape[0]
        seq_len = x.shape[1]

        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                if i == 0:
                    x = self.drop[i](
                        self.act[i](self.ln[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len[i]))))
                else:
                    x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))

            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))

        x = x.view(batch, -1)
        return x
