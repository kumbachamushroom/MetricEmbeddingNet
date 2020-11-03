from pyannote.database.util import load_rttm
from pyannote.core import Segment, notebook
from pyannote.audio.features import RawAudio
#from IPython.display import Audio
import torch
from pyannote.metrics.diarization import DiarizationErrorRate

Audio_File = {'uri':'ES2011a.Mix-Headset', 'audio':'/home/lucas/PycharmProjects/Data/pyannote/amicorpus/ES2011a/audio/ES2011a.Mix-Headset.wav'}
groundtruth = load_rttm('/home/lucas/PycharmProjects/Data/pyannote/AMI/MixHeadset.development.rttm')[Audio_File['uri']]
for segment in groundtruth.get_timeline():
    print(list(groundtruth.get_labels(segment))[0])

pipeline = torch.hub.load('pyannote/pyannote-audio','dia_ami')
diarization = pipeline(Audio_File)

#print(diarization)

metric = DiarizationErrorRate(collar=0.25, skip_overlap=True)
der = metric(groundtruth, diarization)

print(der)
#print('done')