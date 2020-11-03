import torch
import torchaudio
pipeline = torch.hub.load('pyannote/pyannote-audio', 'sad', pipeline=True)
#speech_activity_detection = pipeline({'audio': '/home/luvanwyk/Data/pyannote/Aug_Conversations/FTD019UID_MTD018ID_20_3_5.wav'})

speech_activity_detection = pipeline({'audio': '/home/lucvanwyk/Data/pyannote/amicorpus_individual/IS1001d/audio/IS1001d.Headset-0.wav'})

#with open('/home/lucvanwyk/Data/pyannote/Aug_Conversations/audio.sad.rttm', 'w') as f:
#    speech_activity_detection.write_rttm(f)
#with open('/home/lucvanwyk/Data/pyannote/Aug_Conversations/audio.sad.rttm', 'w') as f:
#    speech_activity_detection.write_rttm(f)


utterances = []
track, _ = torchaudio.load('/home/lucvanwyk/Data/pyannote/amicorpus_individual/IS1001d/audio/IS1001d.Headset-0.wav')


for speech_region in speech_activity_detection.get_timeline():
    #print(speech_region.end - speech_region.start)
    if (speech_region.end - speech_region.start) > 3:
        print(f'There is speech between t={speech_region.start:.1f}s and t={speech_region.end:.1f}s.')
        utterances.append(track[0][int(speech_region.start*16000):int(speech_region.end*16000)])
#print(utterances)
utterances = torch.cat(utterances,dim=0)

torchaudio.save('/home/lucvanwyk/vad_test4.wav',utterances, 16000)
#torchaudio.save()
