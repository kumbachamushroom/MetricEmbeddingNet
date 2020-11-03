import torch
pipeline = torch.hub.load('pyannote/pyannote-audio', 'scd', pipeline=True)
scd = pipeline({'audio': '/home/lucvanwyk/Data/pyannote/Aug_Conversations/FTD019UID_MTD018ID_20_3_5.wav'})

previous = 0
for change_point in scd.get_timeline():
    if (change_point.start - previous) > 3:
        print(change_point)
        previous = change_point.end



