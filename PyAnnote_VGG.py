import torch
import numpy as np
from pyannote.core import Segment
from scipy.spatial.distance import cdist

model = torch.hub.load('pyannote/pyannote-audio', 'emb_ami')
print(f'Embedding has dimension {model.dimension:d}.')

import numpy as np
from pyannote.core import Segment
#embedding = model({'audio': '/home/lucas/PycharmProjects/Data/pyannote/Extracted_Speech/ES2002d_MEE007.wav'})
#for window, emb in embedding:
#    assert isinstance(window, Segment)
#    assert isinstance(emb, np.ndarray)

excerpt1 = Segment(start=90.0, end=120.0)

emb1 = model.crop({'audio': '/home/lucvanwyk/Data/pyannote/Extracted_Speech/Test_Set/EN2002a_FEO072.wav', 'duration':770.0},segment=excerpt1)
print(torch.from_numpy(emb1).size())

excerpt2 = Segment(start=23.0, end=26.0)
emb2 = model.crop({'audio':'/home/lucvanwyk/Data/pyannote/Extracted_Speech/Test_Set/EN2002a_MEE071.wav', 'duration':770.0}, segment=excerpt2)
print(torch.from_numpy(emb2).size())

excerpt3 = Segment(start=20.0, end=23.0)
emb3 = model.crop({'audio':'/home/lucvanwyk/Data/pyannote/Extracted_Speech/Test_Set/EN2002a_FEO072.wav', 'duration':932.0}, segment=excerpt3)
print(torch.from_numpy(emb3).size())

distance = cdist(np.mean(emb1, axis=0, keepdims=True),
                 np.mean(emb2, axis=0,  keepdims=True),
                 metric='euclidean')[0][0]
print(distance)

distance = cdist(np.mean(emb1, axis=0, keepdims=True),
                 np.mean(emb3, axis=0,  keepdims=True),
                 metric='euclidean')[0][0]
print(distance)

distance = cdist(np.mean(emb2, axis=0, keepdims=True),
                 np.mean(emb3, axis=0,  keepdims=True),
                 metric='euclidean')[0][0]
print(distance)



print('done')