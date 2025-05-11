import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy
class IEMOCAPDataset(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText_,\
        self.videoAudio_, self.videoVisual_, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')


        self.videoText = pickle.load(open('IEMOCAP/TextFeatures.pkl', 'rb'))
        self.videoAudio = pickle.load(open('IEMOCAP/AudioFeatures.pkl', 'rb'))
        self.videoVisual = pickle.load(open('IEMOCAP/VisualFeatures.pkl', 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]


        return torch.FloatTensor(numpy.array(self.videoText[vid])), \
            torch.FloatTensor(numpy.array(self.videoText_[vid])), \
            torch.FloatTensor(numpy.array(self.videoVisual_[vid])), \
            torch.FloatTensor(numpy.array(self.videoAudio_[vid])), \
            torch.FloatTensor(numpy.array(self.videoVisual[vid])), \
            torch.FloatTensor(numpy.array(self.videoAudio[vid])), \
            torch.FloatTensor(numpy.array([[1, 0] if x == 'M' else [0, 1] for x in \
                                           self.videoSpeakers[vid]])), \
            torch.FloatTensor(numpy.array([1] * len(self.videoLabels[vid]))), \
            torch.LongTensor(numpy.array(self.videoLabels[vid])), \
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 7 else pad_sequence(dat[i], True) if i < 9 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):

    def __init__(self, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText_,\
        self.videoAudio_, self.videoVisual_, self.videoSentence, self.trainVid,\
        self.testVid, _ = pickle.load(open('MELD_features/MELD_features_raw1.pkl', 'rb'))


        self.videoAudio = pickle.load(open('/MELD/AudioFeatures.pkl', 'rb'))
        self.videoVisual = pickle.load(open('/MELD/VisualFeatures.pkl', 'rb'))
        self.videoText = pickle.load(open('/MELD/TextFeatures.pkl', 'rb'))

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(numpy.array(self.videoText[vid])),\
               torch.FloatTensor(numpy.array(self.videoText_[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual_[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio_[vid])),\
               torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
               torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
               torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
               torch.FloatTensor(numpy.array([1]*len(self.videoLabels[vid]))),\
               torch.LongTensor(numpy.array(self.videoLabels[vid])),\
               vid  

    def __len__(self):
        return self.len

    def return_labels(self):
        return_label = []
        for key in self.keys:
            return_label+=self.videoLabels[key]
        return return_label

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 7 else pad_sequence(dat[i], True) if i < 9 else dat[i].tolist() for i in dat]
