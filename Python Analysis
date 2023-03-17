conda install -c conda-forge librosa

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('TESS Toronto emotional speech set data'):
    for filename in filenames:
        print(filename)
        
## Load the Dataset

paths = []
labels = []
# Need to change path once in server
for dirname, _, filenames in os.walk('TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')

## Check 
len(paths)
paths[:5]


## Create a dataframe

df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


## Exploratory Data Analysis

sns.countplot(df['label'])

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveplot(data, sr=sr)
    plt.show()
    
def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    
## Importing the candidate's recordings

# Audio 1

emotion = ''
# Path of files
path = "C:/Users/dxd210021/Downloads/hr-RecordRTC-1668074036352-blob.wav"
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)

# Audio 2

emotion = ''
# Path of files
path = "C:/Users/dxd210021/Downloads/hr-RecordRTC-1668074108566-blob.wav"
data, sampling_rate = librosa.load(path)
librosa.display.waveshow(data, sr=sampling_rate)
spectogram(data, sampling_rate, emotion)
Audio(path)
