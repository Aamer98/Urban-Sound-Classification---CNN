import keras.backend as K
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

# Required libraries
import os
import pandas as pd
import IPython as IP
import struct
import matplotlib.pyplot as plt
import numpy as np
# Set your path to the original dataset
us8k_path = os.path.abspath('/home/aamer/Desktop/UrbanSound8K' )

# Global settings
metadata_path = os.path.join(us8k_path, 'metadata/UrbanSound8K.csv')
audio_path = os.path.join(us8k_path, 'audio')

print("Loading CSV file {}".format(metadata_path))

# Ltqdmoad metadata as a Pandas dataframe
metadata = pd.read_csv(metadata_path)

# Examine dataframe's head
metadata.head()
import sys
import os
import IPython as IP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import metrics 
import os
import pickle
import time
import struct
import sys
import os
import IPython as IP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from IPython.display import clear_output, display
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import metrics 
import os
import pickle
import time
import struct




train={}
test={}
#Preparing empty lists for 10 folds
for fno in range(1,11):
  train['X_train{}'.format(fno)]=[]
  test['X_test{}'.format(fno)]=[]
  train['y_train{}'.format(fno)]=[]
  test['y_test{}'.format(fno)]=[]



def add_noise(data):
    noise = np.random.rand(len(data))
    noise_amp = random.uniform(0.005, 0.008)
    data_noise = data + (noise_amp * noise)
    return data_noise


features = []
labels = []
frames_max = 0
counter = 0
total_samples = len(metadata)
n_mels=40
tone_steps = [-1, -2, 1, 2]
rates = [0.81, 1.07]
path="/home/aamer/Desktop/UrbanSound8K/audio/fold"
from tqdm import tqdm

for i in tqdm(range(len(metadata))):
    fold_no=str(metadata.iloc[i]["fold"])
    file=metadata.iloc[i]["slice_file_name"]
    label=metadata.iloc[i]["classID"]
    label_name=metadata.iloc[i]["class"]
    filename=path+fold_no+"/"+file
    
   
    mels = get_mel_spectrogram(filename, 0, n_mels=n_mels)
    melsa=spec_augment(mels)
    num_frames = mels.shape[1]

    y, sr = librosa.load(filename)
    y_changed = librosa.effects.time_stretch(y, rate=rates[0])
    melst1 = get_aug_mel_spectrogram(y_changed, sr, 0, n_mels=n_mels)
    melst1a=spec_augment(melst1)
    
    y_changed = librosa.effects.time_stretch(y, rate=rates[1])
    melst2 = get_aug_mel_spectrogram(y_changed,sr, 0, n_mels=n_mels)
    melst2a=spec_augment(melst2)


    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_steps[0])
    melsp1 = get_aug_mel_spectrogram(y_changed,sr, 0, n_mels=n_mels)
    melsp1a=spec_augment(melsp1)


    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_steps[1])
    melsp2 = get_aug_mel_spectrogram(y_changed,sr, 0, n_mels=n_mels)
    melsp2a=spec_augment(melsp2)


    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_steps[2])
    melsp3= get_aug_mel_spectrogram(y_changed,sr, 0, n_mels=n_mels)
    melsp3a=spec_augment(melsp3)


    y_changed = librosa.effects.pitch_shift(y, sr, n_steps=tone_steps[3])
    melsp4= get_aug_mel_spectrogram(y_changed,sr, 0, n_mels=n_mels)
    melsp4a=spec_augment(melsp4)


    y_changed = add_noise(y)
    melsn= get_aug_mel_spectrogram(y_changed,sr, 0, n_mels=n_mels)
    melsna=spec_augment(melsn)

    
    features.append(mels)
    labels.append(label)

    if (num_frames > frames_max):
      frames_max = num_frames
    
    
    test["X_test{}".format(fold_no)].append(mels)
    test["y_test{}".format(fold_no)].append(label)

    
    
    if label_name=='children_playing':

      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp3)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp4)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsn)
      test["y_test{}".format(fold_no)].append(label)     
    
    
    
    
    
    
    
    elif label_name=='drilling':

 


      test["X_test{}".format(fold_no)].append(melsn)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsa)
      test["y_test{}".format(fold_no)].append(label)


      test["X_test{}".format(fold_no)].append(melsna)
      test["y_test{}".format(fold_no)].append(label)




    
   

    
    elif label_name=='dog_bark':
    
      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsa)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst1a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2a)
      test["y_test{}".format(fold_no)].append(label)     
    



    elif label_name=='engine_idling':

      test["X_test{}".format(fold_no)].append(melsn)
      test["y_test{}".format(fold_no)].append(label)


    
    elif label_name=='jackhammer':
      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp3)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp4)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsn)
      test["y_test{}".format(fold_no)].append(label)


    
    
    elif label_name=='siren':


      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp3)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp4)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsn)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsa)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst1a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp1a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp2a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp3a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsp4a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsna)
      test["y_test{}".format(fold_no)].append(label)


    
    elif label_name=='car_horn':


      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsa)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst1a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2a)
      test["y_test{}".format(fold_no)].append(label)     
    
    elif label_name=='street_music':


      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsa)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst1a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2a)
      test["y_test{}".format(fold_no)].append(label)  

    elif label_name=='gun_shot':


      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melsa)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst1a)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2a)
      test["y_test{}".format(fold_no)].append(label)     
    



    elif label_name=='air_conditioner':    
    

      test["X_test{}".format(fold_no)].append(melst1)
      test["y_test{}".format(fold_no)].append(label)

      test["X_test{}".format(fold_no)].append(melst2)
      test["y_test{}".format(fold_no)].append(label)



    for fno in range(1,11):
      
      
      if fold_no!=str(fno):
        
        train["X_train{}".format(fold_no)].append(mels)
        train["y_train{}".format(fold_no)].append(label)
        
        if label_name=='jackhammer':
          
          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp2)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp3)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp4)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsn)
          train["y_train{}".format(fold_no)].append(label)



        elif label_name=='air_conditioner':



          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)




        
        elif label_name=='drilling':



          train["X_train{}".format(fold_no)].append(melsn)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsa)
          train["y_train{}".format(fold_no)].append(label)



          train["X_train{}".format(fold_no)].append(melsna)
          train["y_train{}".format(fold_no)].append(label) 


        



        elif label_name=='engine_idling':
        


          train["X_train{}".format(fold_no)].append(melsn)
          train["y_train{}".format(fold_no)].append(label)



        
        elif label_name=='children_playing':


          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp2)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp3)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp4)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsn)
          train["y_train{}".format(fold_no)].append(label)
        



        elif label_name=='siren':

          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp2)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp3)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp4)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsn)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsa)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst1a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp1a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp2a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp3a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsp4a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melsna)
          train["y_train{}".format(fold_no)].append(label)       
        
        

        elif label_name=='street_music':
          

          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)
       
          train["X_train{}".format(fold_no)].append(melsa)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst1a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2a)
          train["y_train{}".format(fold_no)].append(label) 


        elif label_name=='car_horn':
          

          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)
       
          train["X_train{}".format(fold_no)].append(melsa)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst1a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2a)
          train["y_train{}".format(fold_no)].append(label)   


        elif label_name=='dog_bark':

          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)
       
          train["X_train{}".format(fold_no)].append(melsa)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst1a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2a)
          train["y_train{}".format(fold_no)].append(label)   

        elif label_name=='gun_shot':

          train["X_train{}".format(fold_no)].append(melst1)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2)
          train["y_train{}".format(fold_no)].append(label)
       
          train["X_train{}".format(fold_no)].append(melsa)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst1a)
          train["y_train{}".format(fold_no)].append(label)

          train["X_train{}".format(fold_no)].append(melst2a)
          train["y_train{}".format(fold_no)].append(label)         


    
    counter += 1
    
