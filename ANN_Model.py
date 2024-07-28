import librosa.display
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


#### Extracting MFCC's For every audio file



audio_dataset_path='C:/Users/DELL/Desktop/MP/Dataset_wav'
metadata=pd.read_csv('C:/Users/DELL/Desktop/MP/metadata.csv')
print(metadata.head())


### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
#    file_name = os.path.join(os.path.abspath(filename),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    file_name = os.path.join(audio_dataset_path, 'fold' + str(row["fold"]) + '/', str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
print(extracted_features_df.head())

### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())

### Label Encoder
import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import numpy as np
from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
#from keras.optimizers import Adam
from sklearn import metrics

### No of classes
num_labels=y.shape[1]

model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

## Trianing my model
from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)

test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])

#model.predict_classes(X_test)
predictions = (model.predict(X_test) > 0.5).astype("int32")
print(predictions)

filename="C:/Users/DELL/Desktop/MP/Dataset_wav/fold1/071788_door-openclose_1-2aif-89957.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=(model.predict(mfccs_scaled_features) > 0.5).astype("int32")
print(predicted_label)
#prediction_class = labelencoder(predicted_label)
#prediction_class'''