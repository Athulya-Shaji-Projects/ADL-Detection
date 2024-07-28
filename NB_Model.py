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

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=10)
# 5-fold cross-validation, you can change the number of folds here

# Print the cross-validation scores
print("Cross-validation scores: ", cv_scores)
print("Mean CV score: ", np.mean(cv_scores))

from sklearn.metrics import precision_score, recall_score, roc_auc_score
# Compute precision, recall, and AUC
precision = precision_score(y_test, y_pred,average='micro')
recall = recall_score(y_test, y_pred,average='micro')
#auc = roc_auc_score(y_test, y_pred)

print("Precision: ", precision)
print("Recall: ", recall)
#print("AUC: ", auc)

filename="C:/Users/DELL/Desktop/MP/Dataset_wav/fold1/071788_door-openclose_1-2aif-89957.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

#print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
predicted_label=clf.predict(mfccs_scaled_features)
pred_proba = clf.predict_proba(mfccs_scaled_features)
print(predicted_label)
print(pred_proba)
#prediction_class = labelencoder(predicted_label)
#prediction_class