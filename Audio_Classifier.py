import joblib
import librosa
import numpy as np
import wave
import os
import pandas as pd
import csv
import datetime

# Load the saved SVM model from the file
svm = joblib.load('svm_model.pkl')

# Load the ADL sound
filename='C:/Users/DELL/Desktop/MP/Realtime_data/sample.wav'
# Feature extraction
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
# predition
predicted_label=svm.predict(mfccs_scaled_features)
pred_proba = svm.predict_proba(mfccs_scaled_features)
print(predicted_label)
print(pred_proba.max())
ADL_class = predicted_label[0]
# classification on bases of accuracy of prediction
if pred_proba.max() < 0.7:
    print("Is it a sound of",predicted_label, " : yes/no")
    input1 = input()
    if input1 == 'yes':
        ADL_class = predicted_label[0]
        fold = np.argmax(pred_proba, axis=1) + 1
        fold = fold[0]
        audio_dataset_path = 'C:/Users/DELL/Desktop/MP/Dataset_wav_20'
        audio_dataset_path = os.path.join(audio_dataset_path, 'fold' + str(fold))
        # list all files in the folder
        files = os.listdir(audio_dataset_path)
        # count the number of files
        num_files = len(files) + 1
        audio_dataset_path = os.path.join(audio_dataset_path, 'audio' + str(num_files) + '.wav')
    elif input1 == 'no':
        print("What sound is it !!")
        ADL_class = input()
        audio_dataset_path = 'C:/Users/DELL/Desktop/MP/Dataset_wav_20'
        # read the CSV file into a pandas DataFrame
        df = pd.read_csv('C:/Users/DELL/Desktop/MP/metadata_20.csv')
        # check if a value exists in a specific column
        if ADL_class in df['class'].values:
            row = df.loc[df['class'] == ADL_class]
            fold = row['fold'].iloc[0]
            audio_dataset_path = os.path.join(audio_dataset_path, 'fold' + str(fold))
            # list all files in the folder
            files = os.listdir(audio_dataset_path)
            num_files = len(files) + 1
            audio_dataset_path = os.path.join(audio_dataset_path, 'audio' + str(num_files) + '.wav')
        else:
            # list all files in the folder
            files = os.listdir(audio_dataset_path)
            # count the number of files
            fold = len(files) + 1
            audio_dataset_path = os.path.join(audio_dataset_path, 'fold' + str(fold))
            os.makedirs(audio_dataset_path)
            num_files = 1
            audio_dataset_path = os.path.join(audio_dataset_path, 'audio' + str(num_files) + '.wav')

    print(audio_dataset_path)
    # Adding the new ADL to the respective folder
    # Open the original WAV file
    with wave.open('C:/Users/DELL/Desktop/MP/Realtime_data/sample.wav', 'rb') as original_file:
        # Get the parameters of the WAV file
        params = original_file.getparams()
        # Read the data from the WAV file
        data = original_file.readframes(original_file.getnframes())
        # Create a new WAV file in a directory
        with wave.open(audio_dataset_path, 'wb') as new_file:
            # Set the parameters of the new WAV file
            new_file.setparams(params)
            # Write the data to the new WAV file
            new_file.writeframes(data)
    # Adding the metadata of the new ADL signal
    metadata = pd.read_csv('C:/Users/DELL/Desktop/MP/metadata_20.csv')
    with open('C:/Users/DELL/Desktop/MP/metadata_20.csv', mode='a', newline='') as file:
        # create a writer object
        writer = csv.writer(file)
        # write a new row with the current date and time in separate columns
        writer.writerow(['audio' + str(num_files), fold, fold, ADL_class])
    # close the file
    file.close()

# Creation of ADL log
# open the CSV file in append mode
with open('C:/Users/DELL/Desktop/MP/ADL_log.csv', mode='a', newline='') as file:
    # create a writer object
    writer = csv.writer(file)
    # get current date and time
    now = datetime.datetime.now()
    # write a new row with the current date and time in separate columns
    writer.writerow([now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),ADL_class])
# close the file
file.close()