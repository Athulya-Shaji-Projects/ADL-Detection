import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

filename='C:/Users/DELL/Desktop/MP/Dataset_wav/fold4/Untitled.wav'
### sample Sound
plt.figure(figsize=(14,5))
data,sample_rate=librosa.load(filename)
librosa.display.waveshow(data,sr=sample_rate)
ipd.Audio(filename)
plt.show()

# Compute spectrogram
spectrogram = librosa.feature.melspectrogram(y=data, sr=sample_rate)

# Plot spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                         y_axis='mel', fmax=8000,
                         x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
print('Data',data)
print('Sample Rate',sample_rate)
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print('MFCC Shape',mfccs.shape)
print('MFCC',mfccs)


