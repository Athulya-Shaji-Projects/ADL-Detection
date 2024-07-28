import os
import csv

import pydub as pydub
from pydub import AudioSegment

#pydub.AudioSegment.converter = 'c:\\FFmpeg\\bin\\ffmpeg.exe'

# Directory containing MP3 files
input_folder = 'C:/Users/DELL/Desktop/MP/Dataset_new/Microwave'

# Directory to save WAV files
output_folder = 'C:/Users/DELL/Desktop/MP/Dataset_wav/fold4'

# CSV file path
csv_file_path = 'C:/Users/DELL/Desktop/MP/Book3.csv'

# Create a CSV file with headers
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['slice_file_name', 'fold', 'classID','class']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each file in the input folder
    for file_name in os.listdir(output_folder):
        writer.writerow({'slice_file_name': file_name, 'fold': '4', 'classID': '4', 'class': 'Washing Machine'})
            #if file_name.endswith('.mp3'):  # Check if the file is an MP3 file
            #mp3_file_path = os.path.join(input_folder + '/', str(file_name))  # Get the full path of the MP3 file
            #wav_file_name = file_name[:-4] + '.wav'  # Generate the name for the WAV file
            #wav_file_path = os.path.join(output_folder + '/', str(file_name))  # Get the full path of the WAV file
            #print(mp3_file_path)
            #print(wav_file_path)
            #Load MP3 file
            #mp3_file = AudioSegment.from_file(mp3_file_path, format='mp3')

            # Convert MP3 to WAV
            #mp3_file.export(wav_file_path, format='wav')

            #print(f"Converted {file_name} to {wav_file_name}")
            # Write file names to CSV
            #writer.writerow({'slice_file_name': file_name, 'fold':'4','classID' : '4','class':'Microwave'})

print("Conversion complete!")
print(f"CSV file saved at {csv_file_path}")
