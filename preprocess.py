import librosa
import os
import numpy as np
import pandas as pd

DATASET_PATH = './../datasetFiles/CREMA-D/AudioWAV'
FEATURE_MFCC_PATH = '_mfcc_feature.npy'
CSV_PATH = '_cremad.csv'
CLEAN_CSV_PATH='cleaned_data.csv'
CLEAN_MFCC_PATH='./cleaned_mfcc_feature.npy'
SAMPLE_TO_CONSIDER = 22050
FEMALE_ID = ['1002', '1003', '1004', '1006', '1007', '1008', '1009', '1010', '1012', '1013', '1018',
             '1020', '1021', '1024', '1025', '1028', '1029', '1030', '1037', '1043', '1046', '1047',
             '1049', '1052', '1053', '1054', '1055', '1056', '1058', '1060', '1061', '1063', '1072',
             '1073', '1074', '1075', '1076', '1078', '1079', '1082', '1084', '1089', '1091']


def preprocess_dataset(dataset_path, output_path,feature_mfcc_path, n_mfcc=22, hop_length=512, n_fft=2048):
    data = {
        'data_count': {'ANG': 0, 'DIS': 0, 'FEA': 0, 'HAP': 0, 'NEU': 0, 'SAD': 0},
        'emotion': [],
        'mfcc': [],
        'filename': [],
        'sex': []  # 0:female ,1:male
    }

    for filename in os.listdir(dataset_path):
        label = filename.split('_')[2]
        data['data_count'][label] += 1

        if label == 'ANG':
            data['emotion'].append(0)
        elif label == 'DIS':
            data['emotion'].append(1)
        elif label == 'FEA':
            data['emotion'].append(2)
        elif label == 'HAP':
            data['emotion'].append(3)
        elif label == 'SAD':
            data['emotion'].append(4)
        elif label == 'NEU':
            data['emotion'].append(5)

        signal, sr = librosa.load(os.path.join(DATASET_PATH, filename))
        zero_signal = np.zeros(int(sr * 1.5))
        if len(signal) > (SAMPLE_TO_CONSIDER * 1.5):  # enforce 1.5 s
            mid = signal.size / 2
            left = int(mid - 0.75 * sr)
            right = int((mid + 0.75 * sr))
            signal = signal[left:right]
        zero_signal[:signal.shape[0]] = signal
        padded_signal = zero_signal

        mfcc = librosa.feature.mfcc(padded_signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
        data['mfcc'].append(mfcc.T.tolist())

        data['filename'].append(filename)

        if filename.split('_')[0] in FEMALE_ID:
            data['sex'].append(0)
        else:
            data['sex'].append(1)

        print(data['data_count'])

    np.save(feature_mfcc_path, data['mfcc'])

    print('file generated')

    with open(output_path, 'w') as f:
        for i in range(len(data["emotion"])):
            f.writelines(
                f'{data["emotion"][i]},{data["filename"][i]},{data["sex"][i]}{os.linesep}')

def clean_data(csv_path,mfcc_path):
    raw_data=pd.read_csv(csv_path,names=['emotion','filename','sex'])
    l=raw_data['emotion'].unique().tolist()
    l.pop(l.index(5))
    for u in l:
        count=0
        for i in raw_data.where(raw_data['emotion']==u).dropna().index:
            if count>=50:
                break
            if raw_data.loc[i,:]['sex']==1:
                raw_data.drop(index=[int(i)],inplace=True)
                count+=1
    count=0
    for j in raw_data.where(raw_data['emotion']==5).dropna().index:
        if count>=75:
            break
        if raw_data.loc[j,:]['sex']==1:
            raw_data.drop(index=[int(j)],inplace=True)
            count+=1
    
    processed_data0=pd.DataFrame(raw_data[raw_data['emotion']==0][:1000])
    processed_data1=pd.DataFrame(raw_data[raw_data['emotion']==1][:1000])
    processed_data2=pd.DataFrame(raw_data[raw_data['emotion']==2][:1000])
    processed_data3=pd.DataFrame(raw_data[raw_data['emotion']==3][:1000])
    processed_data4=pd.DataFrame(raw_data[raw_data['emotion']==4][:1000])
    processed_data5=pd.DataFrame(raw_data[raw_data['emotion']==5][:1000])
    output=pd.concat([processed_data0,processed_data1,processed_data2,processed_data3,processed_data4,processed_data5])

    output.to_csv(CLEAN_CSV_PATH,chunksize=50)
    print('csv generated')

    mfcc=np.load(mfcc_path)
    mfcc_out=mfcc[output.index]
    np.save(CLEAN_MFCC_PATH,mfcc_out,allow_pickle=True)
    print('mfccs generated')

if __name__ == '__main__':
    preprocess_dataset(DATASET_PATH, CSV_PATH,FEATURE_MFCC_PATH)
    clean_data(CSV_PATH,FEATURE_MFCC_PATH)
