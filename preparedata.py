import warnings
warnings.filterwarnings('ignore')

import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import librosa
import random


def features(file):
    x, sample_rate = librosa.core.load(file)
    mel = np.mean(librosa.feature.melspectrogram(x, sr=sample_rate).T, axis=0)
    result = np.hstack((np.array([]), mel))

    return result


def savefile():
    datadir = 'F:\\LocalJupyterData\\Voice\\CommonVoice'
    filename = []
    gender = []

    for file in os.listdir('Data'):
        if os.path.isdir(os.path.join('Data', file)):
            npy = os.listdir(os.path.join('Data', file))
            csvfile = file + '.csv'
            df = pd.read_csv(os.path.join(datadir, csvfile))

            for j in tqdm(range(len(npy)), f'Extracting for {file}'):
                path = file + '/' + npy[j][:-4] + '.mp3'
                if path in list(df['filename']):
                    filename.append(file + '/' + npy[j])
                    gender.append(df['gender'][list(df['filename']).index(path)])

    newdf = pd.DataFrame({'filename': filename, 'gender': gender})
    newdf.to_csv('finaldata.csv', index=False)


def main():
    if not os.path.isdir('Data'):
        os.mkdir('Data')

    datadir = 'F:\\LocalJupyterData\\Voice\\CommonVoice\\'
    csv = glob(datadir + '*.csv')

    for files in csv:
        if 'invalid' not in files:
            filename = files.split('\\')[-1].split('.')[0]

            df = pd.read_csv(files)

            gendf = df[['filename', 'gender']]
            gendf = gendf[gendf['gender'].notna()]

            r = np.where(np.array(gendf['gender']) == 'male')[0]
            random.shuffle(r)
            r = r[:list(gendf['gender']).count('female')]
            r = np.append(r, np.where(np.array(gendf['gender']) == 'female')[0])

            finallist = list(np.array(gendf['filename'])[r])
            
            for file in tqdm(finallist, f'Extracting for {filename}'):
                dirname = 'Data\\' + file.split('/')[0]
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)

                np.save(dirname + '\\' + file.split('/')[1].split('.')[0], features(datadir + file.split('/')[0] + '\\' + file.split('/')[0] + '\\' + file.split('/')[1]))


if __name__ == '__main__':
    main()
    savefile()
