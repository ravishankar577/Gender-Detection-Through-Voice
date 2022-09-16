import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack
from time import sleep

from train import modelcreate
from preparedata import features


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30


def is_silent(snd_data):
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r


def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r


def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, output=True, frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    
    return sample_width, r


def record_to_file(path):
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def main(new=True, delfile=True, file='test.wav'):
    model = modelcreate()
    model.load_weights('Results/model.h5')

    file = os.path.join('Recordings', file)

    if new:
        print('Listening..')
        record_to_file(file)

    predfeat = features(file).reshape(1, -1)
    male_prob = model.predict(predfeat)[0][0]

    female_prob = 1 - male_prob
    gender = 'male' if male_prob > female_prob else 'female'

    if delfile:
        os.remove(file)

    return gender, female_prob, male_prob


if __name__ == '__main__':
    model = modelcreate()
    model.load_weights('Results/model.h5')

    file = 'test.wav'

    new = input('Do you want to test new ? ')

    if new == 'y':
        os.remove(file)
        print('Listening.. \n  ------ PLEASE READ THE TEXT GIVEN BELOW ------ \n Properly fitted masks can help prevent the spread of the virus from the person wearing the mask to others. \n Masks alone do not protect against COVID-19, and should be combined with physical distancing and hand hygiene. \n Follow the advice provided by your local health authority.')
        record_to_file(file)
    elif new == 'n':
        if not os.path.isfile(file):
            print('No existing file found, record now.')
            sleep(1)
            print('Listening..')
            record_to_file(file)

    predfeat = features(file).reshape(1, -1)
    male_prob = model.predict(predfeat)[0][0]

    female_prob = 1 - male_prob
    gender = 'male' if male_prob > female_prob else 'female'
    if(gender) :
        # print('Your Result:', gender)
        print(f'Male: {male_prob * 100:.2f}% \nFemale: {female_prob * 100:.2f}%')
    else :
        record_to_file(file)
   