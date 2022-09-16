import os
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


label = {'female': 0, 'male': 1}


def load(vec=128):
    if not os.path.isdir('Results'):
        os.mkdir('Results')

    if os.path.isfile('Results/features.npy') and os.path.isfile('Results/labels.npy'):
        return np.load('Results/features.npy'), np.load('Results/labels.npy')

    df = pd.read_csv('finaldata.csv')
    x = np.zeros((len(df), vec))
    y = np.zeros((len(df), 1))

    for i, (filename, gender) in tqdm(enumerate(zip(df['filename'], df['gender'])), 'Loading data', total=len(df)):
        features = np.load('Data\\' + filename)
        x[i] = features
        y[i] = label[gender]

    np.save('Results/features', x)
    np.save('Results/labels', y)
    return x, y


def modelcreate(vec=128):
    model = Sequential()
    model.add(Dense(512, input_shape=(vec,)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')

    return model


def main():
    x, y = load()
    
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, random_state=7)
    xTrain, xValid, yTrain, yValid = train_test_split(xTrain, yTrain, test_size=0.1, random_state=7)

    data = {'xTrain': xTrain, 'xValid': xValid, 'xTest': xTest, 'yTrain': yTrain, 'yValid': yValid, 'yTest': yTest}
    
    model = modelcreate()
    print(model.summary())

    early_stopping = EarlyStopping(mode='min', patience=5, restore_best_weights=True)

    batch = 64
    epochs = 100

    model.fit(data['xTrain'], data['yTrain'], epochs=epochs, batch_size=batch, validation_data=(data['xValid'], data['yValid']), callbacks=[early_stopping])

    model.save('Results/model.h5')

    print(f'Evaluating the model using {len(data["xTest"])} samples...')
    loss, accuracy = model.evaluate(data['xTest'], data['yTest'], verbose=0)
    print(f'Loss: {loss:.4f}')
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()