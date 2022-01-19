import numpy as np
from numpy.lib import stride_tricks
import os
from PIL import Image, ImageEnhance
import scipy.io.wavfile as wav
import random
import math
import base64
from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
from keras.utils import np_utils
from keras.models import load_model
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import wave
import cv2
import io
import json
# pre processing steps


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    samples = np.append(sig, np.zeros((frameSize//2), dtype=int))
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    samples = np.append(samples, np.zeros(frameSize, dtype=int))
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                               samples.strides[0])).copy()
    frames = frames.astype('float64')
    frames *= win
    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(
                spec[:, int(scale[i]):int(scale[i+1])], axis=1)
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs


def stft_matrix(audiopath, binsize=2**10, png_name='tmp.png',
                save_png=False, offset=0):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    ims = np.transpose(ims)
    ims = np.flipud(ims)  # weird - not sure why it needs flipping
    return ims


def get_cropped_samples(matrix, crop_width=125):
    clipped_mat = matrix[:, (matrix.shape[1] % crop_width):]
    n_splits = clipped_mat.shape[1] / crop_width
    samples = np.split(clipped_mat, n_splits, axis=1)
    return samples


def preprocess(data):
    data = np.array(data)
    print(type(data))
    data = data.astype('float32')
    data = np.array([(X - X.min()) / (X.max() - X.min()) for X in data])
    return data


def prep_train_test(data):
    data = preprocess(data)
    data = np_utils.to_categorical(data)
    return data


def keras_img_prep(data, img_rows, img_cols):
    data = data.reshape(2*data.shape[0], img_rows, img_cols, 1)
    return data


def create_sample_dictionary(audio_path):
    data = stft_matrix(audio_path)
    data_samples = get_cropped_samples(data)
    data_samples = prep_train_test(data_samples)
    print(np.shape(data_samples))
    img_rows, img_cols = data_samples.shape[1], data_samples.shape[2]
    data_samples = keras_img_prep(data_samples, img_rows, img_cols)
    return data_samples


def remove_silence(filename, smoothing=1.0, weight=0.3, plot=False):
    # create participant directory for segmented wav files
    [Fs, x] = aIO.read_audio_file(filename)
    segments = aS.silence_removal(x, Fs, 0.020, 0.020,
                                  smooth_window=smoothing,
                                  weight=weight,
                                  plot=plot)
    data = []
    for s in segments:
        seg_name = "segment_{:.2f}-{:.2f}.wav".format(s[0], s[1])
        wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])
    files = os.listdir(os.curdir)
    for file in files:
        if file.endswith('.wav'):
            w = wave.open(file, 'rb')
            data.append([w.getparams(), w.readframes(w.getnframes())])
            w.close()
            os.remove(file)
    # print(filename)
    output = wave.open(filename, 'wb')
    output.setparams(data[0][0])
    for idx in range(len(data)):
        output.writeframes(data[idx][1])
    output.close()


def get_face(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=4,
    )
    print(faces)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        face = img[y:y + h, x:x + w]
        cv2.imwrite(r'face_'+path, face)


app = Flask(__name__)
CORS(app)

model = load_model(
    r'D:\FYP\FYP-Code\emily\model\model-dep.h5', compile=True)
model_fer = load_model(
    r'D:\FYP\FYP-Code\emily\model\model-fer-0.6605.h5', compile=True)


@app.route('/classify_fer', methods=["POST"])
def index_fer():
    # print(request.files)
    recieved = json.loads(request.data)
    recieved = recieved['data']
    # recieved = request.files
    rmStr = "data:image/jpeg;base64,"
    imgdata = base64.b64decode(recieved[len(rmStr):])
    image = Image.open(io.BytesIO(imgdata))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(2)
    image.save('person.jpeg')
    # print(recieved)
    # recieved.save(r'./img/'+recieved.filename)
    # file = base64.b64decode(imgdata)
    # print(type(imgdata))
    # print(type(recieved))
    # img = Image.open(io.BytesIO(
    #     base64.decodebytes(bytes(imgdata))))
    # img.save('person.jpeg')
    # with open('person.png', 'wb') as f:
    #     f.write(imgdata)
    get_face('person.jpeg')
    img = cv2.imread('face_person.jpeg')
    img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape((1, 48, 48, 1))
    # print(np.shape(img))
    # print(img)
    prediction = model_fer.predict(img)
    label = prediction.argmax(axis=-1).tolist()
    # print(prediction)
    exp = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    pred = prediction[0][label][0]
    data = {
        'success': True,
        'label': label,
        'exp': exp[label[0]],
        'accuracy': float(pred),
    }
    print(data)
    return jsonify(data)


@ app.route('/classify_dep', methods=["POST"])
def index_dep():
    # print(request.files)
    # recieved = request.files['file']
    # recieved.save(r'./'+recieved.filename)
    rmStr = 'data:audio/wav;base64,'
    recieved = json.loads(request.data)
    # print(recieved)
    recieved = recieved['data']
    # recieved = request.files
    audiodata = base64.b64decode(recieved[len(rmStr):])
    with open('audio.wav', 'wb') as f:
        f.write(audiodata)
    remove_silence(r'./'+'audio.wav')
    data = create_sample_dictionary(
        r'./'+'audio.wav')
    prediction = model.predict(data)
    labels = prediction.argmax(axis=-1).tolist()
    # print(prediction)
    dep = labels.count(1)
    nodep = labels.count(0)
    # print(labels)
    print(f'Depressed Count is: {dep}')
    print(f'Non-Depressed Count is: {nodep}')
    return jsonify({
        'success': True,
        'label': labels,
        'ratio': nodep/dep
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
