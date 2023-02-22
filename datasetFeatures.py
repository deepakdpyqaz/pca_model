import os
import pathlib
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool
from spafe.features.gfcc import gfcc

# here we have extracted the features

DATA_DIR = "ravdess"        # here we have the dataset
EMOTION_MAP = {
    "anger": "anger",
    "boredom": "boredom",
    "disgust": "disgust",
    "fearful": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprised",
    "calm": "calm"
}


def process_frame(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    flux = librosa.onset.onset_strength(y=y, sr=sr)  
    roll_off = librosa.feature.spectral_rolloff(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fundamentalFreq = librosa.yin(y=y, sr=sr, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'))
    gfccs = np.mean(gfcc(y, fs=sr, num_ceps=13), axis=0, dtype="float32")
    return np.hstack(
        (
            mfcc.mean(axis=1),
            rms.mean(axis=1),
            zero_crossing_rate.mean(axis=1),
            flux.mean(axis=0),
            roll_off.mean(axis=1),
            centroid.mean(axis=1),
            bandwidth.mean(axis=1),
            chroma.mean(axis=1),
            fundamentalFreq.mean(axis=0),
            gfccs,
        )
    )


def process_audio(fname, emotionName):
    y, sr = librosa.load(fname)
    fname = os.path.basename(fname)
    y = librosa.effects.preemphasis(y)
    emotions = np.array([emotionName])
    result = process_frame(y, sr)
    return np.hstack((emotions, result))


if __name__ == "__main__":
    dic = {}
    for emotion in EMOTION_MAP:
        dic[emotion] = []
        for fname in pathlib.Path(f"{DATA_DIR}/{emotion}").glob("*.wav"):
            dic[emotion].append(fname)

    print("Number of files per emotion:")
    for emotion in EMOTION_MAP:
        print(f"{emotion}: {len(dic[emotion])}")

    print("Extracting features...")
    with Pool(6) as p:
        results = p.starmap(process_audio, [(fname, EMOTION_MAP[emotion]) for emotion in EMOTION_MAP for fname in dic[emotion]])
    print("Saving features...")
    df = pd.DataFrame(results)
    df.to_csv("features.csv", index=False)
    print("Done.")
