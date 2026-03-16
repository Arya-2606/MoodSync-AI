import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from .cnn_embedder import get_cnn_embedding

SONGS_FOLDER = r"F:\duhacks\songs"
SR = 22050
DURATION = 30

songs = []
cnn_embeddings = []
mfcc_embeddings = []
chroma_embeddings = []
rhythm_embeddings = []
energy_embeddings = []


def extract_librosa(path):
    y, sr = librosa.load(path, sr=SR, mono=True, duration=DURATION)
    y = librosa.util.normalize(y)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    tempo = librosa.beat.tempo(y=y, sr=sr)[0]
    rms = np.mean(librosa.feature.rms(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return mfcc, chroma, np.array([tempo]), np.array([rms, centroid])


for file in os.listdir(SONGS_FOLDER):
    if file.lower().endswith((".mp3", ".wav")):
        path = os.path.join(SONGS_FOLDER, file)
        try:
            songs.append(file)

            cnn_embeddings.append(get_cnn_embedding(path))
            mfcc, chroma, rhythm, energy = extract_librosa(path)

            mfcc_embeddings.append(mfcc)
            chroma_embeddings.append(chroma)
            rhythm_embeddings.append(rhythm)
            energy_embeddings.append(energy)

        except Exception as e:
            print(f"❌ Failed on {file}: {e}")


def scale(x):
    x = StandardScaler().fit_transform(x)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


cnn_embeddings = np.array(cnn_embeddings)
mfcc_embeddings = scale(np.array(mfcc_embeddings))
chroma_embeddings = scale(np.array(chroma_embeddings))
rhythm_embeddings = scale(np.array(rhythm_embeddings))
energy_embeddings = scale(np.array(energy_embeddings))

song_to_index = {s: i for i, s in enumerate(songs)}

print("✅ Songs:", len(songs))
print("✅ CNN:", cnn_embeddings.shape)
print("✅ MFCC:", mfcc_embeddings.shape)
