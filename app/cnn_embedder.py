import librosa
import numpy as np
from keras.models import load_model

MODEL_PATH = "trained_model.h5"
SR = 22050
SEGMENT_DURATION = 10
N_MELS = 128

model = load_model(MODEL_PATH)


def get_cnn_embedding(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    seg_len = SR * SEGMENT_DURATION

    embeddings = []

    for start in range(0, len(y), seg_len):
        seg = y[start : start + seg_len]
        if len(seg) < seg_len:
            continue

        mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=N_MELS)
        mel = librosa.power_to_db(mel)
        mel = mel[:, :128]

        if mel.shape[1] < 128:
            mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))

        mel = mel[np.newaxis, ..., np.newaxis]
        emb = model.predict(mel, verbose=0)[0]
        embeddings.append(emb)

    if not embeddings:
        return np.zeros(model.output_shape[-1])

    emb = np.mean(embeddings, axis=0)
    return emb / np.linalg.norm(emb)
