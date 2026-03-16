from .embeddings import (
    songs,
    song_to_index,
    cnn_embeddings,
    mfcc_embeddings,
    chroma_embeddings,
    rhythm_embeddings,
    energy_embeddings,
)
from .utils import cosine_sim


def get_mood(song):
    s = song.lower()
    if any(k in s for k in ["love", "sad", "venmegam"]):
        return "sad / melodic"
    if any(k in s for k in ["mass", "dance", "party"]):
        return "energetic"
    return "neutral"


def recommend(song, threshold, mode):
    if song not in song_to_index:
        return []

    idx = song_to_index[song]
    results = []

    for i in range(len(songs)):
        if i == idx:
            continue

        cnn_sim = cosine_sim(cnn_embeddings[idx], cnn_embeddings[i])
        mfcc_sim = cosine_sim(mfcc_embeddings[idx], mfcc_embeddings[i])
        chroma_sim = cosine_sim(chroma_embeddings[idx], chroma_embeddings[i])
        rhythm_sim = cosine_sim(rhythm_embeddings[idx], rhythm_embeddings[i])
        energy_sim = cosine_sim(energy_embeddings[idx], energy_embeddings[i])

        # 🔥 Clean weighted fusion (CNN dominant)
        raw_sim = (
            0.45 * cnn_sim
            + 0.25 * mfcc_sim
            + 0.10 * chroma_sim
            + 0.05 * rhythm_sim
            + 0.05 * energy_sim
        )

        # Rescale [-1,1] → [0,1]
        final_sim = (raw_sim + 1) / 2

        # Mood-aware boost
        if get_mood(songs[i]) == get_mood(song):
            final_sim += 0.05

        final_sim = min(final_sim, 1.0)

        results.append(
            {
                "name": songs[i],
                "similarity": round(final_sim, 3),
                "mood": get_mood(songs[i]),
                "reason": "CNN + MFCC + Chroma + Rhythm + Energy",
            }
        )

    # Rank
    results.sort(key=lambda x: x["similarity"], reverse=True)

    good = [r for r in results if r["similarity"] >= threshold]
    return good[:10] if good else results[:5]
