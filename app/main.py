from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from .models import RecommendRequest, RecommendResponse
from .recommender import recommend, get_mood
from .embeddings import songs

app = FastAPI(title="AI Music Recommendation API")

# UI
app.mount("/static", StaticFiles(directory="ui"), name="static")

# 🎵 SONG FILES
app.mount("/songs", StaticFiles(directory="songs"), name="songs")


@app.get("/")
def root():
    return FileResponse(os.path.join("ui", "index.html"))


@app.get("/songs")
def get_songs():
    return {"songs": songs}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_songs(req: RecommendRequest):
    return {
        "selected_song": req.song,
        "mood": get_mood(req.song),
        "recommendations": recommend(req.song, req.threshold, req.mode),
    }
