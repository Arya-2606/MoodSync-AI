from pydantic import BaseModel
from typing import List

class RecommendRequest(BaseModel):
    song: str
    threshold: float = 0.6
    mode: str = "safe"

class SongResult(BaseModel):
    name: str
    similarity: float
    mood: str
    reason: str

class RecommendResponse(BaseModel):
    selected_song: str
    mood: str
    recommendations: List[SongResult]
