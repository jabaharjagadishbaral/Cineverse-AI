from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import model

app = FastAPI(title="CineVerse AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
S = {}

@app.on_event("startup")
def load(): S["m"] = model.train()

class Req(BaseModel):
    user_id: int | None = None
    genres: list[str] = []
    k: int = 8
    debias: float = .5

@app.post("/recommend")
def recommend(r: Req):
    if r.user_id is None and not r.genres:
        raise HTTPException(422, "Send a user_id, or pick at least one genre for a new user")
    return model.recommend(*S["m"], k=r.k, user=r.user_id, genres=r.genres, debias=r.debias)

@app.get("/health")
def health(): return {"ok": "m" in S}
