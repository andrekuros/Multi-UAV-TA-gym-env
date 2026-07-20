from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path to import mUAV_TA
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

REPLAY_CANDIDATES = [
    ROOT / "experiments" / "results" / "wps_escort_replay.json",
    ROOT / "experiments" / "results" / "wps_commit_replay.json",
]


def resolve_replay_path() -> Path:
    env_path = os.environ.get("UAV_REPLAY_PATH")
    if env_path:
        return Path(env_path)
    for path in REPLAY_CANDIDATES:
        if path.exists():
            return path
    return REPLAY_CANDIDATES[0]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "Multi-UAV-TA Simulation Server Running",
        "replay": "/api/replay",
        "download": "/api/replay/download",
        "replay_file": str(resolve_replay_path()),
    }


def load_replay():
    replay_path = resolve_replay_path()
    if not replay_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Replay not generated. Run: "
                "python experiments/generate_simulation_replay.py --scenario WPS_escort"
            ),
        )
    with replay_path.open(encoding="utf-8") as replay_file:
        return json.load(replay_file)


@app.get("/api/replay")
def get_replay():
    """Return the deterministic WPS_commit log for browser-side playback."""
    return load_replay()


@app.get("/api/replay/download")
def download_replay():
    replay_path = resolve_replay_path()
    if not replay_path.exists():
        load_replay()
    return FileResponse(
        replay_path,
        media_type="application/json",
        filename=replay_path.name,
    )

@app.websocket("/ws/simulation")
async def simulation_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        replay = load_replay()
        while True:
            for frame in replay["frames"]:
                await websocket.send_text(json.dumps(frame))
                await asyncio.sleep(0.12)
    except WebSocketDisconnect:
        print("Client disconnected from simulation")
    except Exception as e:
        print(f"Error in simulation loop: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
