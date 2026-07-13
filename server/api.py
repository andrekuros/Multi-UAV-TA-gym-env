from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os
import sys

# Add parent directory to path to import mUAV_TA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mUAV_TA.DroneEnv import MultiUAVEnv

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
    return {"status": "Multi-UAV-TA Simulation Server Running"}

@app.websocket("/ws/simulation")
async def simulation_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = MultiUAVEnv()
    env.reset()
    
    try:
        while True:
            # For now, just step with random actions or idle actions
            # In a real setup, actions would come from the RL policy
            actions = {agent: 0 for agent in env.possible_agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            # Send state to frontend
            state_data = {
                "time_steps": env.time_steps,
                "agents": [
                    {
                        "name": agent.name,
                        "position": agent.position.tolist(),
                        "state": agent.state,
                    } for agent in env.agents_obj if agent.state != -1
                ],
                "tasks": [
                    {
                        "id": task.id,
                        "position": task.position.tolist(),
                        "status": task.status,
                    } for task in env.tasks if task.id != 0
                ]
            }
            
            await websocket.send_text(json.dumps(state_data))
            
            # Throttle the loop
            await asyncio.sleep(0.05)
            
            # Reset if done
            if all(terminations.values()) or all(truncations.values()):
                env.reset()
                
    except WebSocketDisconnect:
        print("Client disconnected from simulation")
    except Exception as e:
        print(f"Error in simulation loop: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
