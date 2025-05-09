from asyncio import subprocess
from typing import Optional
import os, uuid
import asyncio
import sys
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory setup
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

class WebcamManager:
    def __init__(self):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.current_mode: Optional[str] = None

    async def start_webcam_process(self, mode: str):
        if self.process:
            await self.stop_webcam_process()
        
        # Use sys.executable to ensure we use the same Python interpreter
        python_path = sys.executable
        script_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            "../vision-core/main.py"
        ))
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Webcam script not found at {script_path}")

        try:
            self.process = await asyncio.create_subprocess_exec(
                python_path, 
                script_path, 
                "--mode", mode,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self.current_mode = mode
            print(f"Started webcam process (PID: {self.process.pid}) in {mode} mode")
            return True
        except Exception as e:
            print(f"Error starting webcam process: {str(e)}")
            raise

    async def stop_webcam_process(self):
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
                print(f"Stopped webcam process (PID: {self.process.pid})")
            except ProcessLookupError:
                pass  # Process already terminated
            finally:
                self.process = None
                self.current_mode = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

webcam_manager = WebcamManager()
connection_manager = ConnectionManager()

@app.post("/start-tracking/{mode}")
async def start_tracking(mode: str):
    if mode not in ["blink", "gaze", "head"]:
        raise HTTPException(400, "Invalid tracking mode")
    
    try:
        await webcam_manager.start_webcam_process(mode)
        return {"status": "started", "mode": mode}
    except Exception as e:
        raise HTTPException(500, f"Failed to start tracking: {str(e)}")

@app.post("/stop-tracking")
async def stop_tracking():
    await webcam_manager.stop_webcam_process()
    return {"status": "stopped"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await connection_manager.broadcast(data)
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        print("Client disconnected")


@app.post("/upload")
async def upload_pdf(pdf: UploadFile = File(...)):
    if pdf.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are allowed")
    # give each upload a unique name
    filename = f"{uuid.uuid4()}.pdf"
    path = os.path.join(UPLOAD_DIR, filename)
    contents = await pdf.read()
    with open(path, "wb") as f:
        f.write(contents)
    return {"url": f"http://localhost:8000/uploads/{filename}"}
