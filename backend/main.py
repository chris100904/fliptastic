import os, uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow your React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               # you can lock this down in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# Where we’ll save uploads
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve files in /uploads at http://<host>/uploads/…
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

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