from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

# Serve maps
app.mount(
    "/maps",
    StaticFiles(directory=BASE_DIR / "src" / "visualization" / "maps"),
    name="maps"
)

# Serve frontend
app.mount(
    "/ui",
    StaticFiles(directory=BASE_DIR / "frontend", html=True),
    name="ui"
)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/api/maps")
def list_maps():
    maps_dir = BASE_DIR / "src" / "visualization" / "maps"
    return sorted([f.name for f in maps_dir.glob("*.html")])