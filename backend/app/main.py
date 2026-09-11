from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.neo4j_client import close_driver
from app.routers import graph, master_data, runs

app = FastAPI(title="GraphRAG Thesis Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(master_data.router)
app.include_router(graph.router)
app.include_router(runs.router)


@app.on_event("shutdown")
def shutdown_event():
    close_driver()


@app.get("/api/health")
def health():
    return {"status": "ok"}
