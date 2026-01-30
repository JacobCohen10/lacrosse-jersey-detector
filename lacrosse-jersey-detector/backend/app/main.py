"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import API_PREFIX, CORS_ORIGINS

app = FastAPI(
    title="Lacrosse Jersey Detector API",
    description="API for analyzing lacrosse game footage to detect jersey numbers",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix=API_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Lacrosse Jersey Detector API",
        "docs": "/docs",
        "health": f"{API_PREFIX}/health"
    }
