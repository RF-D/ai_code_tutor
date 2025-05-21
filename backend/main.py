"""
Main entry point for the FastAPI application.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import practice, evaluation, assistance, languages

# Create FastAPI app
app = FastAPI(
    title="AI Code Tutor API",
    description="Backend API for the AI Code Tutor application",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(practice.router)
app.include_router(evaluation.router)
app.include_router(assistance.router)
app.include_router(languages.router)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)