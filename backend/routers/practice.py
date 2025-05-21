"""
Practice question router module.
"""
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/questions",
    tags=["practice"],
    responses={404: {"description": "Not found"}},
)

# TODO: Implement practice question generation endpoints