"""
Code evaluation router module.
"""
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/code",
    tags=["evaluation"],
    responses={404: {"description": "Not found"}},
)

# TODO: Implement code evaluation endpoints