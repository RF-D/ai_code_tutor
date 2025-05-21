"""
Language support router module.
"""
from fastapi import APIRouter

router = APIRouter(
    prefix="/api/languages",
    tags=["languages"],
    responses={404: {"description": "Not found"}},
)

# TODO: Implement language support endpoints