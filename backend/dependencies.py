"""
Dependency injection and shared dependencies for the FastAPI app.
"""
from fastapi import Depends, HTTPException
from typing import Dict, List, Optional

# TODO: Implement dependencies such as:
# - get_llm_manager
# - get_language_service
# - get_code_execution_service