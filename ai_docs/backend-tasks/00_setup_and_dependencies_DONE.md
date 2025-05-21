# Backend Task: Setup and Dependencies

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Setup the basic FastAPI backend environment and dependencies needed for the AI Code Tutor application

## Mid-Level Objective

- Configure proper environment variables for API keys and sensitive data
- Implement dependency injection for service components
- Setup robust error handling and logging
- Complete and validate requirements.txt with all necessary dependencies

## Implementation Notes
- Follow type hints for all function parameters and return types
- Use proper error handling with specific exceptions
- Environment variables should be loaded securely
- Dependency injection pattern should be used for service components
- Follow PEP 8 style guidelines and use snake_case for functions/variables 
- All sensitive information like API keys should be loaded from environment variables

## Context

### Beginning context
- backend/dependencies.py (with TODOs)
- backend/requirements.txt (basic dependencies)
- backend/main.py (basic FastAPI setup)

### Ending context  
- Updated backend/dependencies.py (with implemented dependency functions)
- backend/.env.example (template for environment variables)
- Updated backend/requirements.txt (with additional dependencies)
- Updated error handling in main.py

## Low-Level Tasks
> Ordered from start to finish

1. Create .env.example template file

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Create a .env.example file that defines all the environment variables needed for the application, including API keys for different LLM providers, execution environment settings, and other configuration settings.

File to CREATE: backend/.env.example

The file should include template environment variables for:
- API keys for OpenAI, Anthropic, Groq, and MistralAI
- Debug/production mode flag
- Timeouts for code execution
- Memory limits for code execution
- CORS settings
- Optional: Configuration for database connections if needed in the future

Each variable should have a clear comment explaining its purpose and format.

2. Update requirements.txt with all necessary dependencies

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Update the requirements.txt file to include all necessary dependencies for the AI Code Tutor backend.

File to UPDATE: backend/requirements.txt

Please add:
- Additional dependencies for secure environment variable handling 
- Dependencies for code execution and sandboxing
- Dependencies for testing
- Logging libraries
- Any other dependencies needed based on the existing codebase

Ensure all dependencies have version specifications for reproducibility.

3. Implement dependency injection in dependencies.py

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Implement the dependency injection functions in the dependencies.py file to provide services throughout the application.

File to UPDATE: backend/dependencies.py

Functions to CREATE:
- get_llm_manager: Returns an instance of LLMManager for handling LLM interactions
- get_language_service: Returns the language support service
- get_code_execution_service: Returns the code execution service with proper sandboxing

Include proper error handling, logging, and caching mechanisms where appropriate. Ensure all functions have proper type hints and docstrings.

4. Update main.py with custom exception handlers

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Update the main.py file to add comprehensive exception handling for the FastAPI application.

File to UPDATE: backend/main.py

Add:
- Custom exception handlers for common errors
- Middleware for request logging
- Structured error responses with consistent formatting
- Request ID tracking for debugging
- Development vs. production error detail handling

Ensure the exception handlers provide useful information for debugging while not exposing sensitive information in production.