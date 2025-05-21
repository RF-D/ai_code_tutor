# Python Learning Assistant

An AI-powered tool to help users learn Python through practice questions, code evaluation, and interactive chat for hints.

## Features

- Generate practice questions based on Python topics
- Evaluate user-submitted code
- Provide hints and assistance for solving problems

## Streamlit Frontend Setup

1. Clone the repository
2. Create a Conda environment:
   ```
   conda create -n python-learning-assistant
   conda activate python-learning-assistant
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Set up environment variables (see `.env.example`)
5. Run the app: `streamlit run main.py`

## Frontend

The project includes a React frontend located in the `frontend/` directory. See
[`frontend/README.md`](frontend/README.md) for setup and development
instructions.

## FastAPI Backend Setup

The project includes a FastAPI backend for the new React frontend. To run the backend:

1. Install backend dependencies:
   ```
   pip install -r backend/requirements.txt
   ```

2. Start the FastAPI server:
   ```
   uvicorn backend.main:app --reload
   ```

3. The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)

4. API documentation can be accessed at:
   - Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)