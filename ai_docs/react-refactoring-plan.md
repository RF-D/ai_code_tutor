# React Refactoring Plan for AI Code Tutor

This document outlines the plan for refactoring the Python Streamlit AI Code Tutor application into a React-based application with a Python FastAPI backend. The refactored application will support multiple programming languages beyond Python, including JavaScript, TypeScript, and React.

## Current Application Analysis

### Functionality Overview

The current application is a Python Learning Assistant with the following key features:

1. **Practice Question Generation**: Generates Python practice questions based on user-selected topics and skill levels.
2. **Code Evaluation**: Evaluates user-submitted Python code against practice questions.
3. **Solution Assistant**: Provides hints and assistance for solving problems through an interactive chat interface.
4. **Model Selection**: Allows users to select different AI models for code evaluation and question generation.

### Technology Stack

- **Frontend & Backend**: Streamlit (Python)
- **AI Integration**: Uses LangChain with various providers (Anthropic, OpenAI, Groq, Ollama, MistralAI)
- **Code Editor**: streamlit-ace for code input
- **State Management**: Streamlit session state

## Refactoring Approach

### New Technology Stack

- **Frontend**: React.js
- **Backend**: Python FastAPI 
- **API Communication**: REST API
- **State Management**: React Context API + useReducer (or Zustand)
- **Code Editor**: Monaco Editor for multi-language support
- **Styling**: CSS Modules or styled-components

## UI/UX Improvement

The refactored application will feature a more integrated, efficient interface inspired by modern coding platforms like boot.dev. Key improvements include:

### Integrated Learning Environment

- **Split-Screen Layout**: Code editor and assistant in the same view for seamless workflow
- **Contextual Help**: Assistance panel directly tied to current exercise
- **Real-time Feedback**: Immediate code validation alongside the editor

### User Interface Updates

1. **Main Layout**:
   - Left sidebar for navigation and language/model selection
   - Main code editor occupying approximately 60-70% of screen
   - Right panel for assistant/hints (collapsible)
   - Bottom panel for code execution results

2. **Workflow Improvements**:
   - Combined code evaluation and assistance in the same view
   - No page switching required between coding and getting help
   - Persistent practice question display while coding

3. **Responsive Design**:
   - Adaptive layout for different screen sizes
   - Collapsible panels for mobile optimization
   - Touch-friendly controls for tablet users

### Project Structure

```
ai_code_tutor/
├── backend/
│   ├── main.py                # FastAPI application entry point
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── practice.py        # Practice question endpoints
│   │   ├── evaluation.py      # Code evaluation endpoints
│   │   ├── assistance.py      # Hint & assistance endpoints
│   │   └── languages.py       # Language support endpoints
│   ├── services/
│   │   ├── llm_manager.py     # LLM service (migrated from utils)
│   │   ├── language_support.py # Language-specific processing
│   │   └── code_execution.py  # Sandboxed code execution service
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic models for request/response validation
│   │   ├── prompts.py         # Prompt templates
│   │   └── languages.py       # Language configurations and metadata
│   ├── dependencies.py        # Dependency injection
│   └── requirements.txt       # Backend dependencies
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── common/
│   │   │   │   ├── CodeEditor.jsx
│   │   │   │   ├── ModelSelector.jsx
│   │   │   │   ├── LanguageSelector.jsx # New component for language selection
│   │   │   │   └── Navigation.jsx
│   │   │   ├── PracticeQuestion/
│   │   │   │   ├── QuestionGenerator.jsx
│   │   │   │   └── TopicSelector.jsx
│   │   │   ├── CodePlayground/          # Combined editor & assistant
│   │   │   │   ├── PlaygroundLayout.jsx # Main container component
│   │   │   │   ├── CodePanel.jsx        # Editor & execution panel
│   │   │   │   ├── AssistantPanel.jsx   # Help & hints panel
│   │   │   │   ├── ResultsPanel.jsx     # Execution results
│   │   │   │   └── QuestionPanel.jsx    # Current question display
│   │   │   ├── CodeEvaluation/
│   │   │   │   ├── CodeSubmission.jsx
│   │   │   │   └── EvaluationResult.jsx
│   │   │   └── SolutionAssistant/
│   │   │       ├── HintChat.jsx
│   │   │       └── ChatInput.jsx
│   │   ├── context/
│   │   │   ├── AppContext.jsx
│   │   │   ├── QuestionContext.jsx
│   │   │   └── LanguageContext.jsx   # New context for language settings
│   │   ├── hooks/
│   │   │   ├── useApi.js      # Custom hook for API requests
│   │   │   ├── useQuestion.js # Custom hook for question state
│   │   │   └── useLanguage.js # Custom hook for language preferences
│   │   ├── services/
│   │   │   └── api.js         # API service for backend communication
│   │   ├── utils/
│   │   │   └── languageUtils.js # Helper functions for language operations
│   │   ├── layouts/
│   │   │   └── MainLayout.jsx # Container for app layout with panels
│   │   ├── App.jsx
│   │   ├── index.jsx
│   │   └── App.css
│   ├── package.json
│   └── README.md
└── README.md
```

## Core Functionality Mapping

### 1. Multi-Language Support

- **Backend**: FastAPI endpoint to provide language metadata and support
- **Frontend**: Language selector component and context
- **Data Flow**:
  - User selects programming language for practice
  - React stores selection in LanguageContext
  - All subsequent API calls include language preference
  - Practice questions, code evaluation, and hints are tailored to selected language

### 2. Practice Question Generation

- **Backend**: FastAPI endpoint to handle question generation requests for multiple languages
- **Frontend**: Question generation form with topic input, skill level selector, and language selector
- **Data Flow**:
  - User inputs topic, selects skill level and programming language
  - React sends request to FastAPI endpoint
  - FastAPI calls LLM service for language-specific question generation
  - Response returns to React for display

### 3. Integrated Code Playground

- **Backend**: 
  - FastAPI endpoints to handle code evaluation and execution
  - FastAPI endpoints for hint generation
  - Sandboxed execution environment for safely running user code
- **Frontend**: 
  - Integrated playground with Monaco Editor, results panel, and assistant panel
  - Side-by-side layout for code and assistance
- **Data Flow**:
  - User writes code in editor and can request execution or evaluation
  - User can ask for hints without leaving the coding environment
  - All interactions share the same language and question context
  - Results and assistance are displayed in adjacent panels
  - State is synchronized across all components

### 4. Model Selection

- **Backend**: FastAPI endpoint to retrieve available models and set active model
- **Frontend**: Model selector dropdowns in settings area
- **Data Flow**:
  - React requests available models from FastAPI endpoint
  - User selects models
  - Selection is stored in React state and sent to backend

## State Management Strategy

The application will use a combination of React Context API and useReducer for global state management:

```jsx
// Primary state structure
const initialState = {
  language: {
    current: "python",
    available: ["python", "javascript", "typescript", "react"],
    preferences: {
      // Language-specific settings
      python: { tabSize: 4, insertSpaces: true },
      javascript: { tabSize: 2, insertSpaces: true },
      typescript: { tabSize: 2, insertSpaces: true },
      react: { tabSize: 2, insertSpaces: true }
    }
  },
  models: {
    availableProviders: [],
    availableModels: {},
    codeEvalProvider: null,
    codeEvalModel: null,
    questionGenProvider: null,
    questionGenModel: null
  },
  practiceQuestion: {
    current: null,
    topic: "",
    skillLevel: "Beginner",
    topicSuggestions: []
  },
  codeEvaluation: {
    code: "",
    result: null,
    loading: false
  },
  assistant: {
    messages: [],
    loading: false
  },
  ui: {
    activeView: "playground",
    sidebarCollapsed: false,
    assistantPanelWidth: 30, // percentage
    showResults: true
  }
};
```

## API Endpoints

### Language Support

- `GET /api/languages` - Get available programming languages
- `GET /api/languages/{language}/topics` - Get topic suggestions for a specific language

### Practice Questions

- `GET /api/topics/suggestions` - Get topic suggestions (with optional language parameter)
- `POST /api/questions/generate` - Generate a practice question

### Code Evaluation

- `POST /api/code/evaluate` - Evaluate submitted code
- `POST /api/code/execute` - Execute code and return result

### Solution Assistant

- `POST /api/assistance/hint` - Get hint for current question

### Model Management

- `GET /api/models` - Get available models and providers
- `POST /api/models/set` - Set active models

## Integrated Code Playground UI

The new integrated code playground will feature:

1. **Question Display**:
   - Clearly visible practice question at the top of the view
   - Collapsible question details for more editor space

2. **Code Editor**:
   - Monaco Editor with language-specific syntax highlighting
   - Run/Evaluate button
   - Language selector
   - Settings for editor preferences

3. **Results Panel**:
   - Execution output
   - Evaluation feedback
   - Error messages
   - Performance metrics (time, memory usage)

4. **Assistant Panel**:
   - Chat-like interface for asking questions
   - Context-aware hints based on current code and question
   - Code suggestions that can be applied directly
   - Option to request full solutions with explanations

5. **Layout Controls**:
   - Adjustable panel sizes (drag handles)
   - Toggle buttons to show/hide panels
   - Full-screen mode for focused coding

This integrated approach streamlines the learning process by providing immediate access to help resources while coding, similar to the boot.dev learning environment.

## Monaco Editor Integration

Monaco Editor will be used as the primary code editor component due to its robust multi-language support:

1. **Language Configuration**:
   - Configure Monaco Editor to support Python, JavaScript, TypeScript, and React (JSX)
   - Set up language-specific settings (tab size, formatting rules)
   - Implement language switching that preserves editor state

2. **Editor Features**:
   - Syntax highlighting for all supported languages
   - Code completion (rich for JavaScript/TypeScript, basic for others)
   - Error highlighting and linting integration
   - Code folding and navigation
   - Theme support (light/dark modes)

3. **Language-Specific Tools**:
   - Format code button (using language-appropriate formatters)
   - Run code functionality with language-specific execution
   - Language-specific code snippets

## Migration Strategy

1. **Phase 1: Backend Development**
   - Create FastAPI application structure
   - Implement Pydantic models for request/response validation
   - Develop language support infrastructure
   - Implement code execution services for multiple languages
   - Migrate LLM manager from Streamlit to FastAPI
   - Implement API endpoints
   - Test API functionality independent of frontend
   - Generate OpenAPI documentation (automatic with FastAPI)

2. **Phase 2: Frontend Scaffolding**
   - Set up React project structure
   - Create component hierarchy
   - Implement state management including language context
   - Configure Monaco Editor with multi-language support
   - Build API service layer

3. **Phase 3: Feature Implementation**
   - Implement integrated code playground layout
   - Implement language selector and preferences
   - Implement question generation with language support
   - Implement code editor with language-specific features
   - Implement code evaluation with multi-language execution
   - Implement solution assistant in same-page view
   - Implement model selection

4. **Phase 4: Testing & Refinement**
   - End-to-end testing across all supported languages
   - UI/UX refinement
   - Performance optimization

## Implementation Considerations

### Language Support Strategy

The application will use a phased approach to language support:

1. **Phase 1 Languages** (Initial release):
   - Python
   - JavaScript

2. **Phase 2 Languages** (First expansion):
   - TypeScript
   - React/JSX

3. **Phase 3 Languages** (Future expansion):
   - Java
   - C/C++
   - Go
   - Other languages based on user demand

### Code Execution Security

For running user-submitted code:

1. **Sandboxed Environments**:
   - Python code: Use restricted Python interpreter
   - JavaScript/TypeScript: Use Node.js sandbox
   - React: Use browser sandbox with iframe isolation

2. **Resource Limitations**:
   - Time limits (maximum execution time)
   - Memory limits
   - Process/thread limits
   - Network access restrictions

3. **Input Validation**:
   - Code scanning for potentially harmful patterns
   - API call restrictions
   - File system access control

### FastAPI Advantages

FastAPI offers several advantages that will benefit this multi-language project:

1. **Better Performance**: FastAPI's asynchronous capabilities enable handling more concurrent requests, which is valuable for handling multiple users querying the LLM services.

2. **Automatic Documentation**: FastAPI provides automatic interactive API documentation via Swagger UI and ReDoc, making it easier to test and document our API endpoints.

3. **Type Safety and Validation**: FastAPI uses Pydantic for request/response validation, ensuring data integrity and reducing bugs.

4. **Asynchronous Support**: Native support for async/await syntax allows for non-blocking operations, important for handling LLM requests that may take time.

5. **Dependency Injection**: Built-in dependency injection system simplifies code organization and testing.

### Monaco Editor Benefits for Multi-Language Support

Monaco Editor is particularly well-suited for this project because:

1. **Rich Language Support**: Monaco natively supports all target languages with syntax highlighting
2. **Language Intelligence**: Rich IntelliSense for JavaScript, TypeScript, and basic support for other languages
3. **Customizability**: Configurable for language-specific settings
4. **Performance**: Handles large files and complex language features efficiently
5. **Familiar Experience**: Provides a VS Code-like experience familiar to many developers

### Authentication

The current application doesn't have authentication. If user accounts are needed in the future, implement:
- JWT-based authentication (FastAPI provides built-in security utilities)
- User profile storage
- Progress tracking per user
- Language preferences per user

### Deployment

- Package the React frontend as static files
- Serve the static files from FastAPI using StaticFiles middleware
- Use Uvicorn for production deployment (recommended ASGI server for FastAPI)
- Consider containerization with Docker
- Set up language-specific execution environments with proper isolation

## Next Steps

1. Set up FastAPI backend with basic endpoints and Pydantic models
2. Create language support infrastructure
3. Configure Monaco Editor for multi-language support
4. Create React application with integrated code playground layout
5. Implement API communication layer
6. Migrate core functionality sequentially
7. Add support for additional languages incrementally