# Frontend Context Setup
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Set up a global context and state management for the React frontend

## Mid-Level Objective

- Create AppContext for managing global state
- Implement Language Context for language selection and preferences
- Implement Question Context for managing practice questions
- Set up API client for backend communication

## Implementation Notes
- Use React's Context API for state management
- Follow the structure from react-refactoring-plan.md
- Implement proper TypeScript/JSX typing
- Organize context providers in a hierarchy
- Provide useful hooks for accessing context

## Context

### Beginning context
- frontend/src/context/.gitkeep
- frontend/src/App.jsx (minimal)
- frontend/src/index.jsx

### Ending context  
- frontend/src/context/AppContext.jsx
- frontend/src/context/LanguageContext.jsx
- frontend/src/context/QuestionContext.jsx
- frontend/src/services/api.js
- Updated frontend/src/App.jsx to include context providers
- frontend/src/hooks/useApi.js
- frontend/src/hooks/useLanguage.js
- frontend/src/hooks/useQuestion.js

## Low-Level Tasks
> Ordered from start to finish

1. Create AppContext.jsx for managing global state
```aider
Create a new file frontend/src/context/AppContext.jsx that implements a global context provider. This should:
- Define initial state matching the structure from react-refactoring-plan.md
- Create a reducer with appropriate actions
- Implement the context provider component
- Export useAppContext hook for accessing context
- Handle loading and error states
```

2. Create LanguageContext.jsx for language selection
```aider
Create frontend/src/context/LanguageContext.jsx that:
- Provides language selection and preferences
- Contains available languages (initially Python and JavaScript)
- Includes language-specific settings (tabSize, etc.)
- Implements functions to change the current language
- Exports a useLanguage hook for easy access
```

3. Create QuestionContext.jsx for practice questions
```aider
Create frontend/src/context/QuestionContext.jsx that:
- Manages the current practice question state
- Provides functionality to fetch and store questions
- Handles question topics and difficulty levels
- Tracks user progress with questions
- Exports a useQuestion hook for components to use
```

4. Create API service for backend communication
```aider
Create frontend/src/services/api.js that:
- Implements functions to call the FastAPI backend
- Handles authentication if needed
- Provides methods for each API endpoint (questions, code evaluation, etc.)
- Implements proper error handling and retries
- Uses modern fetch or axios patterns
```

5. Create custom hooks for context access
```aider
Create frontend/src/hooks/useApi.js, useLanguage.js, and useQuestion.js that:
- Provide convenient access to respective contexts
- Add additional helper functions as needed
- Include proper error handling
- Export typed interfaces for component usage
```

6. Update App.jsx to incorporate context providers
```aider
Update frontend/src/App.jsx to:
- Import all context providers
- Wrap the application in providers
- Set up nesting order correctly
- Add any global error boundaries
- Keep the UI structure minimal for now
```