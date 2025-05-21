# API Integration
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Implement comprehensive API integration between React frontend and FastAPI backend

## Mid-Level Objective

- Create a robust API client with proper request handling
- Implement API endpoints for practice questions
- Set up code evaluation API integration
- Create solution assistance (hint) API integration
- Implement language support API features
- Add model selection API functionality

## Implementation Notes
- Use fetch or axios for API requests
- Implement proper error handling and retries
- Add request/response interceptors for common functionality
- Set up authentication if needed
- Ensure proper TypeScript typings for API responses

## Context

### Beginning context
- frontend/src/services/api.js (basic structure from previous task)
- Backend endpoints as defined in the refactoring plan

### Ending context  
- Complete frontend/src/services/api.js with all endpoint implementations
- frontend/src/services/apiTypes.js for TypeScript interfaces
- frontend/src/hooks/useApi.js with enhanced functionality
- frontend/src/services/mockData.js for development without backend

## Low-Level Tasks
> Ordered from start to finish

1. Create API type definitions
```aider
Create frontend/src/services/apiTypes.js containing:
- TypeScript interfaces for all API requests and responses
- Type definitions matching backend Pydantic models
- Enums for valid values (languages, difficulty levels, etc.)
- Error response types
- Utility types for common patterns
```

2. Implement practice question API endpoints
```aider
Update frontend/src/services/api.js to add:
- Functions for fetching topic suggestions
- API call for generating practice questions
- Methods to fetch questions by language, difficulty, etc.
- Proper error handling and response parsing
- Loading state management
```

3. Implement code evaluation API endpoints
```aider
Update frontend/src/services/api.js to add:
- Functions for submitting code for evaluation
- API calls for code execution
- Methods to handle different language submissions
- Result formatting and error handling
- Performance metric collection
```

4. Implement solution assistance API endpoints
```aider
Update frontend/src/services/api.js to add:
- Functions for requesting hints and assistance
- API calls for submitting user questions
- Methods to handle chat history and context
- Response formatting for different hint types
- Error handling for LLM-related issues
```

5. Implement language support API endpoints
```aider
Update frontend/src/services/api.js to add:
- Functions for fetching available languages
- API calls for language-specific metadata
- Methods to get language-specific topics and settings
- Support for user language preferences
```

6. Implement model selection API endpoints
```aider
Update frontend/src/services/api.js to add:
- Functions for fetching available models
- API calls for setting active models
- Methods to get model performance metrics
- Support for model-specific configuration
```

7. Create development mock data
```aider
Create frontend/src/services/mockData.js with:
- Mock data for all API endpoints
- Realistic sample responses matching API types
- Configurable delay for testing loading states
- Controllable error states for testing
- Documentation on usage patterns
```

8. Enhance useApi hook with additional functionality
```aider
Update frontend/src/hooks/useApi.js to:
- Provide more convenient access patterns for components
- Add caching functionality where appropriate
- Implement pagination helpers for list endpoints
- Add polling for long-running operations
- Include status tracking across multiple components
```