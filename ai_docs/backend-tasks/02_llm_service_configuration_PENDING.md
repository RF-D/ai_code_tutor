# Backend Task: LLM Service Configuration

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Complete and enhance the LLM manager service to handle multiple model providers for code assistance

## Mid-Level Objective

- Finish implementation of the LLMManager class with proper error handling
- Create prompt templates for code assistance scenarios
- Implement provider-specific optimizations
- Add streaming support for real-time responses

## Implementation Notes
- All API keys should be loaded from environment variables
- Proper error handling with specific exceptions is required
- Include logging throughout the service
- Ensure efficient token usage with appropriate context windows
- Provider-specific configurations should be customizable
- Support streaming responses for real-time feedback

## Context

### Beginning context
- backend/services/llm_manager.py (partially implemented)
- backend/models/prompts.py (empty with TODOs)

### Ending context  
- Completed backend/services/llm_manager.py
- Implemented backend/models/prompts.py with prompt templates
- Added streaming support for real-time responses

## Low-Level Tasks
> Ordered from start to finish

1. Complete LLMManager class implementations

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Enhance the LLMManager class in llm_manager.py to add robust error handling, logging, and additional functionality.

File to UPDATE: backend/services/llm_manager.py

Add the following features to the existing LLMManager class:
- API key validation and error handling
- Retry mechanism for API failures
- Token usage tracking
- Context window optimization
- Temperature and other parameter customization
- Method to validate API keys without making full requests
- Logging for debugging and monitoring
- Proper error messages for users when models or providers fail

Ensure all methods have proper docstrings and type hints.

2. Implement prompt templates in prompts.py

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Implement prompt templates in the prompts.py file for different code assistance scenarios.

File to UPDATE: backend/models/prompts.py

Create a structured system for managing prompt templates, including:
- Base PromptTemplate class using Pydantic
- Specialized templates for:
  - Code generation
  - Code explanation
  - Bug fixing
  - Writing tests
  - Providing hints
  - Evaluating code submissions
  - Generating practice questions
  
Each template should have placeholders for context, language, and other variables.
Include example prompts that follow best practices for each supported LLM provider.
Add methods to format prompts based on the chosen provider and model.

3. Add streaming support to LLMManager

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Add streaming response support to the LLMManager for real-time feedback to users.

File to UPDATE: backend/services/llm_manager.py

Implement streaming support:
- Create a method for streaming responses from supported providers
- Handle streaming differences between providers (Anthropic, OpenAI, etc.)
- Add error handling specific to streaming contexts
- Implement backpressure and flow control
- Add timeout handling for long-running streams
- Create utility functions to process streaming chunks
- Ensure compatibility with FastAPI's streaming response

Make streaming configurable and provide fallback options for models that don't support it.

4. Implement LLM response processing utilities

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Implement utility functions in llm_manager.py to process and standardize responses from different LLM providers.

File to UPDATE: backend/services/llm_manager.py

Add utility functions for:
- Parsing and normalizing responses from different providers
- Extracting code blocks from responses
- Handling special tokens and formatting
- Converting between different response formats
- Sanitizing outputs for security
- Calculating token usage for billing or rate limiting
- Handling provider-specific features like JSON mode

Ensure all functions have proper error handling, type hints, and documentation.