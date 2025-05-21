# Backend Task: Language Models Implementation

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Complete the language models implementation to support various programming languages in the AI Code Tutor

## Mid-Level Objective

- Define comprehensive language configuration models
- Implement editor settings for different programming languages
- Create execution environment configurations for each supported language
- Setup language features detection and support

## Implementation Notes
- Language configurations should be extensible for easy addition of new languages
- Each language should have appropriate syntax highlighting, linting, and execution settings
- Execution environments should be secure and isolated
- Type annotations should be used throughout
- Follow existing code style with snake_case for functions/variables
- Editor settings should be compatible with Monaco editor used in the frontend

## Context

### Beginning context
- backend/models/languages.py (with TODOs)
- backend/models/schemas.py (with TODOs)
- backend/services/language_support.py (empty)

### Ending context  
- Completed backend/models/languages.py (with language configuration models)
- Updated backend/models/schemas.py (with relevant schemas)
- Implemented backend/services/language_support.py (with language support functions)

## Low-Level Tasks
> Ordered from start to finish

1. Implement language configuration models in languages.py

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Implement the language configuration models in the languages.py file to represent different programming languages supported by the AI Code Tutor.

File to UPDATE: backend/models/languages.py

Create the following Pydantic models:
- LanguageFeatures: Defines features of a programming language (typing system, paradigms, etc.)
- EditorSettings: Configuration for Monaco editor (indentation, autocompletion, etc.)
- ExecutionEnvironment: Settings for code execution (command, arguments, timeout)
- LanguageConfig: Main model that combines all the above for a specific language

Each model should have appropriate fields, validation, and documentation.
Include at least basic configurations for Python, JavaScript, and Java as examples.

2. Update schemas.py with language-related request/response models

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Update the schemas.py file to include language-related request and response models for API endpoints.

File to UPDATE: backend/models/schemas.py

Create the following Pydantic models:
- LanguageInfoRequest: Request for information about a specific language
- LanguageInfoResponse: Response with language configuration details
- SupportedLanguagesResponse: List of supported languages with basic metadata
- LanguageFeatureRequest: Request to check if a language supports specific features

Each model should have proper validation, documentation, and examples.
Follow the same style as other schemas in the file.

3. Implement language support service

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Implement the language_support.py service to provide functions for working with programming languages.

File to UPDATE: backend/services/language_support.py

Create the following functions:
- get_supported_languages(): Returns a list of all supported languages
- get_language_config(language_id): Returns configuration for a specific language
- is_feature_supported(language_id, feature): Checks if a language supports a feature
- get_editor_settings(language_id): Returns editor settings for a specific language
- get_execution_environment(language_id): Returns execution environment for a language

Include proper error handling, caching for performance, and comprehensive language detection.
Add docstrings and type hints to all functions.

4. Implement predefined languages configurations

What prompt would you run to complete this task?
What file do you want to CREATE or UPDATE?
What function do you want to CREATE or UPDATE?
What are details you want to add to drive the code changes?

Create predefined configurations for common programming languages in the language_support.py file.

File to UPDATE: backend/services/language_support.py

Add configurations for at least:
- Python
- JavaScript
- TypeScript
- Java
- C++
- Go
- Rust

For each language, define:
- Basic language metadata (name, version, file extensions)
- Supported features (typing, paradigms, etc.)
- Editor settings (indentation, code completion, snippets)
- Execution environment (commands, arguments, sandbox settings)

Make the configurations easily extensible for adding more languages in the future.