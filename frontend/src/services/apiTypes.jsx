/**
 * API Type Definitions
 * 
 * This file contains TypeScript interfaces for all API requests and responses.
 */

/**
 * Programming language enum
 */
export const ProgrammingLanguage = {
  PYTHON: 'python',
  JAVASCRIPT: 'javascript',
  TYPESCRIPT: 'typescript',
  JAVA: 'java',
  CSHARP: 'csharp',
  CPP: 'cpp',
  RUST: 'rust',
  GO: 'go'
};

/**
 * Difficulty level enum
 */
export const DifficultyLevel = {
  BEGINNER: 'beginner',
  INTERMEDIATE: 'intermediate',
  ADVANCED: 'advanced'
};

/**
 * LLM provider enum
 */
export const LLMProvider = {
  ANTHROPIC: 'Anthropic',
  OPENAI: 'OpenAI',
  GROQ: 'Groq',
  MISTRAL: 'Mistral',
  OLLAMA: 'Ollama'
};

/**
 * Base API error response
 * @typedef {Object} ApiError
 * @property {number} status - HTTP status code
 * @property {string} message - Error message
 * @property {string} [detail] - Detailed error information
 */

/**
 * Model configuration for available LLM models
 * @typedef {Object} LLMModel
 * @property {string} id - Unique identifier for the model
 * @property {string} name - Display name for the model
 * @property {string} provider - Provider of the model (from LLMProvider enum)
 * @property {Object} capabilities - Feature capabilities of this model
 * @property {boolean} capabilities.codeGeneration - Whether this model can generate code
 * @property {boolean} capabilities.codeEvaluation - Whether this model can evaluate code
 */

/**
 * Provider with available models
 * @typedef {Object} Provider
 * @property {string} name - Provider name
 * @property {LLMModel[]} models - Available models from this provider
 */

/**
 * Topic suggestion
 * @typedef {Object} TopicSuggestion
 * @property {string} id - Unique identifier
 * @property {string} name - Topic name
 * @property {string} category - Topic category (e.g., "basics", "data structures")
 * @property {string[]} [relatedTopics] - Related topics
 */

/**
 * Practice question
 * @typedef {Object} PracticeQuestion
 * @property {string} id - Unique identifier
 * @property {string} topic - Main topic being tested
 * @property {string} question - The actual practice question
 * @property {string} explanation - Brief explanation of the question
 * @property {string} skillLevel - Appropriate skill level (from DifficultyLevel enum)
 * @property {string} [justification] - Why the question is appropriate for the given skill level
 * @property {string} language - Programming language for the question (from ProgrammingLanguage enum)
 */

/**
 * Code evaluation request
 * @typedef {Object} CodeEvaluationRequest
 * @property {string} code - Code to evaluate
 * @property {string} language - Programming language (from ProgrammingLanguage enum)
 * @property {string} questionId - ID of the practice question
 * @property {string} [model] - Optional model to use for evaluation
 */

/**
 * Code evaluation response
 * @typedef {Object} CodeEvaluationResponse
 * @property {string} id - Unique identifier for the evaluation
 * @property {string} overallAssessment - Overall assessment ("Correct", "Partially Correct", "Incorrect")
 * @property {string} correctness - Assessment of code correctness
 * @property {string} comments - Detailed explanation
 * @property {string} suggestions - Specific ideas for improvement
 * @property {Object} [metrics] - Performance metrics if available
 * @property {number} [metrics.executionTime] - Code execution time in ms
 * @property {number} [metrics.memoryUsage] - Memory usage in bytes
 */

/**
 * Code execution request
 * @typedef {Object} CodeExecutionRequest
 * @property {string} code - Code to execute
 * @property {string} language - Programming language (from ProgrammingLanguage enum)
 * @property {Object} [inputs] - Test case inputs if needed
 */

/**
 * Code execution response
 * @typedef {Object} CodeExecutionResponse
 * @property {string} output - Execution output (stdout)
 * @property {string} [error] - Error message if execution failed
 * @property {number} executionTime - Execution time in milliseconds
 */

/**
 * Hint request
 * @typedef {Object} HintRequest
 * @property {string} questionId - ID of the practice question
 * @property {string} question - User's specific question about the problem
 * @property {string} [code] - Current code if available
 * @property {string} [chatHistory] - Previous hints and questions
 */

/**
 * Hint response
 * @typedef {Object} HintResponse
 * @property {string} id - Unique identifier for the hint
 * @property {string} content - The hint content
 * @property {string} type - Type of hint (e.g., "conceptual", "code", "approach")
 */

/**
 * Language metadata
 * @typedef {Object} LanguageMetadata
 * @property {string} id - Language identifier (from ProgrammingLanguage enum)
 * @property {string} name - Display name
 * @property {string} version - Version information
 * @property {string[]} topicsAvailable - Topics available for this language
 * @property {Object} editorConfig - Configuration for the code editor
 */

/**
 * Generate questions request
 * @typedef {Object} GenerateQuestionsRequest
 * @property {string} topic - Topic for the questions
 * @property {string} [language] - Programming language (from ProgrammingLanguage enum)
 * @property {string} skillLevel - Skill level (from DifficultyLevel enum)
 * @property {number} [count] - Number of questions to generate (default 1)
 */

/**
 * Question generation response
 * @typedef {Object} GenerateQuestionsResponse
 * @property {PracticeQuestion[]} questions - Generated questions
 */

/**
 * User preferences
 * @typedef {Object} UserPreferences
 * @property {string} preferredLanguage - Preferred programming language
 * @property {string} preferredTheme - Editor theme preference
 * @property {string} defaultDifficulty - Default difficulty level
 * @property {string} preferredModel - Preferred LLM model
 */