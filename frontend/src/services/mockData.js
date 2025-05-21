/**
 * Mock Data for API Development
 * 
 * This file provides realistic mock data for all API endpoints to facilitate
 * frontend development without requiring a running backend.
 */
import { ProgrammingLanguage, DifficultyLevel, LLMProvider } from './apiTypes';

// Configuration for mock responses
const MOCK_CONFIG = {
  // Simulate network delay (ms)
  delay: 800,
  
  // Occasionally return errors for testing error handling
  errorRate: 0.05,
  
  // Enable/disable specific mock endpoints
  enabledMocks: {
    topics: true,
    questions: true,
    codeEvaluation: true,
    codeExecution: true,
    hints: true,
    languages: true,
    models: true
  }
};

/**
 * Helper to simulate API delay and occasionally return errors
 */
export async function mockResponse(data) {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, MOCK_CONFIG.delay));
  
  // Occasionally return an error
  if (Math.random() < MOCK_CONFIG.errorRate) {
    throw {
      status: 500,
      message: 'Simulated server error',
      detail: 'This is a mock error response for testing error handling'
    };
  }
  
  return data;
}

/**
 * Mock topic suggestions
 */
export const mockTopicSuggestions = [
  { id: '1', name: 'Variables and Data Types', category: 'basics', relatedTopics: ['Operators', 'Type Conversion'] },
  { id: '2', name: 'Control Flow', category: 'basics', relatedTopics: ['Conditionals', 'Loops'] },
  { id: '3', name: 'Functions', category: 'basics', relatedTopics: ['Parameters', 'Return Values', 'Recursion'] },
  { id: '4', name: 'Lists and Arrays', category: 'data structures', relatedTopics: ['Indexing', 'Slicing', 'Methods'] },
  { id: '5', name: 'Dictionaries', category: 'data structures', relatedTopics: ['Keys', 'Values', 'Methods'] },
  { id: '6', name: 'Classes and Objects', category: 'oop', relatedTopics: ['Inheritance', 'Encapsulation', 'Polymorphism'] },
  { id: '7', name: 'File Handling', category: 'io', relatedTopics: ['Reading Files', 'Writing Files', 'CSV Processing'] },
  { id: '8', name: 'Error Handling', category: 'advanced', relatedTopics: ['Try/Except', 'Custom Exceptions'] },
  { id: '9', name: 'Regular Expressions', category: 'advanced', relatedTopics: ['Pattern Matching', 'Substitution'] },
  { id: '10', name: 'Algorithms', category: 'advanced', relatedTopics: ['Sorting', 'Searching', 'Recursion'] },
];

/**
 * Mock practice questions
 */
export const mockPracticeQuestions = [
  {
    id: '1',
    topic: 'Variables and Data Types',
    question: 'Create a function that takes a temperature in Celsius and returns it converted to Fahrenheit. The formula is: F = C * 9/5 + 32',
    explanation: 'This question tests your understanding of arithmetic operations and function creation in Python.',
    skillLevel: DifficultyLevel.BEGINNER,
    justification: 'This question involves basic arithmetic operations and function definition, making it suitable for beginners.',
    language: ProgrammingLanguage.PYTHON
  },
  {
    id: '2',
    topic: 'Control Flow',
    question: 'Write a function that takes a list of numbers and returns a new list containing only the even numbers.',
    explanation: 'This question tests your understanding of loops, conditionals, and list manipulation.',
    skillLevel: DifficultyLevel.BEGINNER,
    justification: 'This question covers basic control flow concepts like loops and conditionals, appropriate for beginners.',
    language: ProgrammingLanguage.PYTHON
  },
  {
    id: '3',
    topic: 'Functions',
    question: 'Create a recursive function to calculate the factorial of a number.',
    explanation: 'This question tests your understanding of recursion and function implementation.',
    skillLevel: DifficultyLevel.INTERMEDIATE,
    justification: 'Recursion is an intermediate concept that requires understanding of function calls and base cases.',
    language: ProgrammingLanguage.PYTHON
  },
  {
    id: '4',
    topic: 'Data Structures',
    question: 'Implement a function that reverses a string using a stack data structure.',
    explanation: 'This question tests your understanding of the stack data structure and string manipulation.',
    skillLevel: DifficultyLevel.INTERMEDIATE,
    justification: 'This requires understanding of both stacks and string operations, making it appropriate for intermediate learners.',
    language: ProgrammingLanguage.PYTHON
  },
  {
    id: '5',
    topic: 'Algorithms',
    question: 'Implement a binary search algorithm for a sorted list of integers.',
    explanation: 'This question tests your understanding of the binary search algorithm and its implementation.',
    skillLevel: DifficultyLevel.ADVANCED,
    justification: 'Binary search requires understanding of algorithmic complexity and recursion or loop control, making it suitable for advanced learners.',
    language: ProgrammingLanguage.PYTHON
  }
];

/**
 * Mock code evaluation responses
 */
export const mockEvaluationResponses = {
  correct: {
    id: 'eval-1',
    overallAssessment: 'Correct',
    correctness: 'The solution correctly implements the required functionality.',
    comments: 'Your solution is well-structured and correctly converts Celsius to Fahrenheit using the formula F = C * 9/5 + 32. The code handles the input appropriately and returns the expected output.',
    suggestions: 'Consider adding input validation to handle potential errors. You could also add type hints to make your function more robust.',
    metrics: {
      executionTime: 5,
      memoryUsage: 4096
    }
  },
  partiallyCorrect: {
    id: 'eval-2',
    overallAssessment: 'Partially Correct',
    correctness: 'The solution implements the main functionality but has minor issues.',
    comments: 'Your solution correctly identifies even numbers, but there\'s a logical error in how you\'re appending elements to the list. Check your conditional statement carefully.',
    suggestions: 'Review how the modulo operator (%) works with even numbers. Also consider using list comprehensions for a more concise solution.',
    metrics: {
      executionTime: 8,
      memoryUsage: 4096
    }
  },
  incorrect: {
    id: 'eval-3',
    overallAssessment: 'Incorrect',
    correctness: 'The solution does not correctly solve the problem.',
    comments: 'Your recursive factorial function is missing a base case, which will cause it to run indefinitely for any input. Additionally, the recursive call is incorrect.',
    suggestions: 'Add a base case for 0 or 1 to stop the recursion. Make sure your recursive call decrements the input value.',
    metrics: {
      executionTime: 0,
      memoryUsage: 0
    }
  }
};

/**
 * Mock code execution responses
 */
export const mockExecutionResponses = {
  success: {
    output: 'Hello, World!\n5\n10\n15\n20\n25',
    executionTime: 12
  },
  error: {
    output: '',
    error: 'NameError: name \'undefined_variable\' is not defined',
    executionTime: 3
  },
  timeout: {
    output: '',
    error: 'Execution timed out after 5000ms',
    executionTime: 5000
  }
};

/**
 * Mock hint responses
 */
export const mockHintResponses = [
  {
    id: 'hint-1',
    content: 'Think about how you can use the modulo operator (%) to determine if a number is even.',
    type: 'conceptual'
  },
  {
    id: 'hint-2',
    content: 'Remember that the formula for converting Celsius to Fahrenheit is F = C * 9/5 + 32.',
    type: 'formula'
  },
  {
    id: 'hint-3',
    content: 'For a recursive function, you need a base case that stops the recursion. For factorial, think about what factorial of 0 or 1 should return.',
    type: 'approach'
  },
  {
    id: 'hint-4',
    content: 'Consider using a list comprehension: `[x for x in numbers if x % 2 == 0]`',
    type: 'code'
  }
];

/**
 * Mock language metadata
 */
export const mockLanguages = [
  {
    id: ProgrammingLanguage.PYTHON,
    name: 'Python',
    version: '3.9',
    topicsAvailable: ['Variables and Data Types', 'Control Flow', 'Functions', 'Data Structures', 'OOP', 'File Handling', 'Error Handling', 'Regular Expressions', 'Algorithms'],
    editorConfig: {
      tabSize: 4,
      insertSpaces: true,
      defaultCode: 'def solution():\n    # Your code here\n    pass\n'
    }
  },
  {
    id: ProgrammingLanguage.JAVASCRIPT,
    name: 'JavaScript',
    version: 'ES2021',
    topicsAvailable: ['Variables and Data Types', 'Control Flow', 'Functions', 'Arrays and Objects', 'DOM Manipulation', 'Asynchronous JS', 'Error Handling', 'Regular Expressions', 'Algorithms'],
    editorConfig: {
      tabSize: 2,
      insertSpaces: true,
      defaultCode: 'function solution() {\n  // Your code here\n}\n'
    }
  },
  {
    id: ProgrammingLanguage.JAVA,
    name: 'Java',
    version: '17',
    topicsAvailable: ['Variables and Data Types', 'Control Flow', 'Methods', 'Classes and Objects', 'Inheritance', 'Interfaces', 'Generics', 'Exception Handling', 'Collections', 'Algorithms'],
    editorConfig: {
      tabSize: 4,
      insertSpaces: true,
      defaultCode: 'public class Solution {\n    public static void main(String[] args) {\n        // Your code here\n    }\n}\n'
    }
  }
];

/**
 * Mock model data
 */
export const mockModels = [
  {
    id: 'claude-3-haiku',
    name: 'Claude 3 Haiku',
    provider: LLMProvider.ANTHROPIC,
    capabilities: {
      codeGeneration: true,
      codeEvaluation: true
    }
  },
  {
    id: 'claude-3-sonnet',
    name: 'Claude 3 Sonnet',
    provider: LLMProvider.ANTHROPIC,
    capabilities: {
      codeGeneration: true,
      codeEvaluation: true
    }
  },
  {
    id: 'gpt-4o',
    name: 'GPT-4o',
    provider: LLMProvider.OPENAI,
    capabilities: {
      codeGeneration: true,
      codeEvaluation: true
    }
  },
  {
    id: 'codestral-latest',
    name: 'Codestral',
    provider: LLMProvider.MISTRAL,
    capabilities: {
      codeGeneration: true,
      codeEvaluation: true
    }
  },
  {
    id: 'llama3-70b',
    name: 'Llama 3 70B',
    provider: LLMProvider.OLLAMA,
    capabilities: {
      codeGeneration: true,
      codeEvaluation: true
    }
  }
];

/**
 * Mock API implementation functions
 * These functions can be used as direct replacements for the real API functions
 */

// Topic and question functions
export async function getTopicSuggestionsData() {
  return mockResponse(mockTopicSuggestions);
}

export async function generatePracticeQuestionsData(topic, skillLevel, language, count = 1) {
  const filteredQuestions = mockPracticeQuestions
    .filter(q => q.language === language && q.skillLevel === skillLevel)
    .slice(0, count);
  
  return mockResponse({ questions: filteredQuestions.length > 0 ? filteredQuestions : [mockPracticeQuestions[0]] });
}

export async function getQuestionsByFiltersData(language, difficulty, topic) {
  const filteredQuestions = mockPracticeQuestions.filter(q => {
    let match = q.language === language;
    if (difficulty) match = match && q.skillLevel === difficulty;
    if (topic) match = match && q.topic.toLowerCase().includes(topic.toLowerCase());
    return match;
  });
  
  return mockResponse({ 
    questions: filteredQuestions,
    total: filteredQuestions.length,
    page: 1,
    pageSize: 10
  });
}

export async function getQuestionByIdData(questionId) {
  const question = mockPracticeQuestions.find(q => q.id === questionId);
  if (!question) {
    throw {
      status: 404,
      message: 'Question not found',
      detail: `No question found with ID ${questionId}`
    };
  }
  return mockResponse(question);
}

// Code evaluation functions
export async function evaluateCodeData(code, questionId) {
  const codeLength = code.length;
  let response;
  
  if (code.includes('return') && codeLength > 50) {
    response = mockEvaluationResponses.correct;
  } else if (code.includes('if') && codeLength > 20) {
    response = mockEvaluationResponses.partiallyCorrect;
  } else {
    response = mockEvaluationResponses.incorrect;
  }
  
  return mockResponse(response);
}

export async function executeCodeData(code) {
  if (code.includes('while True') || code.includes('for i in range(1000000)')) {
    return mockResponse(mockExecutionResponses.timeout);
  } else if (code.includes('undefined_variable') || code.includes('syntax error')) {
    return mockResponse(mockExecutionResponses.error);
  } else {
    return mockResponse(mockExecutionResponses.success);
  }
}

// Hint functions
export async function requestHintData(questionId, question) {
  const questionLower = question.toLowerCase();
  let hintIndex = 0;
  
  if (questionLower.includes('even')) {
    hintIndex = 0;
  } else if (questionLower.includes('celsius') || questionLower.includes('fahrenheit')) {
    hintIndex = 1;
  } else if (questionLower.includes('factorial') || questionLower.includes('recursion')) {
    hintIndex = 2;
  } else {
    hintIndex = 3;
  }
  
  return mockResponse(mockHintResponses[hintIndex]);
}

// Language functions
export async function getAvailableLanguagesData() {
  return mockResponse(mockLanguages.map(lang => ({
    id: lang.id,
    name: lang.name,
    version: lang.version
  })));
}

export async function getLanguageMetadataData(language) {
  const langData = mockLanguages.find(lang => lang.id === language);
  if (!langData) {
    throw {
      status: 404,
      message: 'Language not found',
      detail: `No language found with ID ${language}`
    };
  }
  return mockResponse(langData);
}

// Model functions
export async function getAvailableModelsData() {
  return mockResponse(mockModels);
}

/**
 * Utility to toggle between real and mock API during development
 * @param {Function} realApiFunction - The real API function
 * @param {Function} mockFunction - The mock function
 * @returns {Function} - Either the real or mock function based on configuration
 */
export function chooseMockOrReal(realApiFunction, mockFunction, mockType) {
  // If mocks are disabled or this specific mock is disabled, use real API
  if (!MOCK_CONFIG.enabledMocks[mockType]) {
    return realApiFunction;
  }
  return mockFunction;
}

// Usage in your components:
// import { getAvailableLanguages, getAvailableLanguagesData, chooseMockOrReal } from '../services/mockData';
// const fetchLanguages = chooseMockOrReal(getAvailableLanguages, getAvailableLanguagesData, 'languages');