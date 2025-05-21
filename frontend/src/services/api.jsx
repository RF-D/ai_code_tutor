/**
 * API Service
 * 
 * This file contains functions for calling the backend API endpoints.
 */
import { ProgrammingLanguage, DifficultyLevel } from './apiTypes';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Common configuration for fetch requests
 */
const defaultHeaders = {
  'Content-Type': 'application/json',
};

/**
 * Generic API request handler with error handling and retries
 */
async function apiRequest(url, options, retries = 3) {
  try {
    const response = await fetch(`${API_BASE_URL}${url}`, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options?.headers,
      },
    });

    // Error handling for non-200 responses
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const error = new Error(errorData.message || 'API request failed');
      error.status = response.status;
      error.detail = errorData.detail;
      throw error;
    }

    // Parse response
    return await response.json();
  } catch (error) {
    // Retry logic for network errors
    if (retries > 0 && (error.name === 'TypeError' || error.name === 'NetworkError')) {
      console.warn(`API request failed, retrying (${retries} attempts left)`);
      return apiRequest(url, options, retries - 1);
    }
    throw error;
  }
}

/* -------------------- Practice Question API -------------------- */

/**
 * Fetch topic suggestions based on optional filters
 */
export async function getTopicSuggestions(language = ProgrammingLanguage.PYTHON, category = null) {
  return apiRequest(`/api/topics/suggest?language=${language}${category ? `&category=${category}` : ''}`);
}

/**
 * Generate practice questions based on topic and difficulty
 */
export async function generatePracticeQuestions(topic, skillLevel = DifficultyLevel.BEGINNER, language = ProgrammingLanguage.PYTHON, count = 1) {
  return apiRequest('/api/questions/generate', {
    method: 'POST',
    body: JSON.stringify({
      topic,
      skillLevel,
      language,
      count,
    }),
  });
}

/**
 * Fetch questions by language and difficulty
 */
export async function getQuestionsByFilters(language = ProgrammingLanguage.PYTHON, difficulty = null, topic = null, page = 1, limit = 10) {
  let url = `/api/questions?language=${language}&page=${page}&limit=${limit}`;
  if (difficulty) url += `&difficulty=${difficulty}`;
  if (topic) url += `&topic=${encodeURIComponent(topic)}`;
  
  return apiRequest(url);
}

/**
 * Get a specific practice question by ID
 */
export async function getQuestionById(questionId) {
  return apiRequest(`/api/questions/${questionId}`);
}

/* -------------------- Code Evaluation API -------------------- */

/**
 * Submit code for evaluation
 */
export async function evaluateCode(code, questionId, language = ProgrammingLanguage.PYTHON, model = null) {
  return apiRequest('/api/code/evaluate', {
    method: 'POST',
    body: JSON.stringify({
      code,
      questionId,
      language,
      model,
    }),
  });
}

/**
 * Execute code and return results
 */
export async function executeCode(code, language = ProgrammingLanguage.PYTHON, inputs = {}) {
  return apiRequest('/api/code/execute', {
    method: 'POST',
    body: JSON.stringify({
      code,
      language,
      inputs,
    }),
  });
}

/**
 * Get execution metrics for a specific code submission
 */
export async function getCodeMetrics(submissionId) {
  return apiRequest(`/api/code/metrics/${submissionId}`);
}

/* -------------------- Solution Assistance API -------------------- */

/**
 * Request a hint for a practice question
 */
export async function requestHint(questionId, question, code = null, chatHistory = null) {
  return apiRequest('/api/assistance/hint', {
    method: 'POST',
    body: JSON.stringify({
      questionId,
      question,
      code,
      chatHistory,
    }),
  });
}

/**
 * Get explanation for a solution
 */
export async function getExplanation(questionId, code) {
  return apiRequest('/api/assistance/explain', {
    method: 'POST',
    body: JSON.stringify({
      questionId,
      code,
    }),
  });
}

/**
 * Get code improvement suggestions
 */
export async function getCodeImprovements(code, language = ProgrammingLanguage.PYTHON) {
  return apiRequest('/api/assistance/improve', {
    method: 'POST',
    body: JSON.stringify({
      code,
      language,
    }),
  });
}

/* -------------------- Language Support API -------------------- */

/**
 * Fetch available programming languages
 */
export async function getAvailableLanguages() {
  return apiRequest('/api/languages');
}

/**
 * Get metadata for a specific language
 */
export async function getLanguageMetadata(language) {
  return apiRequest(`/api/languages/${language}`);
}

/**
 * Get available topics for a specific language
 */
export async function getLanguageTopics(language) {
  return apiRequest(`/api/languages/${language}/topics`);
}

/**
 * Save user language preferences
 */
export async function setLanguagePreference(language, isDefault = false) {
  return apiRequest('/api/languages/preferences', {
    method: 'POST',
    body: JSON.stringify({
      language,
      isDefault,
    }),
  });
}

/* -------------------- Model Selection API -------------------- */

/**
 * Fetch available LLM models
 */
export async function getAvailableModels() {
  return apiRequest('/api/models');
}

/**
 * Set active model for specific functionality
 */
export async function setActiveModel(modelId, purpose) {
  return apiRequest('/api/models/active', {
    method: 'POST',
    body: JSON.stringify({
      modelId,
      purpose,
    }),
  });
}

/**
 * Get model performance metrics
 */
export async function getModelMetrics(modelId) {
  return apiRequest(`/api/models/${modelId}/metrics`);
}

/**
 * Get model-specific configuration options
 */
export async function getModelConfig(modelId) {
  return apiRequest(`/api/models/${modelId}/config`);
}

/**
 * Update model configuration
 */
export async function updateModelConfig(modelId, config) {
  return apiRequest(`/api/models/${modelId}/config`, {
    method: 'PUT',
    body: JSON.stringify(config),
  });
}