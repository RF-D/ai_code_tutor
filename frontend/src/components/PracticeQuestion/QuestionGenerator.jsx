import React, { useState, useEffect, useCallback } from 'react';
import { useQuestion } from '../../context/QuestionContext';
import { useLanguage } from '../../context/LanguageContext';
import TopicSelector from './TopicSelector';
import DifficultySelector from './DifficultySelector';
import QuestionDisplay from './QuestionDisplay';
import '../../styles/questions.css';

/**
 * QuestionGenerator component
 * Main component for generating practice questions with selected parameters
 */
const QuestionGenerator = () => {
  const {
    currentQuestion,
    topic,
    skillLevel,
    fetchQuestion,
    fetchTopicSuggestions,
    isLoading,
    error
  } = useQuestion();
  
  const { currentLanguageId, currentLanguageName } = useLanguage();
  const [generationError, setGenerationError] = useState(null);
  const [languageSpecificHints, setLanguageSpecificHints] = useState([]);

  // Update error state when context error changes
  useEffect(() => {
    if (error) {
      setGenerationError(error);
    }
  }, [error]);

  // Generate a question with current parameters
  const handleGenerateQuestion = async () => {
    setGenerationError(null);
    
    if (!topic.trim()) {
      setGenerationError('Please enter a topic before generating a question.');
      return;
    }
    
    try {
      await fetchQuestion(topic, skillLevel);
    } catch (err) {
      setGenerationError(err.message || 'Failed to generate question. Please try again.');
    }
  };

  // Get language-specific hints or templates based on the selected language
  const getLanguageSpecificHints = useCallback(() => {
    const hintsByLanguage = {
      python: [
        'Consider using list comprehensions for concise code',
        'Remember that Python is indentation-sensitive',
        'Use built-in functions like map(), filter(), or reduce() when appropriate'
      ],
      javascript: [
        'Consider using arrow functions for cleaner syntax',
        'Remember to handle asynchronous operations properly',
        'Use modern ES6+ features like destructuring and spread operators'
      ]
    };
    
    return hintsByLanguage[currentLanguageId] || [];
  }, [currentLanguageId]);
  
  // Update language-specific hints when language changes
  useEffect(() => {
    setLanguageSpecificHints(getLanguageSpecificHints());
    
    // Refresh topic suggestions when language changes
    fetchTopicSuggestions('');
  }, [currentLanguageId, getLanguageSpecificHints, fetchTopicSuggestions]);
  
  // Clear error on parameters change
  useEffect(() => {
    setGenerationError(null);
  }, [topic, skillLevel, currentLanguageId]);

  return (
    <div className="question-container">
      <h2>Generate {currentLanguageName} Practice Question</h2>
      
      <div className="question-generator-form">
        <div className="form-row">
          <TopicSelector />
        </div>
        
        <div className="form-row">
          <DifficultySelector />
        </div>
        
        {/* Language-specific hints */}
        {languageSpecificHints.length > 0 && (
          <div className="language-hints">
            <p className="hint-heading">Tips for {currentLanguageName} questions:</p>
            <ul className="hint-list">
              {languageSpecificHints.map((hint, index) => (
                <li key={index} className="hint-item">{hint}</li>
              ))}
            </ul>
          </div>
        )}
        
        {/* Error message display */}
        {generationError && (
          <div className="error-container" role="alert">
            {generationError}
          </div>
        )}
        
        <button
          className="generate-btn"
          onClick={handleGenerateQuestion}
          disabled={isLoading || !topic.trim()}
        >
          {isLoading ? 'Generating...' : 'Generate Question'}
        </button>
      </div>
      
      {/* Loading indicator */}
      {isLoading && (
        <div className="loading-container">
          <div className="loading-spinner" aria-hidden="true"></div>
          <span className="loading-text">Generating your practice question...</span>
        </div>
      )}
      
      {/* Display generated question */}
      {!isLoading && currentQuestion && (
        <QuestionDisplay question={currentQuestion} />
      )}
    </div>
  );
};

export default QuestionGenerator;