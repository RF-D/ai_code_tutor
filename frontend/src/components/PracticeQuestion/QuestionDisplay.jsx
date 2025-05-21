import React, { useState, useMemo } from 'react';
import { useQuestion } from '../../context/QuestionContext';
import { useLanguage } from '../../context/LanguageContext';
import '../../styles/questions.css';

/**
 * QuestionDisplay component
 * Displays a practice question with its details and formatting
 */
const QuestionDisplay = ({ question }) => {
  const { currentLanguageId } = useLanguage();
  const [expandedSections, setExpandedSections] = useState({
    hints: false,
    sampleSolution: false,
    testCases: false,
  });

  // Early return if no question provided
  if (!question) {
    return null;
  }

  // Toggle section expansion
  const toggleSection = (section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  // Helper to get the difficulty class
  const getDifficultyClass = (level) => {
    return `difficulty-${level.toLowerCase()}`;
  };

  // Helper for code formatting based on language
  const formatCode = (code, language) => {
    // In a real implementation, this would use syntax highlighting
    // For now, just basic code display with language class
    return (
      <pre className={`code-block code-${language}`}>
        <code>{code}</code>
      </pre>
    );
  };

  // Get appropriate language-specific templates or examples
  const languageSpecificContent = useMemo(() => {
    const templates = {
      python: {
        functionTemplate: 'def solution(input):\n    # Your code here\n    return result',
        importExample: 'import math\nfrom collections import defaultdict',
        testExample: 'assert solution("test") == "expected"',
      },
      javascript: {
        functionTemplate: 'function solution(input) {\n    // Your code here\n    return result;\n}',
        importExample: 'const fs = require(\'fs\');\nconst path = require(\'path\');',
        testExample: 'console.assert(solution("test") === "expected");',
      }
    };
    
    return templates[currentLanguageId] || templates.python;
  }, [currentLanguageId]);

  return (
    <div className="question-display question-animation">
      <div className="question-header">
        <h2 className="question-title">{question.title || 'Practice Question'}</h2>
        <div className="question-metadata">
          <span className={`difficulty-tag ${getDifficultyClass(question.skillLevel)}`}>
            <span className="difficulty-indicator"></span>
            {question.skillLevel}
          </span>
          <span className="topic-tag">Topic: {question.topic}</span>
          <span className="language-tag">Language: {question.language}</span>
        </div>
      </div>

      <div className="question-description">
        {question.description}
      </div>

      <div className="question-instructions">
        <strong>Instructions:</strong>
        <p>{question.instructions}</p>
      </div>

      {/* Collapsible Hints Section */}
      {question.hints && question.hints.length > 0 && (
        <div className={`collapsible-section ${expandedSections.hints ? 'expanded' : ''}`}>
          <div
            className="collapsible-header"
            onClick={() => toggleSection('hints')}
            role="button"
            tabIndex={0}
          >
            <span>Hints ({question.hints.length})</span>
            <span>{expandedSections.hints ? '▲' : '▼'}</span>
          </div>
          <div className="collapsible-content">
            <ul className="hints-list">
              {question.hints.map((hint, index) => (
                <li key={index} className="hint-item">
                  {hint}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Collapsible Sample Solution Section */}
      {question.sampleSolution && (
        <div className={`collapsible-section ${expandedSections.sampleSolution ? 'expanded' : ''}`}>
          <div
            className="collapsible-header"
            onClick={() => toggleSection('sampleSolution')}
            role="button"
            tabIndex={0}
          >
            <span>Sample Solution</span>
            <span>{expandedSections.sampleSolution ? '▲' : '▼'}</span>
          </div>
          <div className="collapsible-content">
            {formatCode(question.sampleSolution, question.language)}
            
            {/* Language-specific additional information */}
            <div className="language-specific-notes">
              <h4>{question.language} Implementation Notes:</h4>
              <div className="code-examples">
                <div className="code-example">
                  <div className="example-label">Common imports:</div>
                  {formatCode(languageSpecificContent.importExample, question.language)}
                </div>
                <div className="code-example">
                  <div className="example-label">Function template:</div>
                  {formatCode(languageSpecificContent.functionTemplate, question.language)}
                </div>
                <div className="code-example">
                  <div className="example-label">Testing example:</div>
                  {formatCode(languageSpecificContent.testExample, question.language)}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Collapsible Test Cases Section */}
      {question.testCases && question.testCases.length > 0 && (
        <div className={`collapsible-section ${expandedSections.testCases ? 'expanded' : ''}`}>
          <div
            className="collapsible-header"
            onClick={() => toggleSection('testCases')}
            role="button"
            tabIndex={0}
          >
            <span>Test Cases ({question.testCases.length})</span>
            <span>{expandedSections.testCases ? '▲' : '▼'}</span>
          </div>
          <div className="collapsible-content">
            <div className="test-cases">
              {question.testCases.map((testCase, index) => (
                <div key={index} className="test-case">
                  <div className="test-case-header">Test Case {index + 1}</div>
                  <div className="test-case-content">
                    <div className="test-input">
                      <strong>Input:</strong>
                      <pre>{testCase.input}</pre>
                    </div>
                    <div className="test-expected">
                      <strong>Expected Output:</strong>
                      <pre>{testCase.expectedOutput}</pre>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QuestionDisplay;