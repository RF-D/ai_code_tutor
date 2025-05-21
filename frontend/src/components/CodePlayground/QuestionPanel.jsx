import React, { useState } from 'react';
import { FaChevronDown, FaChevronUp, FaLightbulb } from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

/**
 * QuestionPanel displays the current practice problem with formatting and collapsible sections
 * 
 * @param {Object} props
 * @param {Object} props.question - The question data (optional - uses sample if not provided)
 * @param {Function} props.onNavigateQuestion - Callback when navigating to another question
 */
function QuestionPanel({ question = null, onNavigateQuestion }) {
  // Sample question data for demonstration
  const sampleQuestion = {
    id: 'q001',
    title: 'Sum of Two Numbers',
    difficulty: 'easy',
    description: 'Write a function that returns the sum of two numbers.',
    details: `
## Problem Description

Given two integers \`a\` and \`b\`, return their sum.

### Examples:

**Input:** a = 5, b = 3  
**Output:** 8

**Input:** a = -1, b = 1  
**Output:** 0

**Input:** a = 0, b = 0  
**Output:** 0

### Constraints:
- \`-100 <= a, b <= 100\`

### Notes:
- Try to think about edge cases
- Consider different ways to implement this function
    `,
    hints: [
      'This is a basic arithmetic operation.',
      'In most languages, you can use the + operator to add numbers.',
      'Make sure your function returns the value and doesn\'t just print it.'
    ],
    tests: [
      { input: [5, 3], expected: 8 },
      { input: [-1, 1], expected: 0 },
      { input: [0, 0], expected: 0 }
    ]
  };

  // Use provided question or sample
  const currentQuestion = question || sampleQuestion;
  
  // State for expanded/collapsed sections
  const [expandedSections, setExpandedSections] = useState({
    details: true,
    hints: false,
    tests: true
  });
  
  // Toggle section expanded state
  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };
  
  // Get difficulty class
  const getDifficultyClass = (difficulty) => {
    switch (difficulty.toLowerCase()) {
      case 'easy': return 'difficulty-easy';
      case 'medium': return 'difficulty-medium';
      case 'hard': return 'difficulty-hard';
      default: return '';
    }
  };

  return (
    <div className="panel question-panel">
      <div className="panel-header">
        <div>
          <span 
            className={`question-difficulty ${getDifficultyClass(currentQuestion.difficulty)}`}
          >
            {currentQuestion.difficulty}
          </span>
          {currentQuestion.title}
        </div>
      </div>
      <div className="panel-content question-content">
        <ReactMarkdown
          rehypePlugins={[rehypeRaw]}
          remarkPlugins={[remarkGfm]}
        >
          {currentQuestion.description}
        </ReactMarkdown>
        
        {/* Details Section */}
        <div className="question-section">
          <button 
            className="collapse-button"
            onClick={() => toggleSection('details')}
          >
            {expandedSections.details ? <FaChevronUp /> : <FaChevronDown />}
            Details
          </button>
          
          {expandedSections.details && (
            <div className="section-content">
              <ReactMarkdown
                rehypePlugins={[rehypeRaw]}
                remarkPlugins={[remarkGfm]}
              >
                {currentQuestion.details}
              </ReactMarkdown>
            </div>
          )}
        </div>
        
        {/* Hints Section */}
        <div className="question-section">
          <button 
            className="collapse-button"
            onClick={() => toggleSection('hints')}
          >
            {expandedSections.hints ? <FaChevronUp /> : <FaChevronDown />}
            <FaLightbulb style={{ color: '#FFD700' }} /> Hints
          </button>
          
          {expandedSections.hints && (
            <div className="section-content">
              <ol>
                {currentQuestion.hints.map((hint, index) => (
                  <li key={index}>{hint}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
        
        {/* Tests Section */}
        <div className="question-section">
          <button 
            className="collapse-button"
            onClick={() => toggleSection('tests')}
          >
            {expandedSections.tests ? <FaChevronUp /> : <FaChevronDown />}
            Test Cases
          </button>
          
          {expandedSections.tests && (
            <div className="section-content">
              <table className="test-cases">
                <thead>
                  <tr>
                    <th>Input</th>
                    <th>Expected Output</th>
                  </tr>
                </thead>
                <tbody>
                  {currentQuestion.tests.map((test, index) => (
                    <tr key={index}>
                      <td>{JSON.stringify(test.input)}</td>
                      <td>{JSON.stringify(test.expected)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default QuestionPanel;
