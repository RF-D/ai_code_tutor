import React, { useState } from 'react';
import { FaChevronDown, FaChevronUp, FaLightbulb } from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';

/**
 * QuestionPanel displays the current practice problem with formatting and collapsible sections
 * This version uses Tailwind CSS for styling.
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
      case 'easy': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'medium': return 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200';
      case 'hard': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default: return '';
    }
  };

  return (
    <div className="flex flex-col h-full rounded-md bg-white shadow-sm overflow-hidden dark:bg-slate-800">
      <div className="h-10 bg-slate-100 border-b border-slate-200 flex items-center justify-between px-4 dark:bg-slate-800 dark:border-slate-700">
        <div>
          <span 
            className={`inline-block px-2 py-0.5 rounded-full text-xs mr-2 ${getDifficultyClass(currentQuestion.difficulty)}`}
          >
            {currentQuestion.difficulty}
          </span>
          <span className="font-semibold">{currentQuestion.title}</span>
        </div>
      </div>
      <div className="flex-1 p-4 overflow-auto text-sm leading-relaxed">
        <ReactMarkdown
          rehypePlugins={[rehypeRaw]}
          remarkPlugins={[remarkGfm]}
          className="prose prose-slate max-w-none dark:prose-invert"
        >
          {currentQuestion.description}
        </ReactMarkdown>
        
        {/* Details Section */}
        <div className="mt-4 border-t border-slate-200 pt-3 dark:border-slate-700">
          <button 
            className="flex items-center text-blue-600 border-none bg-transparent cursor-pointer dark:text-blue-400"
            onClick={() => toggleSection('details')}
          >
            {expandedSections.details ? <FaChevronUp className="mr-1" /> : <FaChevronDown className="mr-1" />}
            Details
          </button>
          
          {expandedSections.details && (
            <div className="mt-2 pl-1">
              <ReactMarkdown
                rehypePlugins={[rehypeRaw]}
                remarkPlugins={[remarkGfm]}
                className="prose prose-slate max-w-none dark:prose-invert"
              >
                {currentQuestion.details}
              </ReactMarkdown>
            </div>
          )}
        </div>
        
        {/* Hints Section */}
        <div className="mt-4 border-t border-slate-200 pt-3 dark:border-slate-700">
          <button 
            className="flex items-center text-blue-600 border-none bg-transparent cursor-pointer dark:text-blue-400"
            onClick={() => toggleSection('hints')}
          >
            {expandedSections.hints ? <FaChevronUp className="mr-1" /> : <FaChevronDown className="mr-1" />}
            <FaLightbulb className="text-amber-400 mr-1" /> Hints
          </button>
          
          {expandedSections.hints && (
            <div className="mt-2 pl-1">
              <ol className="list-decimal pl-5">
                {currentQuestion.hints.map((hint, index) => (
                  <li key={index} className="mb-1">{hint}</li>
                ))}
              </ol>
            </div>
          )}
        </div>
        
        {/* Tests Section */}
        <div className="mt-4 border-t border-slate-200 pt-3 dark:border-slate-700">
          <button 
            className="flex items-center text-blue-600 border-none bg-transparent cursor-pointer dark:text-blue-400"
            onClick={() => toggleSection('tests')}
          >
            {expandedSections.tests ? <FaChevronUp className="mr-1" /> : <FaChevronDown className="mr-1" />}
            Test Cases
          </button>
          
          {expandedSections.tests && (
            <div className="mt-2 pl-1">
              <div className="overflow-x-auto">
                <table className="min-w-full border-collapse text-sm">
                  <thead>
                    <tr className="bg-slate-100 dark:bg-slate-700">
                      <th className="border border-slate-300 px-4 py-2 text-left dark:border-slate-600">Input</th>
                      <th className="border border-slate-300 px-4 py-2 text-left dark:border-slate-600">Expected Output</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentQuestion.tests.map((test, index) => (
                      <tr key={index} className={index % 2 === 0 ? 'bg-white dark:bg-slate-800' : 'bg-slate-50 dark:bg-slate-900'}>
                        <td className="border border-slate-300 px-4 py-2 font-mono dark:border-slate-600">{JSON.stringify(test.input)}</td>
                        <td className="border border-slate-300 px-4 py-2 font-mono dark:border-slate-600">{JSON.stringify(test.expected)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default QuestionPanel;