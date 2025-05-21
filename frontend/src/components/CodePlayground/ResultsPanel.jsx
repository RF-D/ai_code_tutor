import React from 'react';
import { FaTerminal, FaSpinner, FaExclamationTriangle } from 'react-icons/fa';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

/**
 * ResultsPanel displays code execution results with proper formatting
 * 
 * @param {Object} props
 * @param {string} props.output - The code execution output
 * @param {string} props.error - Any error message from execution
 * @param {boolean} props.isLoading - Whether results are being loaded
 * @param {string} props.executionTime - Time taken to execute the code
 */
function ResultsPanel({ 
  output = '', 
  error = null, 
  isLoading = false,
  executionTime = null
}) {
  // Determine if output is empty (no code has been run yet)
  const isEmpty = !output && !error && !isLoading;

  return (
    <div className="panel results-panel">
      <div className="panel-header">
        <div><FaTerminal /> Execution Results</div>
        {executionTime && <div className="execution-time">{executionTime}</div>}
      </div>
      <div className="panel-content">
        {isLoading ? (
          <div className="loading">
            <div className="spinner"></div>
            <p>Executing code...</p>
          </div>
        ) : isEmpty ? (
          <div className="empty-results">
            <p>Run your code to see results here</p>
          </div>
        ) : error ? (
          <div className="results-error">
            <div className="error-header">
              <FaExclamationTriangle /> Error
            </div>
            <pre className="error-message">{error}</pre>
            {output && <div className="output-with-error">{output}</div>}
          </div>
        ) : (
          <div className="results-success">
            <SyntaxHighlighter
              language="text"
              style={tomorrow}
              className="results-code"
              wrapLines={true}
              showLineNumbers={false}
            >
              {output}
            </SyntaxHighlighter>
          </div>
        )}
      </div>
    </div>
  );
}

export default ResultsPanel;
