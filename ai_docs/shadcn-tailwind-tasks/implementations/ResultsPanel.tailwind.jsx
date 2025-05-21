import React from 'react';
import { FaSpinner } from 'react-icons/fa';

/**
 * ResultsPanel displays the results of code execution
 * This version uses Tailwind CSS for styling.
 * 
 * @param {Object} props
 * @param {string} props.output - The output from code execution
 * @param {string} props.error - Error message if code execution failed
 * @param {boolean} props.isLoading - Whether code is currently executing
 */
function ResultsPanel({ output = '', error = null, isLoading = false }) {
  return (
    <div className="flex flex-col h-full rounded-md bg-white shadow-sm overflow-hidden dark:bg-slate-800">
      <div className="h-10 bg-slate-100 border-b border-slate-200 flex items-center px-4 font-semibold dark:bg-slate-800 dark:border-slate-700">
        Execution Results
      </div>
      <div className="flex-1 p-4 overflow-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-full opacity-70">
            <FaSpinner className="animate-spin text-blue-600 mr-2 dark:text-blue-400" />
            <span>Running code...</span>
          </div>
        ) : output ? (
          <pre className={`font-mono text-sm whitespace-pre-wrap break-words ${error ? 'text-red-600 dark:text-red-400' : ''}`}>
            {output}
          </pre>
        ) : (
          <div className="text-slate-500 h-full flex items-center justify-center text-sm italic dark:text-slate-400">
            Run your code to see results here
          </div>
        )}
      </div>
    </div>
  );
}

export default ResultsPanel;