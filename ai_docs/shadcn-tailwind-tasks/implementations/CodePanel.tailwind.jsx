import React, { useState, useEffect, useCallback } from 'react';
import { FaPlay, FaSpinner } from 'react-icons/fa';
import CodeEditor from '../common/CodeEditor';
import { useLanguage } from '../../context/LanguageContext';

/**
 * CodePanel component that integrates the CodeEditor with execution controls
 * This version uses Tailwind CSS for styling.
 * 
 * @param {Object} props
 * @param {Function} props.onRunCode - Callback when code should be executed
 * @param {boolean} props.isExecuting - Whether code execution is in progress
 */
function CodePanel({ onRunCode, isExecuting = false }) {
  const { language } = useLanguage();
  const [code, setCode] = useState('');
  const [savedCode, setSavedCode] = useState(null);

  // Load saved code from localStorage on initial render
  useEffect(() => {
    const savedCode = localStorage.getItem(`code_${language}`);
    if (savedCode) {
      setCode(savedCode);
      setSavedCode(savedCode);
    }
  }, [language]);

  // Save code to localStorage when it changes
  useEffect(() => {
    if (code !== savedCode) {
      localStorage.setItem(`code_${language}`, code);
      setSavedCode(code);
    }
  }, [code, language, savedCode]);

  // Handle code execution with keyboard shortcuts
  const handleRunCode = useCallback(() => {
    if (!isExecuting && onRunCode) {
      onRunCode(code);
    }
  }, [code, isExecuting, onRunCode]);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl+Enter or Cmd+Enter to run code
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        handleRunCode();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleRunCode]);

  return (
    <div className="flex flex-col h-full rounded-md bg-white shadow-sm overflow-hidden dark:bg-slate-800">
      <div className="h-10 bg-slate-100 border-b border-slate-200 flex items-center justify-between px-4 dark:bg-slate-800 dark:border-slate-700">
        <div className="font-semibold">Code Editor ({language})</div>
        <div className="flex gap-2">
          <button 
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-white font-semibold text-sm transition-colors
              ${isExecuting 
                ? 'bg-blue-400 cursor-not-allowed dark:bg-blue-600' 
                : 'bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600'}`}
            onClick={handleRunCode}
            disabled={isExecuting}
          >
            {isExecuting ? (
              <>
                <FaSpinner className="animate-spin" /> Running...
              </>
            ) : (
              <>
                <FaPlay /> Run Code
              </>
            )}
          </button>
        </div>
      </div>
      <div className="flex-1 overflow-hidden">
        <CodeEditor
          initialValue={code}
          onChange={setCode}
          onRun={handleRunCode}
          options={{
            automaticLayout: true,
            minimap: { enabled: false }
          }}
        />
      </div>
    </div>
  );
}

export default CodePanel;