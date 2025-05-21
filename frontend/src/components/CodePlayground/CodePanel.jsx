import React, { useState, useEffect, useCallback } from 'react';
import { FaPlay, FaSpinner } from 'react-icons/fa';
import CodeEditor from '../common/CodeEditor';
import { useLanguage } from '../../context/LanguageContext';

/**
 * CodePanel component that integrates the CodeEditor with execution controls
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
    <div className="panel code-panel">
      <div className="panel-header">
        <div>Code Editor ({language})</div>
        <div className="code-actions">
          <button 
            className="run-button" 
            onClick={handleRunCode}
            disabled={isExecuting}
          >
            {isExecuting ? (
              <>
                <FaSpinner className="icon-spin" /> Running...
              </>
            ) : (
              <>
                <FaPlay /> Run Code
              </>
            )}
          </button>
        </div>
      </div>
      <div className="panel-content">
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

// Memoize the CodePanel component to avoid unnecessary re-renders
export default React.memo(CodePanel, (prevProps, nextProps) => {
  // Only re-render when execution state changes
  return prevProps.isExecuting === nextProps.isExecuting;
});
