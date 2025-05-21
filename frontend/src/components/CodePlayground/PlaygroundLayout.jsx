import React, { useState } from 'react';
import QuestionPanel from './QuestionPanel';
import CodePanel from './CodePanel';
import AssistantPanel from './AssistantPanel';
import ResultsPanel from './ResultsPanel';
import '../../styles/playground.css';

/**
 * PlaygroundLayout renders the practice question, code editor,
 * assistant chat and execution results. The previous implementation
 * relied on complex responsive logic and the `react-split` library
 * which made the editor unusable. This simplified version arranges
 * the panels using CSS grid so the code editor sits next to the
 * question panel at all times.
 */
function PlaygroundLayout() {
  const [output, setOutput] = useState('');
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const runCode = async (code) => {
    setIsRunning(true);
    setError(null);
    try {
      const result = await simulateRun(code);
      setOutput(result.output);
    } catch (e) {
      setError(e.message);
    } finally {
      setIsRunning(false);
    }
  };

  // Placeholder for real backend call
  const simulateRun = (code) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({ output: `Executed:\n${code}` });
      }, 500);
    });
  };

  return (
    <div className="playground-grid">
      <QuestionPanel />
      <CodePanel onRunCode={runCode} isExecuting={isRunning} />
      <AssistantPanel />
      <ResultsPanel output={output} error={error} isLoading={isRunning} />
    </div>
  );
}

export default PlaygroundLayout;
