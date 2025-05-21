import React, { useState, useRef, useCallback, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { useLanguage } from '../../context/LanguageContext';
import EditorToolbar from './EditorToolbar';
import { 
  getMonacoLanguageId,
  getEditorOptions,
  languageTemplates
} from '../../utils/languageUtils';
import { setEditorTheme } from '../../utils/editorThemes';
import './CodeEditor.css';

// Simple debounce function to improve performance
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

/**
 * CodeEditor component integrates Monaco Editor with support for multiple languages
 * 
 * @param {Object} props
 * @param {string} props.initialValue - Initial code value
 * @param {Function} props.onChange - Callback when code changes
 * @param {Function} props.onRun - Callback when Run is triggered
 * @param {Function} props.onError - Callback when errors occur
 * @param {Array} props.markers - Error/warning markers to display in editor
 * @param {Object} props.options - Additional Monaco editor options
 */
function CodeEditor({
  initialValue,
  onChange,
  onRun,
  onError,
  markers = [],
  options = {}
}) {
  const { language } = useLanguage();
  const editorRef = useRef(null);
  const monacoRef = useRef(null);
  const [isDarkMode, setIsDarkMode] = useState(
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [editorValue, setEditorValue] = useState(
    initialValue || languageTemplates[language] || ''
  );

  // Handle theme changes
  const handleToggleTheme = useCallback(() => {
    setIsDarkMode(prev => !prev);
  }, []);

  // Handle code formatting
  const handleFormat = useCallback(() => {
    if (editorRef.current) {
      editorRef.current.getAction('editor.action.formatDocument').run();
    }
  }, []);

  // Handle fullscreen toggle
  const handleToggleFullscreen = useCallback(() => {
    setIsFullscreen(prev => !prev);
  }, []);

  // Update error decorations in the editor
  // Memoize the error decoration function for better performance
  const updateErrorDecorations = useCallback((editor, monaco, markers) => {
    if (!editor || !monaco) return;
    
    // Convert markers to Monaco format
    const modelMarkers = markers.map(marker => ({
      startLineNumber: marker.line || 1,
      startColumn: marker.column || 1,
      endLineNumber: marker.endLine || marker.line || 1,
      endColumn: marker.endColumn || marker.column || 1000,
      message: marker.message || 'Error',
      severity: marker.severity === 'warning' 
        ? monaco.MarkerSeverity.Warning 
        : monaco.MarkerSeverity.Error
    }));
    
    // Set markers on the model
    const model = editor.getModel();
    if (model) {
      monaco.editor.setModelMarkers(model, 'owner', modelMarkers);
    }
  }, []);

  // Set up editor when Monaco is loaded
  // Using useCallback with an empty dependency array for stable mount handler
const handleEditorDidMount = useCallback((editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;
    
    // Set theme based on current mode
    setEditorTheme(monaco, isDarkMode);
    
    // Add keyboard shortcut for running code (Ctrl+Enter)
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
      if (onRun) onRun(editor.getValue());
    });
    
    // Add keyboard shortcut for formatting (Shift+Alt+F)
    editor.addCommand(
      monaco.KeyMod.Shift | monaco.KeyMod.Alt | monaco.KeyCode.KeyF,
      handleFormat
    );
    
    // Add keyboard shortcut for fullscreen toggle (F11)
    editor.addCommand(monaco.KeyCode.F11, () => {
      handleToggleFullscreen();
      // Prevent default browser behavior
      return true;
    });
    
    // Focus the editor
    editor.focus();
    
    // Add error-related decorations
    updateErrorDecorations(editor, monaco, markers);
    
    // Add editor focus/blur events
    editor.onDidFocusEditorText(() => {
      // You could trigger actions when editor gets focus
    });
    
    editor.onDidBlurEditorText(() => {
      // You could save code automatically on blur
    });
  }, [isDarkMode, onRun, markers, handleFormat, handleToggleFullscreen, updateErrorDecorations]);

  // Update theme when it changes
  useEffect(() => {
    if (monacoRef.current) {
      setEditorTheme(monacoRef.current, isDarkMode);
    }
  }, [isDarkMode]);
  
  // Update error markers when they change
  useEffect(() => {
    if (editorRef.current && monacoRef.current) {
      updateErrorDecorations(editorRef.current, monacoRef.current, markers);
    }
  }, [markers, updateErrorDecorations]);

  // Handle code execution
  const handleRunCode = useCallback(() => {
    if (onRun && editorRef.current) {
      onRun(editorRef.current.getValue());
    }
  }, [onRun]);
  
  // Handle editor value changes (debounced)
  // Stable debounced callback with configurable delay
  const debouncedOnChange = useRef(
    debounce((value) => {
      if (onChange) onChange(value);
    }, 300)
  ).current;
  
  // Memoize editor options for better performance
  const editorOptions = React.useMemo(() => ({
    ...getEditorOptions(language),
    ...options
  }), [language, options]);
  
  const handleEditorChange = useCallback((value) => {
    setEditorValue(value);
    debouncedOnChange(value);
  }, [debouncedOnChange]);

  return (
    <div className={`code-editor-container ${isFullscreen ? 'fullscreen' : ''}`}>
      <EditorToolbar
        onRun={handleRunCode}
        onFormat={handleFormat}
        isDarkMode={isDarkMode}
        onToggleTheme={handleToggleTheme}
        onToggleFullscreen={handleToggleFullscreen}
      />
      
      <div className="monaco-editor-wrapper">
        <Editor
          height="100%"
          language={getMonacoLanguageId(language)}
          value={editorValue}
          options={editorOptions}
          onChange={handleEditorChange}
          onMount={handleEditorDidMount}
          theme={isDarkMode ? 'vs-dark' : 'vs-light'}
        />
      </div>
    </div>
  );
}

// Enhanced memoization with deep equality check
export default React.memo(CodeEditor, (prevProps, nextProps) => {
  // Deep comparison for complex objects
  const areMarkersEqual = () => {
    if (!prevProps.markers && !nextProps.markers) return true;
    if (!prevProps.markers || !nextProps.markers) return false;
    if (prevProps.markers.length !== nextProps.markers.length) return false;
    
    // Efficient comparison of marker arrays
    for (let i = 0; i < prevProps.markers.length; i++) {
      const prev = prevProps.markers[i];
      const next = nextProps.markers[i];
      if (prev.line !== next.line || 
          prev.column !== next.column || 
          prev.message !== next.message ||
          prev.severity !== next.severity) {
        return false;
      }
    }
    return true;
  };
  
  const areOptionsEqual = () => {
    // Fast path: reference equality
    if (prevProps.options === nextProps.options) return true;
    
    // Fast path: empty objects
    const prevKeys = Object.keys(prevProps.options || {});
    const nextKeys = Object.keys(nextProps.options || {});
    
    if (prevKeys.length !== nextKeys.length) return false;
    
    // Check each key
    for (const key of prevKeys) {
      if (prevProps.options[key] !== nextProps.options[key]) {
        return false;
      }
    }
    
    return true;
  };
  
  // Only re-render if these props change
  return (
    prevProps.initialValue === nextProps.initialValue &&
    areMarkersEqual() &&
    areOptionsEqual() &&
    prevProps.onRun === nextProps.onRun &&
    prevProps.onChange === nextProps.onChange
  );
});
