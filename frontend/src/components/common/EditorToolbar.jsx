import React, { useCallback } from 'react';
import { useLanguage } from '../../context/LanguageContext';

/**
 * EditorToolbar component provides controls for the code editor
 * 
 * @param {Object} props
 * @param {Function} props.onRun - Callback when Run button is clicked
 * @param {Function} props.onFormat - Callback when Format button is clicked
 * @param {boolean} props.isDarkMode - Current theme mode
 * @param {Function} props.onToggleTheme - Callback to toggle between light/dark themes
 * @param {Function} props.onToggleFullscreen - Callback to toggle fullscreen mode
 */
function EditorToolbar({
  onRun,
  onFormat,
  isDarkMode,
  onToggleTheme,
  onToggleFullscreen,
}) {
  const {
    currentLanguageId,
    availableLanguages,
    changeLanguage,
  } = useLanguage();

  const handleLanguageChange = useCallback(
    (e) => {
      changeLanguage(e.target.value);
    },
    [changeLanguage]
  );

  return (
    <div className="editor-toolbar">
      <div className="editor-toolbar-group">
        <button
          className="toolbar-button run-button"
          onClick={onRun}
          title="Run Code (Ctrl+Enter)"
        >
          Run
        </button>
        <button
          className="toolbar-button format-button"
          onClick={onFormat}
          title="Format Code (Shift+Alt+F)"
        >
          Format
        </button>
      </div>

      <div className="editor-toolbar-group language-select">
        <select
          value={currentLanguageId}
          onChange={handleLanguageChange}
          title="Select Programming Language"
        >
          {availableLanguages.map((lang) => (
            <option key={lang.id} value={lang.id}>
              {lang.name}
            </option>
          ))}
        </select>
      </div>

      <div className="editor-toolbar-group">
        <button
          className="toolbar-button theme-button"
          onClick={onToggleTheme}
          title="Toggle Light/Dark Theme"
        >
          {isDarkMode ? '☀️ Light' : '🌙 Dark'}
        </button>
        <button
          className="toolbar-button fullscreen-button"
          onClick={onToggleFullscreen}
          title="Toggle Fullscreen (F11)"
        >
          Fullscreen
        </button>
      </div>
    </div>
  );
}

// Memoize the EditorToolbar to avoid unnecessary re-renders
export default React.memo(EditorToolbar, (prevProps, nextProps) => {
  // Only re-render when these props change
  return prevProps.isDarkMode === nextProps.isDarkMode;
});