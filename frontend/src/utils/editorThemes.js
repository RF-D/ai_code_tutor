/**
 * Monaco Editor themes configuration
 * Defines light and dark themes with appropriate syntax highlighting
 */

export const lightTheme = {
  base: 'vs',
  inherit: true,
  rules: [
    { token: 'comment', foreground: '008800', fontStyle: 'italic' },
    { token: 'keyword', foreground: '0000ff' },
    { token: 'string', foreground: 'a31515' },
    { token: 'number', foreground: '098658' },
    { token: 'operator', foreground: '000000' },
    { token: 'function', foreground: '795e26' },
    { token: 'variable', foreground: '001080' },
  ],
  colors: {
    'editor.foreground': '#000000',
    'editor.background': '#ffffff',
    'editor.selectionBackground': '#add6ff',
    'editor.lineHighlightBackground': '#f0f0f0',
    'editorCursor.foreground': '#000000',
    'editorWhitespace.foreground': '#d3d3d3',
    'editorIndentGuide.background': '#d3d3d3',
  },
};

export const darkTheme = {
  base: 'vs-dark',
  inherit: true,
  rules: [
    { token: 'comment', foreground: '6A9955', fontStyle: 'italic' },
    { token: 'keyword', foreground: '569CD6' },
    { token: 'string', foreground: 'CE9178' },
    { token: 'number', foreground: 'B5CEA8' },
    { token: 'operator', foreground: 'D4D4D4' },
    { token: 'function', foreground: 'DCDCAA' },
    { token: 'variable', foreground: '9CDCFE' },
  ],
  colors: {
    'editor.foreground': '#D4D4D4',
    'editor.background': '#1E1E1E',
    'editor.selectionBackground': '#264F78',
    'editor.lineHighlightBackground': '#2D2D30',
    'editorCursor.foreground': '#FFFFFF',
    'editorWhitespace.foreground': '#3B3B3B',
    'editorIndentGuide.background': '#3B3B3B',
  },
};

/**
 * Sets the editor theme based on the system/user preference
 * @param {Object} editor - Monaco editor instance
 * @param {boolean} isDarkMode - Whether to use dark mode
 */
export function setEditorTheme(monaco, isDarkMode) {
  const themeName = isDarkMode ? 'darkCustomTheme' : 'lightCustomTheme';
  const themeData = isDarkMode ? darkTheme : lightTheme;
  
  monaco.editor.defineTheme(themeName, themeData);
  monaco.editor.setTheme(themeName);
  
  return themeName;
}