/**
 * Utility functions for Monaco Editor language configuration
 * Provides settings for different programming languages
 */

// Default editor options applicable to all languages
export const defaultEditorOptions = {
  fontSize: 14,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  automaticLayout: true,
  scrollbar: {
    vertical: 'auto',
    horizontal: 'auto',
  },
  lineNumbers: 'on',
  renderLineHighlight: 'all',
  suggestOnTriggerCharacters: true,
  wordWrap: 'on',
};

// Language-specific editor configurations
export const languageConfigurations = {
  python: {
    tabSize: 4,
    insertSpaces: true,
    formatOnPaste: true,
    formatOnType: true,
    autoIndent: 'full',
    snippetSuggestions: 'inline',
    quickSuggestions: {
      other: true,
      comments: false,
      strings: false,
    },
  },
  javascript: {
    tabSize: 2,
    insertSpaces: true,
    formatOnPaste: true,
    formatOnType: true,
    autoIndent: 'full',
    snippetSuggestions: 'inline',
    quickSuggestions: {
      other: true,
      comments: false,
      strings: true,
    },
  },
  typescript: {
    tabSize: 2,
    insertSpaces: true,
    formatOnPaste: true,
    formatOnType: true,
    autoIndent: 'full',
    snippetSuggestions: 'inline',
    quickSuggestions: {
      other: true,
      comments: false,
      strings: true,
    },
  },
  jsx: {
    tabSize: 2,
    insertSpaces: true,
    formatOnPaste: true,
    formatOnType: true,
    autoIndent: 'full',
    snippetSuggestions: 'inline',
    quickSuggestions: {
      other: true,
      comments: false,
      strings: true,
    },
  },
};

// Default code templates for different languages
export const languageTemplates = {
  python: '# Python code\ndef main():\n    print("Hello, World!")\n\nif __name__ == "__main__":\n    main()',
  javascript: '// JavaScript code\nfunction main() {\n  console.log("Hello, World!");\n}\n\nmain();',
  typescript: '// TypeScript code\nfunction main(): void {\n  console.log("Hello, World!");\n}\n\nmain();',
  jsx: '// React JSX code\nimport React from "react";\n\nfunction App() {\n  return (\n    <div>\n      <h1>Hello, World!</h1>\n    </div>\n  );\n}\n\nexport default App;',
};

// Map file extensions to language IDs
export const fileExtensionToLanguage = {
  '.py': 'python',
  '.js': 'javascript',
  '.ts': 'typescript',
  '.jsx': 'javascript',
  '.tsx': 'typescript',
};

/**
 * Get the appropriate language ID for Monaco Editor based on the language name
 * @param {string} language - The language name
 * @returns {string} The Monaco Editor language ID
 */
export function getMonacoLanguageId(language) {
  const languageMap = {
    python: 'python',
    javascript: 'javascript',
    typescript: 'typescript',
    jsx: 'javascript', // Monaco uses javascript for JSX files
  };
  
  return languageMap[language] || 'plaintext';
}

/**
 * Get the file extension for a given language
 * @param {string} language - The language name
 * @returns {string} The file extension
 */
export function getFileExtension(language) {
  const extensionMap = {
    python: '.py',
    javascript: '.js',
    typescript: '.ts',
    jsx: '.jsx',
  };
  
  return extensionMap[language] || '';
}

/**
 * Get the complete editor options for a specific language
 * @param {string} language - The language name
 * @returns {Object} Combined editor options
 */
export function getEditorOptions(language) {
  const langConfig = languageConfigurations[language] || languageConfigurations.python;
  
  return {
    ...defaultEditorOptions,
    ...langConfig,
  };
}