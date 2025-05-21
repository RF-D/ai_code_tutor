# Monaco Editor Integration
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Implement a fully-featured Monaco editor component for code input with multi-language support

## Mid-Level Objective

- Install and configure Monaco Editor React component
- Set up language-specific settings and syntax highlighting
- Implement editor themes (light/dark)
- Create a toolbar with language selection and formatting options
- Integrate with the language context for language switching

## Implementation Notes
- Use @monaco-editor/react package for integration
- Configure editor with proper settings for each language
- Support Python, JavaScript, TypeScript, and React syntax
- Implement editor layout and sizing that works well in the application
- Keep performance in mind when loading the editor

## Context

### Beginning context
- frontend/src/components/common/CodeEditor.jsx (empty)
- frontend/src/context/LanguageContext.jsx
- frontend/package.json

### Ending context  
- Updated frontend/package.json with Monaco dependencies
- frontend/src/components/common/CodeEditor.jsx (fully implemented)
- frontend/src/components/common/EditorToolbar.jsx
- frontend/src/utils/languageUtils.js
- frontend/src/utils/editorThemes.js

## Low-Level Tasks
> Ordered from start to finish

1. Update package.json with Monaco Editor dependencies
```aider
Update frontend/package.json to add:
- @monaco-editor/react as a dependency
- monaco-editor as a dependency
- Add any other dependencies needed for syntax highlighting or language features
- Update the version numbers to latest compatible versions
```

2. Create editorThemes.js utility
```aider
Create frontend/src/utils/editorThemes.js that:
- Defines light and dark themes for Monaco Editor
- Sets appropriate colors for syntax highlighting
- Ensures good contrast for readability
- Matches the application's overall design aesthetic
```

3. Create languageUtils.js for editor language configuration
```aider
Create frontend/src/utils/languageUtils.js that:
- Defines configurations for each supported language (Python, JavaScript, TypeScript, React)
- Sets up language-specific editor settings (tabSize, insertSpaces, etc.)
- Provides utility functions for language detection and switching
- Sets up syntax highlighting rules if needed
```

4. Create EditorToolbar component
```aider
Create frontend/src/components/common/EditorToolbar.jsx that:
- Provides buttons for common editor actions (run, format, etc.)
- Displays the current language and allows switching
- Includes theme toggle (light/dark)
- Has proper styling and responsive layout
- Integrates with the language context
```

5. Implement the CodeEditor component
```aider
Update frontend/src/components/common/CodeEditor.jsx to:
- Integrate the Monaco editor with proper configuration
- Connect to language context for language-specific settings
- Include the EditorToolbar component
- Handle code changes and saving
- Support resizing and full-screen mode
- Implement proper editor event handling
```

6. Add editor-specific event handlers
```aider
Enhance the CodeEditor component to:
- Add keyboard shortcut handlers (run, format, etc.)
- Implement proper error display in the editor
- Add line highlighting for errors/warnings
- Implement clean debouncing for performance
- Handle editor focus and blur events appropriately
```