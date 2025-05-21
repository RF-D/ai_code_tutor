# Frontend Setup Plan - Phase 1 & 2

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objectives

- Create the initial React frontend scaffolding described in `react-refactoring-plan.md`.
- Optimize frontend performance through code splitting, memoization, and build configuration.

## Mid-Level Objectives

### Phase 1: Initial Setup
- Set up the `frontend/` directory with `public/` and `src/` subfolders.
- Add placeholder component files matching the plan (e.g., `CodeEditor.jsx`, `QuestionGenerator.jsx`).
- Provide minimal `App.jsx`, `index.jsx`, and `App.css` files.
- Include a `package.json` with basic dependencies.
- Document frontend setup instructions in `frontend/README.md` and reference it from the root `README.md`.

### Phase 2: Performance Optimization
- Implement code splitting with React.lazy in the routes.jsx file.
- Add memoization to frequently re-rendered components like CodeEditor.
- Update the Vite configuration for production builds.
- Optimize the Tailwind configuration to reduce CSS bundle size.
- Document performance optimization strategies.

## Implementation Notes
- Components should export simple functional components returning a `<div>` with the component name.
- Use modern React with ES modules.
- Keep version numbers in `package.json` as placeholders (e.g., "0.1.0").
- Development server may use `vite` or `react-scripts` depending on preference.

## Context

### Beginning context
- `README.md`
- `main.py`
- `requirements.txt`
- `utils/llm_manager.py`

### Ending context
- `frontend/` with `public/` and `src/` directories containing component stubs
- `frontend/package.json` and `frontend/README.md`
- Updated root `README.md` referencing frontend instructions

## Low-Level Tasks
> Ordered from start to finish

### Phase 1: Initial Setup
1. Scaffold directories and component stubs
```aider
Create `frontend/public`, `frontend/src`, and nested component folders according to the refactoring plan.
Add placeholder component files under `frontend/src/components/...` each exporting a React function returning a `<div>` with the component name.
```
2. Add application entry files
```aider
Create `frontend/src/App.jsx` with a basic "Hello World" component.
Create `frontend/src/index.jsx` that renders `<App />`.
Create `frontend/src/App.css` with minimal placeholder styles.
```
3. Configure package and documentation
```aider
Create `frontend/package.json` with placeholder name, version, and scripts for starting the dev server.
Write `frontend/README.md` explaining how to install dependencies and run the dev server.
Update the root `README.md` with a short section pointing to the frontend README.
```

### Phase 2: Performance Optimization (✓ Completed)
1. Implement code splitting with React.lazy
```aider
Enhance routes.jsx with React.lazy for code splitting.
Add error boundaries for lazy-loaded components.
Implement route prefetching for common routes.
```
2. Add memoization to components
```aider
Apply React.memo to CodeEditor with custom comparison function.
Add memoization to EditorToolbar, CodePanel, and UI components.
Implement useMemo and useCallback for computed values and callbacks.
```
3. Update build configuration
```aider
Optimize Vite configuration for production builds.
Add compression plugins, HTML minification, and bundle analysis.
Enhance chunk splitting for better caching.
```
4. Optimize Tailwind CSS
```aider
Reduce Tailwind CSS bundle size by optimizing configuration.
Add purge settings for unused styles.
Disable unused core plugins and features.
```
5. Document optimization strategies
```aider
Create documentation for all performance optimizations.
Include metrics and future improvement suggestions.
```
