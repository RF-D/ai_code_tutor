# Specification: Performance Optimization

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Optimize the frontend application for performance after implementing Shadcn/UI and Tailwind CSS

## Mid-Level Objective

- Configure Tailwind to purge unused CSS classes in production
- Implement code splitting for better load times
- Optimize Shadcn/UI component imports
- Improve React component rendering performance
- Set up proper caching strategies for assets

## Implementation Notes
- Use Tailwind's built-in purge capabilities to remove unused CSS
- Implement React.lazy and Suspense for code splitting
- Ensure that only necessary Shadcn/UI components are included in the bundle
- Use memoization techniques for expensive components
- Configure Vite for optimal production builds
- Add performance monitoring and measurement tools

## Context

### Beginning context
- tailwind.config.js - Current Tailwind configuration
- vite.config.js - Current Vite configuration
- src/App.jsx - Main application component
- src/components/ - All components including Shadcn/UI components

### Ending context
- Updated tailwind.config.js with purge configuration
- Updated vite.config.js with performance optimizations
- Updated App.jsx with code splitting
- Optimized component imports and rendering

## Low-Level Tasks
> Ordered from start to finish

1. Configure Tailwind CSS purging for production

What prompt would you run to complete this task?
"Update the Tailwind configuration to properly purge unused CSS classes in production builds. Ensure that all template files are included in the purge configuration."

What file do you want to UPDATE?
UPDATE tailwind.config.js

What are details you want to add to drive the code changes?
Configure the content array in tailwind.config.js to include all files that contain Tailwind classes. Ensure dynamic class usage is properly safeguarded. Add any needed safelist for classes that might be generated dynamically and wouldn't be detected by the purge process.

2. Implement code splitting for main application routes

What prompt would you run to complete this task?
"Implement code splitting for the main application routes using React.lazy and Suspense. Update the routes configuration to use lazy loading for page components."

What file do you want to UPDATE?
UPDATE src/routes.jsx
UPDATE src/App.jsx (or main component that renders routes)

What are details you want to add to drive the code changes?
Replace static imports of page components with React.lazy imports. Wrap the route rendering with Suspense and provide appropriate fallback components. Ensure that related components are grouped logically to prevent too many small chunks.

3. Optimize Shadcn component imports

What prompt would you run to complete this task?
"Review and optimize the Shadcn/UI component imports to ensure only necessary code is included in the bundle. Implement a strategy for importing only the components that are needed."

What file do you want to UPDATE?
UPDATE files importing Shadcn/UI components

What are details you want to add to drive the code changes?
Replace any "import from index" style imports with direct imports from specific component files. Create utilities for commonly used component combinations if needed. Consider creating a custom components file that re-exports only the Shadcn components actually used in the application.

4. Implement memoization for expensive components

What prompt would you run to complete this task?
"Identify components with expensive rendering and implement memoization using React.memo, useMemo, and useCallback where appropriate. Focus on components in the code playground that may cause performance issues."

What file do you want to UPDATE?
UPDATE components that handle frequent updates or complex rendering

What are details you want to add to drive the code changes?
Wrap components with React.memo where appropriate. Use useMemo for expensive calculations. Use useCallback for functions passed as props to memoized components. Focus especially on the CodeEditor component and any components that handle user input or frequent updates.

5. Configure Vite for optimal production builds

What prompt would you run to complete this task?
"Update the Vite configuration to optimize production builds. Configure chunk sizes, asset handling, and build optimization settings."

What file do you want to UPDATE?
UPDATE vite.config.js

What are details you want to add to drive the code changes?
Configure build.chunkSizeWarningLimit and build.rollupOptions to optimize chunk sizes. Set up build.minify to use terser for better minification. Configure proper asset handling with build.assetsInlineLimit. Set up appropriate caching strategies with build.rollupOptions.output.manualChunks.