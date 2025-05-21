# Specification: Convert Layout to Tailwind

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Convert the main layout components to use Tailwind CSS classes instead of custom CSS

## Mid-Level Objective

- Update MainLayout.jsx to use Tailwind CSS for styling
- Convert Navigation.jsx to use Tailwind CSS
- Convert basic layout containers and common elements to Tailwind
- Ensure layout functionality and appearance remains consistent
- Keep existing CSS modules for components that haven't been migrated yet

## Implementation Notes
- Use utility-first approach with Tailwind classes
- Extract commonly used class combinations into @apply directives if they're reused frequently
- Maintain the same visual appearance and functionality
- Keep accessibility features intact (focus states, keyboard navigation, etc.)
- Add comments explaining complex class combinations if necessary
- Implement responsive design using Tailwind's responsive modifiers

## Context

### Beginning context
- src/layouts/MainLayout.jsx - Main application layout
- src/components/common/Navigation.jsx - Main navigation component
- src/styles/layout.css - Current layout styles
- src/styles/components/Navigation.module.css - Current navigation styles

### Ending context
- Updated MainLayout.jsx with Tailwind classes
- Updated Navigation.jsx with Tailwind classes
- Possibly updated globals.css with @apply directives for reused patterns
- Original CSS files kept but no longer imported by converted components

## Low-Level Tasks
> Ordered from start to finish

1. Convert MainLayout.jsx to use Tailwind CSS

What prompt would you run to complete this task?
"Convert the MainLayout.jsx component to use Tailwind CSS classes instead of custom CSS. Maintain the same visual appearance and functionality, and ensure that the layout remains responsive."

What file do you want to UPDATE?
UPDATE src/layouts/MainLayout.jsx

What are details you want to add to drive the code changes?
Replace CSS class names with equivalent Tailwind utility classes. Remove any imports of CSS modules that are no longer needed. Ensure that the layout container, main content area, and any padding/margin styles are properly converted.

2. Convert Navigation.jsx to use Tailwind CSS

What prompt would you run to complete this task?
"Convert the Navigation.jsx component to use Tailwind CSS classes instead of the Navigation.module.css CSS module. Maintain the same visual appearance, functionality, and responsive behavior."

What file do you want to UPDATE?
UPDATE src/components/common/Navigation.jsx

What are details you want to add to drive the code changes?
Replace module CSS classes with Tailwind utilities. Pay special attention to active states, hover effects, and responsive behavior. Ensure the navigation links maintain their styling and that mobile navigation works correctly if applicable.

3. Create Tailwind utility patterns for common layout styles

What prompt would you run to complete this task?
"Identify common style patterns used across layout components and create @apply directives in globals.css to make these patterns reusable. Update the converted components to use these patterns if applicable."

What file do you want to UPDATE?
UPDATE src/styles/globals.css
UPDATE src/layouts/MainLayout.jsx (if needed)
UPDATE src/components/common/Navigation.jsx (if needed)

What are details you want to add to drive the code changes?
Create custom utility classes using @apply for any styles that are used repeatedly. For example, if there's a common card style, button style, or layout container style. Update previously converted components to use these utilities if applicable.

4. Test layout in different viewport sizes

What prompt would you run to complete this task?
"Test the converted layout components at different viewport sizes by adding media query debugging code. Update any responsive styles that don't match the original behavior."

What file do you want to UPDATE?
UPDATE src/layouts/MainLayout.jsx (if needed)
UPDATE src/components/common/Navigation.jsx (if needed)

What are details you want to add to drive the code changes?
Add temporary debugging elements or styles that show the current responsive breakpoint. Verify that layout behaves correctly at all breakpoints. Update any Tailwind classes that need adjustment to match the original responsive behavior.

5. Clean up unused CSS files and imports

What prompt would you run to complete this task?
"Remove any CSS imports that are no longer needed in the converted components. Do not delete the original CSS files yet, but comment them out to verify everything works without them."

What file do you want to UPDATE?
UPDATE src/layouts/MainLayout.jsx
UPDATE src/components/common/Navigation.jsx
UPDATE any other files that import from the converted components

What are details you want to add to drive the code changes?
Remove or comment out imports of CSS modules that are no longer needed. Update any other files that import or use these components if necessary. Comment out the CSS imports rather than deleting them completely, so they can be restored if needed.