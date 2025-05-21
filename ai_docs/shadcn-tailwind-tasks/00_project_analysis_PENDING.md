# Specification: Project Analysis

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Analyze the current frontend architecture to understand components, layout, and styling approach before implementing Shadcn and Tailwind CSS

## Mid-Level Objective

- Review the existing component structure to identify what needs to be migrated to Tailwind/Shadcn
- Analyze the current styling approach (CSS modules, global CSS) and components
- Document the current layout composition in PlaygroundLayout
- Identify key UI components that will need Shadcn equivalents
- Understand the current theming system for light/dark mode support

## Implementation Notes
- Don't modify any actual code in this phase - just analyze and document
- Focus on understanding how the React components are structured and how they interact
- Pay special attention to the PlaygroundLayout.jsx component which will be restructured
- Document current CSS usage patterns (modules vs. global, CSS variables, etc.)
- Note potential challenges in migration (complex components, custom styling, etc.)

## Context

### Beginning context
- src/components/ - Contains all React components
- src/styles/ - Contains CSS files
- src/context/ - Contains React context providers
- src/layouts/ - Contains layout components

### Ending context
- No code changes, just detailed analysis documentation

## Low-Level Tasks
> Ordered from start to finish

1. Analyze component directory structure and component organization

What prompt would you run to complete this task?
"Analyze the frontend component structure by examining directories in src/components/ and create a hierarchical diagram showing how components are organized."

What file do you want to CREATE or UPDATE?
CREATE a markdown document with the component hierarchy analysis

What are details you want to add to drive the code changes?
Document the component hierarchy, component responsibilities, and how they relate to each other. Note which components might need Shadcn replacements.

2. Analyze current styling approach

What prompt would you run to complete this task?
"Analyze the styling approach used in the frontend by examining CSS files, module imports, and theme implementation. Document how styles are currently applied and how theming works."

What file do you want to CREATE or UPDATE?
UPDATE the analysis document with CSS and styling information

What are details you want to add to drive the code changes?
Document CSS module usage, global styles, theme variables, responsive design approach, and how components are styled. This will inform the Tailwind migration strategy.

3. Analyze PlaygroundLayout component structure

What prompt would you run to complete this task?
"Analyze the PlaygroundLayout component to understand how panels are structured, how react-split is used, and how the layout is currently implemented. Focus on identifying changes needed to place the editor next to the question panel instead of under it."

What file do you want to CREATE or UPDATE?
UPDATE the analysis document with PlaygroundLayout details

What are details you want to add to drive the code changes?
Document the current layout structure, panel sizing, localStorage persistence, and the specific changes needed to implement the horizontal layout for the question and editor panels.

4. Evaluate third-party dependencies

What prompt would you run to complete this task?
"Analyze package.json to identify current dependencies and determine compatibility with Shadcn/UI and Tailwind. Note any potential conflicts or dependencies that might need updates."

What file do you want to CREATE or UPDATE?
UPDATE the analysis document with dependency analysis

What are details you want to add to drive the code changes?
List all UI-related dependencies, their compatibility with Tailwind/Shadcn, and recommend which ones to keep, replace, or update.

5. Create migration path recommendation

What prompt would you run to complete this task?
"Based on all previous analysis, create a recommended migration path that outlines steps, priorities, and potential challenges for implementing Shadcn/UI and Tailwind CSS."

What file do you want to CREATE or UPDATE?
UPDATE the analysis document with migration recommendations

What are details you want to add to drive the code changes?
Provide a strategic migration plan with recommended order of components to convert, approach for handling global styles, and how to transition from current CSS modules to Tailwind classes.