# Specification: Tailwind CSS and Shadcn/UI Setup

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Set up Tailwind CSS and Shadcn/UI in the frontend project

## Mid-Level Objective

- Install necessary dependencies for Tailwind CSS and Shadcn/UI
- Configure Tailwind CSS with the appropriate plugins
- Set up the Shadcn/UI CLI and component system
- Configure theme colors to match the current design system
- Create a basic test component to verify setup

## Implementation Notes
- Tailwind CSS requires PostCSS and autoprefixer
- Shadcn/UI uses a CLI tool for installation and component generation
- Use the existing color scheme in src/styles/theme.css as a reference for Tailwind theme colors
- Ensure the tailwind.config.js includes all necessary plugins and theme settings
- Implement proper configuration for CSS variable support in Tailwind
- Configure Shadcn/UI to use React without TypeScript 

## Context

### Beginning context
- package.json - Current dependencies
- src/styles/theme.css - Current theme color definitions
- vite.config.js - Current build configuration

### Ending context
- package.json - Updated with new dependencies
- tailwind.config.js - New configuration file for Tailwind
- postcss.config.js - New configuration file for PostCSS
- components.json - New configuration file for Shadcn/UI
- src/styles/globals.css - New global styles file with Tailwind directives
- src/components/ui/button.jsx - Example Shadcn component for testing

## Low-Level Tasks
> Ordered from start to finish

1. Install Tailwind CSS and its dependencies

What prompt would you run to complete this task?
"Install Tailwind CSS, PostCSS, and autoprefixer as dev dependencies using npm. Then create the basic configuration files for Tailwind and PostCSS."

What file do you want to CREATE or UPDATE?
UPDATE package.json
CREATE tailwind.config.js
CREATE postcss.config.js

What are details you want to add to drive the code changes?
Install tailwindcss, postcss, and autoprefixer. Configure tailwind.config.js with the content paths for all React components, and set up postcss.config.js to use Tailwind and autoprefixer.

2. Set up global CSS with Tailwind directives

What prompt would you run to complete this task?
"Create a new globals.css file in the styles directory with the necessary Tailwind directives (@tailwind base, components, utilities) and update the main CSS import in the app."

What file do you want to CREATE or UPDATE?
CREATE src/styles/globals.css
UPDATE src/index.jsx

What are details you want to add to drive the code changes?
Include the three Tailwind directives in globals.css and update the main entry point to import this new CSS file. Ensure that other CSS imports are maintained for now during the transition.

3. Configure Tailwind theme with existing color palette

What prompt would you run to complete this task?
"Extract color variables from the existing theme.css file and configure them in the Tailwind theme settings in tailwind.config.js. Use CSS variables for theme compatibility."

What file do you want to UPDATE?
UPDATE tailwind.config.js
UPDATE src/styles/globals.css

What are details you want to add to drive the code changes?
Configure the Tailwind theme with the same colors from theme.css, using CSS variables to support the existing theme switching. Add color definitions to globals.css that reference these variables.

4. Install Shadcn/UI CLI and configure components

What prompt would you run to complete this task?
"Install the Shadcn/UI CLI, initialize it for a React project, and configure it to add components to the UI directory."

What file do you want to CREATE or UPDATE?
UPDATE package.json
CREATE components.json

What are details you want to add to drive the code changes?
Install the shadcn-ui CLI, run the init command, and configure it to use JSX files (not TypeScript) and store components in the src/components/ui directory. Set the style to be "default" and use CSS variables for theming.

5. Add a test Shadcn/UI component and verify setup

What prompt would you run to complete this task?
"Use the Shadcn CLI to add a Button component, then create a simple test page to verify that both Tailwind and Shadcn/UI are working correctly."

What file do you want to CREATE or UPDATE?
CREATE src/components/ui/button.jsx
CREATE src/pages/ShadcnTestPage.jsx
UPDATE src/routes.jsx

What are details you want to add to drive the code changes?
Add the Button component using the Shadcn CLI, then create a simple test page that uses both Tailwind CSS classes and the new Button component. Add a route to this page in routes.jsx for testing purposes.