# Specification: Theme Configuration

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Configure light/dark theme support with Tailwind and Shadcn/UI that matches the current theming system

## Mid-Level Objective

- Implement a theme provider that works with both Tailwind and Shadcn/UI
- Configure Tailwind CSS to use the application's theme color variables
- Ensure theme switching works correctly with both systems
- Maintain compatibility with the existing ThemeContext
- Implement persistent theme selection with localStorage

## Implementation Notes
- Use the existing ThemeContext as a foundation
- Shadcn/UI uses a specific theme-provider component
- Tailwind's dark mode should be class-based, not media query based
- CSS variables should be defined in globals.css for use by both systems
- Transition effects should be maintained during theme switching
- Light and dark themes should match the current design system colors

## Context

### Beginning context
- src/context/ThemeContext.jsx - Current theme context implementation
- src/styles/theme.css - Current theme color variables
- tailwind.config.js - Current Tailwind configuration

### Ending context
- Updated ThemeContext.jsx with Shadcn/UI integration
- Updated tailwind.config.js with theme configuration
- Updated globals.css with theme variables for Shadcn/UI
- New components/ui/theme-provider.jsx file for Shadcn/UI theme support

## Low-Level Tasks
> Ordered from start to finish

1. Add Shadcn/UI theme provider component

What prompt would you run to complete this task?
"Add the Shadcn/UI theme provider component and configure it to work with the application's existing theme system. Use the Shadcn CLI to add the theme provider component."

What file do you want to CREATE or UPDATE?
CREATE src/components/ui/theme-provider.jsx

What are details you want to add to drive the code changes?
Use the Shadcn CLI to add the theme-provider component. This component will be used to provide theme context to Shadcn/UI components. Ensure it accepts the theme from the application's ThemeContext.

2. Update Tailwind configuration for theme support

What prompt would you run to complete this task?
"Update the tailwind.config.js file to support dark mode via class instead of media query. Configure the theme colors to use CSS variables that match the current theme system."

What file do you want to UPDATE?
UPDATE tailwind.config.js

What are details you want to add to drive the code changes?
Set darkMode to "class" in the configuration. Define theme colors using CSS variables that match the existing theme.css variables. Configure the Tailwind theme object to reference these variables. Ensure proper fallback values for each variable.

3. Update globals.css with Shadcn/UI theme variables

What prompt would you run to complete this task?
"Update the globals.css file to include the CSS variables required by Shadcn/UI components for both light and dark themes. Match these variables to the existing theme color system."

What file do you want to UPDATE?
UPDATE src/styles/globals.css

What are details you want to add to drive the code changes?
Add the Shadcn/UI CSS variables (--radius, --background, --foreground, etc.) to the :root selector for light theme. Add the same variables with dark theme values to the .dark selector. Ensure values match the current theme colors from theme.css.

4. Integrate ThemeContext with Shadcn Theme Provider

What prompt would you run to complete this task?
"Update the ThemeContext to integrate with the Shadcn/UI theme provider. Maintain the existing functionality while adding support for Shadcn/UI theming."

What file do you want to UPDATE?
UPDATE src/context/ThemeContext.jsx

What are details you want to add to drive the code changes?
Modify the ThemeContext to wrap the Shadcn/UI ThemeProvider around the application. Ensure theme switching logic updates both the data-theme attribute and the dark class for Tailwind. Maintain localStorage persistence for theme preference.

5. Test theme switching with Shadcn/UI components

What prompt would you run to complete this task?
"Create a simple test component that uses both Tailwind classes and Shadcn/UI components to verify that theme switching works correctly with both systems."

What file do you want to CREATE or UPDATE?
CREATE src/components/ThemeTest.jsx
UPDATE src/pages/SettingsPage.jsx (to add the test component)

What are details you want to add to drive the code changes?
Create a component that displays various UI elements using both Tailwind classes and Shadcn/UI components. Include text, buttons, cards, and other elements that should change appearance when the theme changes. Add this component to the settings page for testing.