# Specification: Responsive Layout

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Ensure the application layout is fully responsive and works well on all device sizes using Tailwind's responsive utilities

## Mid-Level Objective

- Implement responsive breakpoints in Tailwind configuration
- Create responsive behavior for the PlaygroundLayout on smaller screens
- Ensure Navigation component works well on mobile devices
- Add responsive utilities for small screen optimization
- Implement a mobile-friendly version of the playground with adaptive panels

## Implementation Notes
- Use Tailwind's responsive modifiers (sm:, md:, lg:, xl:, 2xl:)
- Consider using a sidebar toggle for mobile navigation
- PlaygroundLayout should reconfigure on smaller screens to stack panels vertically
- Add touch-friendly controls for panel resizing on mobile
- Ensure readable font sizes and proper spacing on all device sizes
- Test on various viewport sizes to ensure usability

## Context

### Beginning context
- src/components/CodePlayground/PlaygroundLayout.jsx - Playground layout
- src/components/common/Navigation.jsx - Navigation component
- tailwind.config.js - Tailwind configuration
- Other UI components that need responsive behavior

### Ending context
- Updated PlaygroundLayout.jsx with responsive behavior
- Updated Navigation.jsx with mobile-friendly controls
- Updated tailwind.config.js with responsive breakpoints
- Any new helper components for responsive behavior

## Low-Level Tasks
> Ordered from start to finish

1. Configure Tailwind responsive breakpoints

What prompt would you run to complete this task?
"Update the Tailwind configuration to define responsive breakpoints that match the application's design requirements. Ensure these breakpoints are appropriate for the coding playground's layout needs."

What file do you want to UPDATE?
UPDATE tailwind.config.js

What are details you want to add to drive the code changes?
Define appropriate breakpoints in the theme.screens section of tailwind.config.js. Consider common device sizes: sm (640px), md (768px), lg (1024px), xl (1280px), and 2xl (1536px). Adjust these values if needed to better match the application's layout requirements.

2. Implement responsive Navigation component

What prompt would you run to complete this task?
"Update the Navigation component to be responsive and mobile-friendly using Tailwind's responsive utilities. Implement a mobile menu toggle for small screens."

What file do you want to UPDATE?
UPDATE src/components/common/Navigation.jsx

What are details you want to add to drive the code changes?
Replace existing navigation styles with Tailwind classes including responsive modifiers. For mobile screens, implement a collapsible menu with a hamburger toggle button. Use Tailwind's responsive classes to show/hide elements at different screen sizes.

3. Make PlaygroundLayout responsive for small screens

What prompt would you run to complete this task?
"Update the PlaygroundLayout component to adapt to smaller screens by changing panel arrangement and sizes. Implement a stacked layout for mobile devices while maintaining the horizontal layout for larger screens."

What file do you want to UPDATE?
UPDATE src/components/CodePlayground/PlaygroundLayout.jsx

What are details you want to add to drive the code changes?
Create a responsive layout that changes based on screen size. On smaller screens, stack all panels vertically for better readability. On larger screens, maintain the side-by-side layout. Use Tailwind's responsive utilities and possibly React hooks to detect screen size and adjust the layout accordingly.

4. Implement touch-friendly panel controls

What prompt would you run to complete this task?
"Add touch-friendly controls for panel resizing and navigation on mobile devices. Implement buttons to maximize/minimize panels or switch between them on small screens."

What file do you want to CREATE or UPDATE?
CREATE src/components/common/PanelControls.jsx
UPDATE src/components/CodePlayground/PlaygroundLayout.jsx

What are details you want to add to drive the code changes?
Create a new component for panel controls that provides buttons to expand, collapse, or toggle between panels on mobile devices. Add this component to the PlaygroundLayout and show it only on smaller screens using Tailwind's responsive utilities.

5. Test and optimize for various devices

What prompt would you run to complete this task?
"Add utility functions and styles to optimize the application for various device sizes. Test the layout on different viewports and fix any issues."

What file do you want to CREATE or UPDATE?
CREATE src/hooks/useResponsiveLayout.jsx
UPDATE any components with responsive issues

What are details you want to add to drive the code changes?
Create a custom hook to handle responsive layout logic, provide information about the current viewport size, and determine which layout configuration to use. Test the application at various screen sizes and fix any issues with component sizing, text readability, or interactive elements.