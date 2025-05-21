# Specification: Integrate Shadcn Components

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Replace current UI components with Shadcn/UI components for a consistent design system

## Mid-Level Objective

- Replace Button.jsx with Shadcn Button component
- Replace Input.jsx with Shadcn Input component
- Replace Card.jsx with Shadcn Card component
- Replace Modal.jsx with Shadcn Dialog component
- Replace Badge.jsx with Shadcn Badge component
- Update any components that use these UI elements

## Implementation Notes
- Shadcn/UI components should be added via the Shadcn CLI
- Maintain the same functionality and behavior as the original components
- Update prop interfaces to match Shadcn/UI component requirements
- Use Tailwind CSS for additional styling needs
- Maintain theme compatibility with dark/light mode
- Update import statements in components that use these UI elements

## Context

### Beginning context
- src/components/ui/ - Current UI components
- src/components/ui/Button.jsx
- src/components/ui/Input.jsx
- src/components/ui/Card.jsx
- src/components/ui/Modal.jsx
- src/components/ui/Badge.jsx
- src/styles/components/ - CSS modules for current UI components

### Ending context
- src/components/ui/ - Updated with Shadcn components
- src/components/ui/button.jsx (Shadcn naming convention uses lowercase)
- src/components/ui/input.jsx
- src/components/ui/card.jsx
- src/components/ui/dialog.jsx (replaces Modal)
- src/components/ui/badge.jsx
- Updated imports in components that use these UI elements

## Low-Level Tasks
> Ordered from start to finish

1. Add Shadcn Button component and update existing Button usage

What prompt would you run to complete this task?
"Add the Shadcn Button component using the Shadcn CLI, analyze the current Button.jsx implementation, and update components that use Button to use the new Shadcn version."

What file do you want to CREATE or UPDATE?
CREATE src/components/ui/button.jsx
UPDATE components that import Button.jsx

What are details you want to add to drive the code changes?
Use the Shadcn CLI to add the button component. Compare the props API between the original Button and Shadcn Button, and create adapter functions or wrapper components if necessary to maintain compatibility. Update import paths in files that use Button.

2. Add Shadcn Input component and update existing Input usage

What prompt would you run to complete this task?
"Add the Shadcn Input component using the Shadcn CLI, analyze the current Input.jsx implementation, and update components that use Input to use the new Shadcn version."

What file do you want to CREATE or UPDATE?
CREATE src/components/ui/input.jsx
UPDATE components that import Input.jsx

What are details you want to add to drive the code changes?
Use the Shadcn CLI to add the input component. Compare the props API between the original Input and Shadcn Input, and create adapter functions or wrapper components if necessary to maintain compatibility. Update import paths in files that use Input.

3. Add Shadcn Card component and update existing Card usage

What prompt would you run to complete this task?
"Add the Shadcn Card component using the Shadcn CLI, analyze the current Card.jsx implementation, and update components that use Card to use the new Shadcn version."

What file do you want to CREATE or UPDATE?
CREATE src/components/ui/card.jsx
UPDATE components that import Card.jsx

What are details you want to add to drive the code changes?
Use the Shadcn CLI to add the card component (including CardHeader, CardContent, etc.). Compare the props API between the original Card and Shadcn Card, and create adapter functions or wrapper components if necessary to maintain compatibility. Update import paths in files that use Card.

4. Add Shadcn Dialog component and update existing Modal usage

What prompt would you run to complete this task?
"Add the Shadcn Dialog component using the Shadcn CLI, analyze the current Modal.jsx implementation, and update components that use Modal to use the new Shadcn Dialog component."

What file do you want to CREATE or UPDATE?
CREATE src/components/ui/dialog.jsx
UPDATE components that import Modal.jsx

What are details you want to add to drive the code changes?
Use the Shadcn CLI to add the dialog component. Compare the props API between the original Modal and Shadcn Dialog, and create adapter functions or wrapper components if necessary to maintain compatibility. Update import paths and component usage in files that use Modal.

5. Add Shadcn Badge component and update existing Badge usage

What prompt would you run to complete this task?
"Add the Shadcn Badge component using the Shadcn CLI, analyze the current Badge.jsx implementation, and update components that use Badge to use the new Shadcn version."

What file do you want to CREATE or UPDATE?
CREATE src/components/ui/badge.jsx
UPDATE components that import Badge.jsx

What are details you want to add to drive the code changes?
Use the Shadcn CLI to add the badge component. Compare the props API between the original Badge and Shadcn Badge, and create adapter functions or wrapper components if necessary to maintain compatibility. Update import paths in files that use Badge.