# Specification: Implement Horizontal Layout

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Modify the PlaygroundLayout to place code editor next to the question panel instead of under it

## Mid-Level Objective

- Update the PlaygroundLayout component to use a horizontal split for question and code panels
- Maintain resizable panels using react-split
- Adjust default sizes and min-sizes appropriately
- Preserve existing functionality while changing layout direction
- Apply Tailwind classes for styling the modified layout

## Implementation Notes
- The current implementation uses a vertical split for question and code panels
- The layout should be changed to use a horizontal split instead
- Continue using react-split for resizable panels
- Ensure localStorage persistence still works for panel sizes
- Make sure all panels maintain proper sizing and overflow behavior
- Use Tailwind CSS for all new styling

## Context

### Beginning context
- src/components/CodePlayground/PlaygroundLayout.jsx - Current playground layout
- src/styles/playground.css - Current playground styles

### Ending context
- Updated PlaygroundLayout.jsx with horizontal question/editor layout
- Updated or removed playground.css if replaced with Tailwind

## Low-Level Tasks
> Ordered from start to finish

1. Update PlaygroundLayout structure for horizontal question/editor layout

What prompt would you run to complete this task?
"Update the PlaygroundLayout.jsx component to modify the left section so that the QuestionPanel and CodePanel are placed side by side horizontally instead of vertically stacked. Maintain the react-split functionality."

What file do you want to UPDATE?
UPDATE src/components/CodePlayground/PlaygroundLayout.jsx

What are details you want to add to drive the code changes?
Change the Split component direction in the left section from "vertical" to "horizontal". Update the state variable names and localStorage keys to reflect this change. Adjust default sizes (e.g., 40% question, 60% editor) and min-sizes appropriately for horizontal layout.

2. Apply Tailwind CSS to the playground layout structure

What prompt would you run to complete this task?
"Convert the PlaygroundLayout component's CSS classes to Tailwind CSS utility classes. Focus on the container elements, maintaining the same layout behavior while updating the styling approach."

What file do you want to UPDATE?
UPDATE src/components/CodePlayground/PlaygroundLayout.jsx

What are details you want to add to drive the code changes?
Replace custom CSS classes (like "playground-container", "left-section", etc.) with Tailwind utility classes. Ensure proper height, width, overflow, and flex properties are maintained. Use Tailwind's flex and sizing utilities to achieve the same layout structure.

3. Update Split component configuration and gutter styling

What prompt would you run to complete this task?
"Update the react-split configuration to optimize for the new horizontal layout in the left panel. Also, update the gutter styling to use Tailwind instead of CSS classes."

What file do you want to UPDATE?
UPDATE src/components/CodePlayground/PlaygroundLayout.jsx
UPDATE src/styles/globals.css (if needed for gutter styling)

What are details you want to add to drive the code changes?
Adjust minSize, gutterSize, and other Split component properties to work well with the horizontal layout. Create Tailwind-based styling for split gutters, either directly in the component or using @apply directives in globals.css for reusability.

4. Update panel overflow and content styling

What prompt would you run to complete this task?
"Update the QuestionPanel and CodePanel components to handle the new horizontal layout properly, ensuring content overflow is handled correctly and that both panels use the available space effectively."

What file do you want to UPDATE?
UPDATE src/components/CodePlayground/QuestionPanel.jsx
UPDATE src/components/CodePlayground/CodePanel.jsx

What are details you want to add to drive the code changes?
Ensure both panels have proper overflow handling (overflow-auto or overflow-scroll) with Tailwind. Make sure content remains readable and accessible in the new layout. Adjust padding, margins, and content alignment for the horizontal arrangement.

5. Test and optimize responsive behavior

What prompt would you run to complete this task?
"Test the new horizontal layout at different viewport sizes and add responsive behavior to switch back to vertical layout on smaller screens if necessary. Optimize min-sizes and default sizes for different screen widths."

What file do you want to UPDATE?
UPDATE src/components/CodePlayground/PlaygroundLayout.jsx

What are details you want to add to drive the code changes?
Add responsive logic to detect screen width and switch layout direction if the screen is too narrow. Use Tailwind's responsive modifiers to adjust classes based on viewport size. Consider adding a custom hook for responsive layout switching if complex logic is needed.