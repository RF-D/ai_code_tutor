# Horizontal Layout Implementation Guide

This guide explains how to implement the horizontal layout for the PlaygroundLayout component using Tailwind CSS.

## Overview of Changes

The implementation changes the left panel layout from vertical (question above code) to horizontal (question beside code). The changes use Tailwind CSS for styling instead of the traditional CSS classes, and implements responsive behavior for smaller screens.

## Key Changes

1. **Layout Direction Change**: 
   - Changed the left section Split direction from `"vertical"` to `"horizontal"`.
   - Updated state variable names from `leftVerticalSizes` to `leftHorizontalSizes`.
   - Updated localStorage keys accordingly.
   - Adjusted default split ratios to 40/60 (question/code) for horizontal layout.

2. **Tailwind CSS Integration**:
   - Replaced all custom CSS classes with Tailwind utility classes.
   - Implemented appropriate dark mode support with dark variants.
   - Added consistent color scheme using Tailwind's slate and blue palettes.

3. **Split Component Configuration**:
   - Updated min-sizes to be appropriate for horizontal layout.
   - Applied Tailwind classes to gutters for consistent styling.
   - Added visual indicators for gutters with hover effects.

4. **Responsive Behavior**:
   - Added screen width detection to switch to vertical layout on small screens.
   - Used a combination of flex direction changes based on screen width.
   - Adjusted minimum sizes for various screen widths.

5. **Panel Content Styling**:
   - Updated overflow handling for all panels to ensure proper scrolling.
   - Adjusted content layout for horizontal arrangement.
   - Ensured all components maintain proper sizing in both layouts.

## Implementation Steps

1. Replace the current PlaygroundLayout.jsx with the Tailwind version.
2. Replace the related panel components with their Tailwind versions:
   - QuestionPanel.jsx
   - CodePanel.jsx
   - AssistantPanel.jsx
   - ResultsPanel.jsx
3. Add the custom react-split gutter styles to your globals.css.
4. Remove or comment out the old CSS from playground.css.

## Testing Guidelines

1. Test the layout at different screen sizes to verify responsive behavior.
2. Verify that the split functionality works correctly in horizontal layout.
3. Test localStorage persistence of panel sizes.
4. Check overflow behavior for all panels, especially with larger content.
5. Ensure proper dark mode support.

## Important Notes

- The horizontal layout is more space-efficient on wider screens.
- The implementation automatically reverts to vertical layout on screens narrower than 768px.
- All gutters maintain consistent styling with the Tailwind theme.
- Dark mode is fully supported with appropriate color adjustments.

## Potential Enhancements

1. Add a user preference to choose between horizontal and vertical layouts.
2. Implement more granular responsive adjustments for different screen sizes.
3. Add transition animations when switching between layout modes.
4. Optimize further for tablets and other mid-size devices.