# Shadcn UI Component Implementation Guide

This guide provides instructions for integrating the Shadcn UI components into the existing AI Code Tutor codebase. The implementation ensures that the new components maintain the same API as the original components to minimize refactoring efforts.

## Prerequisites

Before beginning the implementation, make sure you have:

1. Installed Tailwind CSS:
```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

2. Configured Tailwind CSS in your `tailwind.config.js`:
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
    './app/**/*.{js,jsx}',
    './src/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        success: {
          DEFAULT: "hsl(var(--success))",
          foreground: "hsl(var(--success-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
```

3. Added CSS variables to your global CSS file:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
 
@layer base {
  :root {
    --background: 0 0% 100%;
    --foreground: 222.2 84% 4.9%;
 
    --card: 0 0% 100%;
    --card-foreground: 222.2 84% 4.9%;
 
    --popover: 0 0% 100%;
    --popover-foreground: 222.2 84% 4.9%;
 
    --primary: 222.2 47.4% 11.2%;
    --primary-foreground: 210 40% 98%;
 
    --secondary: 210 40% 96.1%;
    --secondary-foreground: 222.2 47.4% 11.2%;
 
    --muted: 210 40% 96.1%;
    --muted-foreground: 215.4 16.3% 46.9%;
 
    --accent: 210 40% 96.1%;
    --accent-foreground: 222.2 47.4% 11.2%;
 
    --destructive: 0 84.2% 60.2%;
    --destructive-foreground: 210 40% 98%;
 
    --success: 142.1 76.2% 36.3%;
    --success-foreground: 355.7 100% 97.3%;
 
    --border: 214.3 31.8% 91.4%;
    --input: 214.3 31.8% 91.4%;
    --ring: 222.2 84% 4.9%;
 
    --radius: 0.5rem;
  }
 
  .dark {
    --background: 222.2 84% 4.9%;
    --foreground: 210 40% 98%;
 
    --card: 222.2 84% 4.9%;
    --card-foreground: 210 40% 98%;
 
    --popover: 222.2 84% 4.9%;
    --popover-foreground: 210 40% 98%;
 
    --primary: 210 40% 98%;
    --primary-foreground: 222.2 47.4% 11.2%;
 
    --secondary: 217.2 32.6% 17.5%;
    --secondary-foreground: 210 40% 98%;
 
    --muted: 217.2 32.6% 17.5%;
    --muted-foreground: 215 20.2% 65.1%;
 
    --accent: 217.2 32.6% 17.5%;
    --accent-foreground: 210 40% 98%;
 
    --destructive: 0 62.8% 30.6%;
    --destructive-foreground: 210 40% 98%;
 
    --success: 142.1 70.6% 45.3%;
    --success-foreground: 144.9 80.4% 10%;
 
    --border: 217.2 32.6% 17.5%;
    --input: 217.2 32.6% 17.5%;
    --ring: 212.7 26.8% 83.9%;
  }
}
 
@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground;
  }
}
```

4. Created a `utils.js` file to help with class name merging:
```javascript
// lib/utils.js
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"
 
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}
```

5. Installed necessary dependencies:
```bash
npm install clsx tailwind-merge lucide-react
```

## Installation Steps

### Step 1: Install Shadcn UI CLI and Components

Install the Shadcn UI CLI tool:
```bash
npm install -g shadcn-ui
```

Then, add the components one by one:
```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add input
npx shadcn-ui@latest add label
npx shadcn-ui@latest add card
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add badge
```

### Step 2: Copy Implementation Files

Copy the implementation files from this directory to your project's component directory:

```bash
cp Button.jsx Input.jsx Card.jsx Dialog.jsx Badge.jsx /Users/rauldiaz/AI/ai_code_tutor/frontend/src/components/ui/
```

### Step 3: Update Component Imports

Update imports throughout your application to use the new components:

#### Before:
```jsx
import Button from './components/ui/Button';
import Input from './components/ui/Input';
import Card from './components/ui/Card';
import Modal from './components/ui/Modal';
import Badge from './components/ui/Badge';
```

#### After:
```jsx
import Button from './components/ui/Button';
import Input from './components/ui/Input';
import Card from './components/ui/Card';
import Dialog from './components/ui/Dialog'; // Note the name change
import Badge from './components/ui/Badge';
```

> **Note:** The only component that has changed its name is `Modal`, which is now `Dialog`. You'll need to update all references to `Modal` in your code.

## Component-Specific Migration Notes

### Button Component

The Shadcn Button component supports the following variants:
- default (primary)
- secondary
- destructive (danger)
- outline
- ghost (text)

Make sure to customize the theme to add the 'success' variant if needed.

### Input Component

The Shadcn Input component works differently from your current implementation:
- It doesn't have built-in label and helper text support
- Our wrapper adds these features while maintaining the same API
- You'll need to install the Label component separately

### Card Component

The Shadcn Card component is structurally different:
- Our wrapper maintains the Card.Body and Card.Footer API
- Shadcn has CardHeader, CardTitle, CardDescription, CardContent, and CardFooter
- We map these accordingly to maintain the original usage patterns

### Dialog Component (formerly Modal)

The Shadcn Dialog component:
- Works similarly to the Modal component but with different naming
- Has slightly different behavior for backdrop and escape key handling
- Our wrapper handles these differences to maintain the same behavior

### Badge Component

The Shadcn Badge component:
- Has limited variants by default
- Our wrapper adds all the original variants and maps them appropriately
- The dot feature is implemented with custom styling

## Testing Your Implementation

After implementing these changes, thoroughly test each component to ensure they work as expected:

1. Test all variants, sizes, and states
2. Check for any styling inconsistencies
3. Verify that the component API behaves the same way as before
4. Test in both light and dark modes

## Additional Customization

You may need to add custom variants or styles to match your current design system:

1. Extend the Tailwind theme in `tailwind.config.js`
2. Add custom classes to the component implementations
3. Update the component's CSS to match your design requirements

## Troubleshooting

Common issues you might encounter:

1. **Missing dependencies**: Make sure you've installed all required packages
2. **Path issues**: Adjust the import paths in the components to match your project structure
3. **Styling conflicts**: If styles don't appear correctly, check for CSS specificity issues
4. **Type errors**: If using TypeScript, you may need to add proper type definitions

## Next Steps

After successfully implementing these components:

1. Consider implementing additional Shadcn UI components like Select, Checkbox, etc.
2. Refine your theme to better match your design system
3. Document any component API changes for your team
4. Consider adding unit tests for the new components