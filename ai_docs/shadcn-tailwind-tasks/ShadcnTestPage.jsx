import React from 'react';
import { Button } from '../components/ui/button';

/**
 * Test page to demonstrate Tailwind CSS and Shadcn/UI components
 */
const ShadcnTestPage = () => {
  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Shadcn/UI + Tailwind CSS Test Page</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-10">
        <div className="bg-card rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Tailwind CSS Test</h2>
          <p className="text-foreground mb-4">
            This text uses Tailwind's text colors based on our theme variables.
          </p>
          <div className="flex flex-wrap gap-2 mb-4">
            <div className="w-20 h-20 bg-primary rounded-md flex items-center justify-center text-white">Primary</div>
            <div className="w-20 h-20 bg-secondary rounded-md flex items-center justify-center text-white">Secondary</div>
            <div className="w-20 h-20 bg-accent rounded-md flex items-center justify-center">Accent</div>
            <div className="w-20 h-20 bg-destructive rounded-md flex items-center justify-center text-white">Destructive</div>
          </div>
          <p className="text-sm text-muted-foreground">
            The above colors should match our application theme.
          </p>
        </div>
        
        <div className="bg-card rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Shadcn/UI Button Test</h2>
          <div className="grid grid-cols-2 gap-4">
            <Button>Default Button</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="destructive">Destructive</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="link">Link</Button>
          </div>
          
          <h3 className="text-lg font-medium mt-6 mb-3">Button Sizes</h3>
          <div className="flex flex-wrap gap-4">
            <Button size="sm">Small</Button>
            <Button>Default</Button>
            <Button size="lg">Large</Button>
          </div>
        </div>
      </div>
      
      <div className="p-6 border border-border rounded-lg bg-muted">
        <h2 className="text-xl font-semibold mb-3">Theme Testing</h2>
        <p className="mb-4">
          The components above should respect the current theme settings and change appropriately when the theme changes.
        </p>
        <Button variant="outline" className="border-dashed">
          Toggle Theme Mode
        </Button>
      </div>
    </div>
  );
};

export default ShadcnTestPage;