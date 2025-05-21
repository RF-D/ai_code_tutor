import React from 'react';
import PropTypes from 'prop-types';
import { Input as ShadcnInput } from "@/components/ui/input"; // This path assumes shadcn components are installed in the @/components/ui directory
import { Label } from "@/components/ui/label"; // You'll need to add the Label component with: npx shadcn-ui@latest add label
import { cn } from "@/lib/utils"; // This is a utility function provided by shadcn for class name merging

/**
 * Input component integrating shadcn/ui while maintaining the original API
 */
const Input = React.forwardRef(({
  id,
  name,
  type = 'text',
  label,
  value,
  onChange,
  onBlur,
  placeholder,
  disabled = false,
  readOnly = false,
  required = false,
  error,
  success,
  helperText,
  size = 'md',
  iconLeft,
  iconRight,
  className = '',
  ...props
}, ref) => {
  
  // Generate a unique ID if not provided
  const inputId = id || `input-${name}-${Math.random().toString(36).substr(2, 9)}`;

  // Map our sizes to tailwind classes
  const sizeClasses = {
    sm: "h-8 text-sm",
    md: "h-10",
    lg: "h-12 text-lg",
  };

  // Build the wrapper and input classes
  const wrapperClasses = cn(
    "space-y-2",
    className
  );

  const inputClasses = cn(
    sizeClasses[size] || sizeClasses.md,
    error && "border-destructive",
    success && "border-success",
    (iconLeft || iconRight) && "pl-10", // Add padding for icon
    className
  );

  const helperTextClasses = cn(
    "text-sm mt-1",
    error ? "text-destructive" : "text-muted-foreground"
  );

  return (
    <div className={wrapperClasses}>
      {label && (
        <Label 
          htmlFor={inputId}
          className={cn(required && "after:content-['*'] after:ml-0.5 after:text-destructive")}
        >
          {label}
        </Label>
      )}
      
      <div className="relative">
        {iconLeft && (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground">
            {iconLeft}
          </div>
        )}
        
        <ShadcnInput
          ref={ref}
          id={inputId}
          name={name}
          type={type}
          className={inputClasses}
          value={value}
          onChange={onChange}
          onBlur={onBlur}
          placeholder={placeholder}
          disabled={disabled}
          readOnly={readOnly}
          required={required}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={helperText ? `${inputId}-helper` : undefined}
          {...props}
        />
        
        {iconRight && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground">
            {iconRight}
          </div>
        )}
      </div>
      
      {helperText && (
        <p 
          id={`${inputId}-helper`}
          className={helperTextClasses}
        >
          {helperText}
        </p>
      )}
    </div>
  );
});

Input.displayName = 'Input';

Input.propTypes = {
  /** Input id - will be auto-generated if not provided */
  id: PropTypes.string,
  /** Input name */
  name: PropTypes.string.isRequired,
  /** Input type */
  type: PropTypes.oneOf([
    'text', 'email', 'password', 'number', 'tel', 'url', 'search',
    'date', 'time', 'datetime-local', 'month', 'week', 'color'
  ]),
  /** Input label */
  label: PropTypes.node,
  /** Input value */
  value: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  /** Change event handler */
  onChange: PropTypes.func,
  /** Blur event handler */
  onBlur: PropTypes.func,
  /** Input placeholder */
  placeholder: PropTypes.string,
  /** Whether the input is disabled */
  disabled: PropTypes.bool,
  /** Whether the input is read-only */
  readOnly: PropTypes.bool,
  /** Whether the input is required */
  required: PropTypes.bool,
  /** Error message or state */
  error: PropTypes.oneOfType([PropTypes.string, PropTypes.bool]),
  /** Success state */
  success: PropTypes.bool,
  /** Helper text to display below the input */
  helperText: PropTypes.node,
  /** Input size */
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  /** Left icon component */
  iconLeft: PropTypes.node,
  /** Right icon component */
  iconRight: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

export default Input;