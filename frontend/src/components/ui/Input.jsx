import * as React from "react";
import PropTypes from 'prop-types';

import { cn } from "../../lib/utils";

/**
 * Input component using Tailwind CSS for styling
 * Maintains the API of the original Input component
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

  // Map size to Tailwind classes
  const sizeClasses = {
    sm: "h-8 px-2 py-1 text-xs",
    md: "h-10 px-3 py-2 text-sm",
    lg: "h-12 px-4 py-3 text-base",
  };

  // Map status to Tailwind classes
  const statusClasses = error 
    ? "border-danger focus:border-danger focus:ring-danger/30" 
    : success
      ? "border-success focus:border-success focus:ring-success/30"
      : "border-input focus:border-primary focus:ring-primary/30";

  // Base input styling with Tailwind
  const inputClasses = cn(
    "flex w-full rounded-md border bg-background text-text-primary",
    "focus:outline-none focus:ring-2 focus:ring-offset-0",
    "disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-muted",
    sizeClasses[size],
    statusClasses,
    iconLeft && "pl-9",
    iconRight && "pr-9",
    className
  );

  return (
    <div className="w-full mb-4">
      {label && (
        <label 
          className={cn(
            "block mb-1 text-sm font-medium",
            error ? "text-danger" : success ? "text-success" : "text-text-secondary"
          )} 
          htmlFor={inputId}
        >
          {label} {required && <span className="text-danger">*</span>}
        </label>
      )}
      
      <div className="relative">
        {iconLeft && (
          <div className="absolute left-3 top-1/2 -translate-y-1/2 text-text-tertiary">
            {iconLeft}
          </div>
        )}
        
        <input
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
          <div className="absolute right-3 top-1/2 -translate-y-1/2 text-text-tertiary">
            {iconRight}
          </div>
        )}
      </div>
      
      {helperText && (
        <div 
          id={`${inputId}-helper`}
          className={cn(
            "mt-1 text-xs",
            error ? "text-danger" : success ? "text-success" : "text-text-tertiary"
          )}
        >
          {helperText}
        </div>
      )}
    </div>
  );
});

Input.displayName = "Input";

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