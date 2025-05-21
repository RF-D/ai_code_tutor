import React from "react";
import PropTypes from 'prop-types';
import { cn } from "../../lib/utils";

/**
 * Card component using Tailwind CSS for styling
 * Maintains the API of the original Card component
 */
const Card = ({
  children,
  title,
  subtitle,
  variant = 'default',
  size = 'md',
  interactive = false,
  onClick,
  className = '',
  ...props
}) => {
  // Map variants to Tailwind classes
  const variantClasses = {
    default: "bg-background-primary border border-border shadow-sm hover:shadow-md",
    flat: "bg-background-primary border border-border",
    elevated: "bg-background-primary border border-border shadow-md hover:shadow-lg",
  };
  
  // Map sizes to Tailwind classes for padding
  const sizeClasses = {
    sm: "p-2",
    md: "p-4",
    lg: "p-6",
  };
  
  // Interactive styles
  const interactiveClasses = interactive 
    ? "cursor-pointer transition-transform hover:-translate-y-1" 
    : "";
  
  return (
    <div 
      className={cn(
        "flex flex-col min-w-0 overflow-hidden rounded-lg",
        "transition-all duration-300",
        variantClasses[variant],
        interactiveClasses,
        className
      )}
      onClick={interactive ? onClick : undefined}
      {...props}
    >
      {(title || subtitle) && (
        <div className={cn(
          "border-b border-border flex items-center justify-between",
          sizeClasses[size]
        )}>
          <div>
            {title && <h3 className="text-xl font-semibold text-text-primary mb-1">{title}</h3>}
            {subtitle && <div className="text-md text-text-secondary">{subtitle}</div>}
          </div>
        </div>
      )}
      {children}
    </div>
  );
};

/**
 * Card.Body component for the main content area
 */
Card.Body = ({ children, className = '', ...props }) => {
  return (
    <div className={cn("flex-1 p-4", className)} {...props}>
      {children}
    </div>
  );
};

/**
 * Card.Footer component for the footer area
 */
Card.Footer = ({ children, className = '', ...props }) => {
  return (
    <div className={cn("border-t border-border p-4", className)} {...props}>
      {children}
    </div>
  );
};

Card.propTypes = {
  /** Card content */
  children: PropTypes.node,
  /** Card title */
  title: PropTypes.node,
  /** Card subtitle */
  subtitle: PropTypes.node,
  /** Card variant */
  variant: PropTypes.oneOf(['default', 'flat', 'elevated']),
  /** Card size */
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  /** Whether the card is interactive (clickable) */
  interactive: PropTypes.bool,
  /** Click event handler (only used when interactive is true) */
  onClick: PropTypes.func,
  /** Additional CSS class */
  className: PropTypes.string,
};

Card.Body.propTypes = {
  /** Card body content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

Card.Footer.propTypes = {
  /** Card footer content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

export default Card;