import React from 'react';
import PropTypes from 'prop-types';
import {
  Card as ShadcnCard,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card"; // This path assumes shadcn components are installed in the @/components/ui directory
import { cn } from "@/lib/utils"; // This is a utility function provided by shadcn for class name merging

/**
 * Card component integrating shadcn/ui while maintaining the original API
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
  
  // Map our variants to tailwind classes
  const variantClasses = {
    default: "",
    flat: "border-none shadow-none bg-background/50",
    elevated: "shadow-lg",
  };

  // Map our sizes to tailwind classes
  const sizeClasses = {
    sm: "p-3",
    md: "p-4",
    lg: "p-6",
  };

  // Build card classes based on props
  const cardClasses = cn(
    variantClasses[variant],
    interactive && "cursor-pointer hover:shadow-md transition-shadow",
    className
  );

  const headerContentClass = sizeClasses[size] || sizeClasses.md;

  // Check if there's a header to render
  const hasHeader = title || subtitle;

  return (
    <ShadcnCard 
      className={cardClasses} 
      onClick={interactive ? onClick : undefined}
      {...props}
    >
      {hasHeader && (
        <CardHeader className={headerContentClass}>
          {title && <CardTitle>{title}</CardTitle>}
          {subtitle && <CardDescription>{subtitle}</CardDescription>}
        </CardHeader>
      )}
      {children}
    </ShadcnCard>
  );
};

// Create Body component to match the original API
Card.Body = ({ children, className = '', ...props }) => {
  return (
    <CardContent className={className} {...props}>
      {children}
    </CardContent>
  );
};

// Create Footer component to match the original API
Card.Footer = ({ children, className = '', ...props }) => {
  return (
    <CardFooter className={className} {...props}>
      {children}
    </CardFooter>
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