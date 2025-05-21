import React from 'react';
import PropTypes from 'prop-types';
import { Badge as ShadcnBadge } from "@/components/ui/badge"; // This path assumes shadcn components are installed in the @/components/ui directory
import { cn } from "@/lib/utils"; // This is a utility function provided by shadcn for class name merging

/**
 * Badge component integrating shadcn/ui while maintaining the original API
 */
const Badge = ({
  children,
  variant = 'primary',
  size = 'md',
  shape = 'rounded',
  outline = false,
  dot = false,
  iconLeft,
  iconRight,
  className = '',
  ...props
}) => {
  
  // Map our variants to shadcn variants
  const variantMap = {
    primary: 'default',
    secondary: 'secondary',
    success: 'success',
    info: 'info',
    warning: 'warning',
    danger: 'destructive',
    light: 'outline',
    dark: 'default',
  };

  // Map our sizes to tailwind classes
  const sizeClasses = {
    sm: "h-5 text-xs px-1.5",
    md: "h-6 text-xs px-2.5",
    lg: "h-7 text-sm px-3",
  };

  // Map our shapes to tailwind classes
  const shapeClasses = {
    rounded: "rounded",
    square: "rounded-none",
    pill: "rounded-full",
  };

  // Determine the shadcn variant based on our props
  let shadcnVariant = variantMap[variant] || 'default';
  if (outline) {
    shadcnVariant = 'outline';
  }

  // Build the badge classes
  const badgeClasses = cn(
    sizeClasses[size] || sizeClasses.md,
    shapeClasses[shape] || shapeClasses.rounded,
    dot && "w-2 h-2 p-0 rounded-full",
    (iconLeft || iconRight) && "flex items-center gap-1",
    className
  );

  // If it's a dot badge, don't render children
  if (dot) {
    return <ShadcnBadge variant={shadcnVariant} className={badgeClasses} {...props} />;
  }

  return (
    <ShadcnBadge variant={shadcnVariant} className={badgeClasses} {...props}>
      {iconLeft && <span className="mr-1">{iconLeft}</span>}
      {children}
      {iconRight && <span className="ml-1">{iconRight}</span>}
    </ShadcnBadge>
  );
};

Badge.propTypes = {
  /** Badge content */
  children: PropTypes.node,
  /** Badge variant */
  variant: PropTypes.oneOf([
    'primary', 'secondary', 'success', 
    'info', 'warning', 'danger', 
    'light', 'dark'
  ]),
  /** Badge size */
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  /** Badge shape */
  shape: PropTypes.oneOf(['rounded', 'square', 'pill']),
  /** Whether the badge should have an outline style */
  outline: PropTypes.bool,
  /** Whether the badge should be rendered as a dot */
  dot: PropTypes.bool,
  /** Left icon component */
  iconLeft: PropTypes.node,
  /** Right icon component */
  iconRight: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

export default Badge;