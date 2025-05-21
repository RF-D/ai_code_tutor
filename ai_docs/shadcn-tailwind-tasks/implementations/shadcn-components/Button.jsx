import React from 'react';
import PropTypes from 'prop-types';
import { Button as ShadcnButton } from "@/components/ui/button"; // This path assumes shadcn components are installed in the @/components/ui directory
import { cn } from "@/lib/utils"; // This is a utility function provided by shadcn for class name merging

/**
 * Button component with shadcn/ui integration while maintaining the original API
 */
const Button = React.forwardRef(({
  children,
  variant = 'primary',
  size = 'md',
  type = 'button',
  outline = false,
  text = false,
  disabled = false,
  loading = false,
  fullWidth = false,
  onClick,
  className = '',
  iconLeft,
  iconRight,
  ...props
}, ref) => {
  
  // Map our variants to shadcn variants
  const variantMap = {
    primary: 'default',
    secondary: 'secondary',
    success: 'success',
    danger: 'destructive',
  };

  // Map our sizes to shadcn sizes
  const sizeMap = {
    sm: 'sm',
    md: 'default',
    lg: 'lg',
  };

  // Convert text prop to variant
  let shadcnVariant = variantMap[variant] || 'default';
  if (text) {
    shadcnVariant = 'ghost';
  } else if (outline) {
    shadcnVariant = 'outline';
  }

  // Convert our size to shadcn size
  const shadcnSize = sizeMap[size] || 'default';

  return (
    <ShadcnButton
      ref={ref}
      type={type}
      variant={shadcnVariant}
      size={shadcnSize}
      disabled={disabled || loading}
      onClick={onClick}
      className={cn(
        fullWidth && "w-full",
        loading && "opacity-70 cursor-progress",
        className
      )}
      {...props}
    >
      {iconLeft && <span className="mr-2">{iconLeft}</span>}
      {children}
      {iconRight && <span className="ml-2">{iconRight}</span>}
    </ShadcnButton>
  );
});

Button.displayName = 'Button';

Button.propTypes = {
  /** Button content */
  children: PropTypes.node.isRequired,
  /** Button variants */
  variant: PropTypes.oneOf(['primary', 'secondary', 'success', 'danger']),
  /** Button sizes */
  size: PropTypes.oneOf(['sm', 'md', 'lg']),
  /** Button HTML type */
  type: PropTypes.oneOf(['button', 'submit', 'reset']),
  /** Whether the button should have an outline style */
  outline: PropTypes.bool,
  /** Whether the button should have a text style (no background) */
  text: PropTypes.bool,
  /** Whether the button is disabled */
  disabled: PropTypes.bool,
  /** Whether the button is in loading state */
  loading: PropTypes.bool,
  /** Whether the button should take up full width */
  fullWidth: PropTypes.bool,
  /** Click event handler */
  onClick: PropTypes.func,
  /** Additional CSS class */
  className: PropTypes.string,
  /** Left icon component */
  iconLeft: PropTypes.node,
  /** Right icon component */
  iconRight: PropTypes.node,
};

export default Button;