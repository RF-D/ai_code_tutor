import * as React from "react";
import { cva } from "class-variance-authority";
import PropTypes from 'prop-types';

import { cn } from "../../lib/utils";

/**
 * Button component variants using class-variance-authority
 * Combines Shadcn/UI styling with our app's existing button API
 */
const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        primary: "bg-primary text-primary-foreground hover:bg-primary-dark",
        secondary: "bg-secondary text-secondary-foreground hover:bg-secondary-dark",
        success: "bg-success text-white hover:bg-success/90",
        danger: "bg-danger text-white hover:bg-danger/90",
        outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
        text: "bg-transparent underline-offset-4 hover:bg-muted/20",
      },
      size: {
        sm: "h-8 px-2 py-1 text-xs",
        md: "h-10 px-4 py-2 text-sm",
        lg: "h-12 px-6 py-3 text-base",
      },
      buttonStyle: {
        default: "",
        outline: "bg-transparent border",
        text: "bg-transparent border-transparent",
      },
      fullWidth: {
        true: "w-full",
        false: "",
      },
    },
    compoundVariants: [
      {
        variant: "primary",
        buttonStyle: "outline",
        className: "text-primary border-primary hover:bg-primary/10",
      },
      {
        variant: "secondary",
        buttonStyle: "outline",
        className: "text-secondary border-secondary hover:bg-secondary/10",
      },
      {
        variant: "success",
        buttonStyle: "outline",
        className: "text-success border-success hover:bg-success/10",
      },
      {
        variant: "danger",
        buttonStyle: "outline",
        className: "text-danger border-danger hover:bg-danger/10",
      },
      {
        variant: "primary",
        buttonStyle: "text",
        className: "text-primary hover:bg-primary/10",
      },
      {
        variant: "secondary",
        buttonStyle: "text",
        className: "text-secondary hover:bg-secondary/10",
      },
      {
        variant: "success",
        buttonStyle: "text",
        className: "text-success hover:bg-success/10",
      },
      {
        variant: "danger",
        buttonStyle: "text",
        className: "text-danger hover:bg-danger/10",
      },
    ],
    defaultVariants: {
      variant: "primary",
      size: "md",
      buttonStyle: "default",
      fullWidth: false,
    },
  }
);

/**
 * Button component with multiple variants, sizes, and states
 * This is an adapter between our existing Button API and Shadcn/UI button
 */
const Button = React.memo(React.forwardRef(({
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
  // Convert our API to Shadcn/UI style API
  const buttonStyle = outline ? "outline" : text ? "text" : "default";
  
  // Create loading indicator
  const loadingIndicator = loading ? (
    <div className="absolute inline-flex h-4 w-4 animate-spin rounded-full border-2 border-solid border-current border-r-transparent motion-reduce:animate-[spin_1.5s_linear_infinite]"></div>
  ) : null;
  
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        buttonVariants({ 
          variant, 
          size, 
          buttonStyle, 
          fullWidth, 
          className 
        }),
        loading && "relative text-transparent",
      )}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {iconLeft && <span className="mr-2">{iconLeft}</span>}
      {children}
      {iconRight && <span className="ml-2">{iconRight}</span>}
      {loadingIndicator}
    </button>
  );
}), (prevProps, nextProps) => {
  // Only re-render when these props change
  return (
    prevProps.variant === nextProps.variant &&
    prevProps.size === nextProps.size &&
    prevProps.disabled === nextProps.disabled &&
    prevProps.loading === nextProps.loading &&
    prevProps.fullWidth === nextProps.fullWidth &&
    prevProps.outline === nextProps.outline &&
    prevProps.text === nextProps.text
  );
});

Button.displayName = "Button";

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
export { buttonVariants };