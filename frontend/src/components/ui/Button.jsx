import React from 'react';
import PropTypes from 'prop-types';
import styles from '../../styles/components/Button.module.css';

/**
 * Button component with multiple variants, sizes, and states
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
  
  // Build the class list based on props
  const classes = [
    styles.button,
    styles[size],
    styles[variant],
    outline ? styles.outline : '',
    text ? styles.text : '',
    fullWidth ? styles.fullWidth : '',
    loading ? styles.loading : '',
    (iconLeft || iconRight) ? styles.withIcon : '',
    className
  ].filter(Boolean).join(' ');

  return (
    <button
      ref={ref}
      type={type}
      className={classes}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {iconLeft && <span className={styles.iconLeft}>{iconLeft}</span>}
      {children}
      {iconRight && <span className={styles.iconRight}>{iconRight}</span>}
    </button>
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