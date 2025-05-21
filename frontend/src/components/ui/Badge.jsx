import React from 'react';
import PropTypes from 'prop-types';
import styles from '../../styles/components/Badge.module.css';

/**
 * Badge component for status indicators and counters
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
  
  // Build badge classes based on props
  const classes = [
    styles.badge,
    styles[variant],
    styles[size],
    styles[shape],
    outline ? styles.outline : '',
    dot ? styles.dot : '',
    (iconLeft || iconRight) ? styles.withIcon : '',
    className
  ].filter(Boolean).join(' ');

  // If it's a dot badge, don't render children
  if (dot) {
    return <span className={classes} {...props} />;
  }

  return (
    <span className={classes} {...props}>
      {iconLeft && <span className={styles.iconLeft}>{iconLeft}</span>}
      {children}
      {iconRight && <span className={styles.iconRight}>{iconRight}</span>}
    </span>
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