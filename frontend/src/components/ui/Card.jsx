import React from 'react';
import PropTypes from 'prop-types';
import styles from '../../styles/components/Card.module.css';

/**
 * Card component for displaying content in a contained format
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
  
  // Build card classes based on props
  const cardClasses = [
    styles.card, 
    styles[variant], 
    styles[size],
    interactive ? styles.interactive : '',
    className
  ].filter(Boolean).join(' ');

  return (
    <div 
      className={cardClasses} 
      onClick={interactive ? onClick : undefined}
      {...props}
    >
      {(title || subtitle) && (
        <div className={styles.header}>
          <div>
            {title && <h3 className={styles.title}>{title}</h3>}
            {subtitle && <div className={styles.subtitle}>{subtitle}</div>}
          </div>
        </div>
      )}
      {children}
    </div>
  );
};

Card.Body = ({ children, className = '', ...props }) => {
  const bodyClasses = [styles.body, className].filter(Boolean).join(' ');
  return (
    <div className={bodyClasses} {...props}>
      {children}
    </div>
  );
};

Card.Footer = ({ children, className = '', ...props }) => {
  const footerClasses = [styles.footer, className].filter(Boolean).join(' ');
  return (
    <div className={footerClasses} {...props}>
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