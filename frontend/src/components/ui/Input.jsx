import React from 'react';
import PropTypes from 'prop-types';
import styles from '../../styles/components/Input.module.css';

/**
 * Input component for form controls
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

  // Build classes based on props
  const formControlClasses = [
    styles.formControl,
    error ? styles.error : '',
    success ? styles.success : '',
    label ? styles.withLabel : '',
    iconLeft ? styles.withIconLeft : '',
    iconRight ? styles.withIconRight : '',
    className
  ].filter(Boolean).join(' ');

  const inputClasses = [
    styles.input,
    styles[size]
  ].filter(Boolean).join(' ');

  return (
    <div className={formControlClasses}>
      {label && (
        <label className={styles.label} htmlFor={inputId}>
          {label} {required && <span className={styles.required}>*</span>}
        </label>
      )}
      
      <div className={styles.inputWrapper}>
        {iconLeft && <div className={styles.iconLeft}>{iconLeft}</div>}
        
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
        
        {iconRight && <div className={styles.iconRight}>{iconRight}</div>}
      </div>
      
      {helperText && (
        <div 
          id={`${inputId}-helper`}
          className={styles.helperText}
        >
          {helperText}
        </div>
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