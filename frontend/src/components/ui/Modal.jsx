import React, { useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import ReactDOM from 'react-dom';
import styles from '../../styles/components/Modal.module.css';
import Button from './Button';

/**
 * Modal component for dialog boxes
 */
const Modal = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
  centered = false,
  showCloseButton = true,
  closeOnEsc = true,
  closeOnBackdrop = true,
  className = '',
  backdropClassName = '',
  footer,
  ...props
}) => {
  const modalRef = useRef(null);
  
  // Close on Escape key press
  useEffect(() => {
    const handleEscapeKey = (event) => {
      if (closeOnEsc && event.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscapeKey);
      // Prevent scrolling on the body when modal is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscapeKey);
      // Restore scrolling on the body when modal is closed
      document.body.style.overflow = '';
    };
  }, [isOpen, onClose, closeOnEsc]);

  // Click outside to close
  const handleBackdropClick = (event) => {
    if (closeOnBackdrop && event.target === event.currentTarget) {
      onClose();
    }
  };

  // Build modal classes
  const backdropClasses = [
    styles.modalBackdrop,
    isOpen ? styles.open : '',
    backdropClassName
  ].filter(Boolean).join(' ');

  const modalClasses = [
    styles.modal,
    styles[size],
    centered ? styles.centered : '',
    className
  ].filter(Boolean).join(' ');

  // Don't render anything if the modal is closed
  if (!isOpen) return null;

  // Portal the modal to the body to avoid z-index issues
  return ReactDOM.createPortal(
    <div 
      className={backdropClasses} 
      onClick={handleBackdropClick} 
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? 'modal-title' : undefined}
    >
      <div className={modalClasses} ref={modalRef} {...props}>
        {(title || showCloseButton) && (
          <div className={styles.modalHeader}>
            {title && (
              <h2 className={styles.modalTitle} id="modal-title">
                {title}
              </h2>
            )}
            {showCloseButton && (
              <button
                className={styles.closeButton}
                onClick={onClose}
                type="button"
                aria-label="Close modal"
              >
                &times;
              </button>
            )}
          </div>
        )}
        
        <div className={styles.modalBody}>
          {children}
        </div>
        
        {footer && (
          <div className={styles.modalFooter}>
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body
  );
};

// Convenience components for modal parts
Modal.Header = ({ children, className = '', ...props }) => {
  const headerClasses = [styles.modalHeader, className].filter(Boolean).join(' ');
  return (
    <div className={headerClasses} {...props}>
      {children}
    </div>
  );
};

Modal.Body = ({ children, className = '', ...props }) => {
  const bodyClasses = [styles.modalBody, className].filter(Boolean).join(' ');
  return (
    <div className={bodyClasses} {...props}>
      {children}
    </div>
  );
};

Modal.Footer = ({ children, className = '', ...props }) => {
  const footerClasses = [styles.modalFooter, className].filter(Boolean).join(' ');
  return (
    <div className={footerClasses} {...props}>
      {children}
    </div>
  );
};

// Convenience component for common modal footer with Cancel/Confirm buttons
Modal.Actions = ({ 
  onCancel, 
  onConfirm, 
  cancelText = 'Cancel', 
  confirmText = 'Confirm',
  cancelVariant = 'secondary',
  confirmVariant = 'primary',
  confirmDisabled = false,
  className = ''
}) => {
  const footerClasses = [styles.modalFooter, className].filter(Boolean).join(' ');
  return (
    <div className={footerClasses}>
      <Button variant={cancelVariant} onClick={onCancel}>
        {cancelText}
      </Button>
      <Button variant={confirmVariant} onClick={onConfirm} disabled={confirmDisabled}>
        {confirmText}
      </Button>
    </div>
  );
};

Modal.propTypes = {
  /** Whether the modal is open */
  isOpen: PropTypes.bool.isRequired,
  /** Function to call when the modal should close */
  onClose: PropTypes.func.isRequired,
  /** Modal title */
  title: PropTypes.node,
  /** Modal content */
  children: PropTypes.node.isRequired,
  /** Modal size */
  size: PropTypes.oneOf(['sm', 'md', 'lg', 'xl', 'fullscreen']),
  /** Whether the modal content should be centered */
  centered: PropTypes.bool,
  /** Whether to show the close button in the top right corner */
  showCloseButton: PropTypes.bool,
  /** Whether the modal should close when the Escape key is pressed */
  closeOnEsc: PropTypes.bool,
  /** Whether the modal should close when the backdrop is clicked */
  closeOnBackdrop: PropTypes.bool,
  /** Additional CSS class for the modal */
  className: PropTypes.string,
  /** Additional CSS class for the backdrop */
  backdropClassName: PropTypes.string,
  /** Content for the modal footer */
  footer: PropTypes.node,
};

Modal.Header.propTypes = {
  /** Header content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

Modal.Body.propTypes = {
  /** Body content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

Modal.Footer.propTypes = {
  /** Footer content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

Modal.Actions.propTypes = {
  /** Function to call when the cancel button is clicked */
  onCancel: PropTypes.func.isRequired,
  /** Function to call when the confirm button is clicked */
  onConfirm: PropTypes.func.isRequired,
  /** Text for the cancel button */
  cancelText: PropTypes.string,
  /** Text for the confirm button */
  confirmText: PropTypes.string,
  /** Variant for the cancel button */
  cancelVariant: PropTypes.string,
  /** Variant for the confirm button */
  confirmVariant: PropTypes.string,
  /** Whether the confirm button should be disabled */
  confirmDisabled: PropTypes.bool,
  /** Additional CSS class */
  className: PropTypes.string,
};

export default Modal;