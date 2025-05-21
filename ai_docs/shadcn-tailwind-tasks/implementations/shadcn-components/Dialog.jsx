import React, { useEffect } from 'react';
import PropTypes from 'prop-types';
import {
  Dialog as ShadcnDialog,
  DialogPortal,
  DialogOverlay,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog"; // This path assumes shadcn components are installed in the @/components/ui directory
import { Button } from "./Button"; // Use our wrapped Button component
import { X as XIcon } from "lucide-react"; // You'll need to install lucide-react for icons
import { cn } from "@/lib/utils"; // This is a utility function provided by shadcn for class name merging

/**
 * Dialog component integrating shadcn/ui while maintaining the original Modal API
 */
const Dialog = ({
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
  // When the dialog opens, prevent scrolling
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Map our sizes to tailwind classes
  const sizeClasses = {
    sm: "sm:max-w-sm",
    md: "sm:max-w-md",
    lg: "sm:max-w-lg",
    xl: "sm:max-w-xl",
    fullscreen: "sm:max-w-[95vw] sm:h-[95vh]",
  };

  // Build dialog content classes
  const dialogContentClasses = cn(
    sizeClasses[size] || sizeClasses.md,
    centered && "sm:align-middle",
    className
  );

  // Note: Shadcn Dialog handles ESC key closing by default
  // For closeOnBackdrop, we need to customize the DialogOverlay

  return (
    <ShadcnDialog open={isOpen} onOpenChange={(open) => {
      if (!open) onClose();
    }}>
      <DialogPortal>
        <DialogOverlay 
          className={cn(
            "bg-background/80 backdrop-blur-sm",
            backdropClassName
          )}
          onClick={(e) => {
            // If closeOnBackdrop is true and the click is on the overlay, close the dialog
            if (closeOnBackdrop && e.target === e.currentTarget) {
              onClose();
            }
          }}
        />
        <DialogContent 
          className={dialogContentClasses} 
          onEscapeKeyDown={(e) => {
            if (!closeOnEsc) {
              e.preventDefault();
            }
          }}
          {...props}
        >
          {(title || showCloseButton) && (
            <DialogHeader>
              {title && <DialogTitle>{title}</DialogTitle>}
              {showCloseButton && (
                <DialogClose 
                  className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none"
                  onClick={onClose}
                >
                  <XIcon className="h-4 w-4" />
                  <span className="sr-only">Close</span>
                </DialogClose>
              )}
            </DialogHeader>
          )}
          
          <div className="py-4">
            {children}
          </div>
          
          {footer && (
            <DialogFooter>
              {footer}
            </DialogFooter>
          )}
        </DialogContent>
      </DialogPortal>
    </ShadcnDialog>
  );
};

// Convenience components for dialog parts that match the original Modal API
Dialog.Header = ({ children, className = '', ...props }) => {
  return (
    <DialogHeader className={className} {...props}>
      {children}
    </DialogHeader>
  );
};

Dialog.Body = ({ children, className = '', ...props }) => {
  return (
    <div className={cn("py-4", className)} {...props}>
      {children}
    </div>
  );
};

Dialog.Footer = ({ children, className = '', ...props }) => {
  return (
    <DialogFooter className={className} {...props}>
      {children}
    </DialogFooter>
  );
};

// Convenience component for common dialog footer with Cancel/Confirm buttons
Dialog.Actions = ({ 
  onCancel, 
  onConfirm, 
  cancelText = 'Cancel', 
  confirmText = 'Confirm',
  cancelVariant = 'secondary',
  confirmVariant = 'primary',
  confirmDisabled = false,
  className = ''
}) => {
  return (
    <DialogFooter className={className}>
      <Button variant={cancelVariant} onClick={onCancel}>
        {cancelText}
      </Button>
      <Button variant={confirmVariant} onClick={onConfirm} disabled={confirmDisabled}>
        {confirmText}
      </Button>
    </DialogFooter>
  );
};

Dialog.propTypes = {
  /** Whether the dialog is open */
  isOpen: PropTypes.bool.isRequired,
  /** Function to call when the dialog should close */
  onClose: PropTypes.func.isRequired,
  /** Dialog title */
  title: PropTypes.node,
  /** Dialog content */
  children: PropTypes.node.isRequired,
  /** Dialog size */
  size: PropTypes.oneOf(['sm', 'md', 'lg', 'xl', 'fullscreen']),
  /** Whether the dialog content should be centered */
  centered: PropTypes.bool,
  /** Whether to show the close button in the top right corner */
  showCloseButton: PropTypes.bool,
  /** Whether the dialog should close when the Escape key is pressed */
  closeOnEsc: PropTypes.bool,
  /** Whether the dialog should close when the backdrop is clicked */
  closeOnBackdrop: PropTypes.bool,
  /** Additional CSS class for the dialog */
  className: PropTypes.string,
  /** Additional CSS class for the backdrop */
  backdropClassName: PropTypes.string,
  /** Content for the dialog footer */
  footer: PropTypes.node,
};

Dialog.Header.propTypes = {
  /** Header content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

Dialog.Body.propTypes = {
  /** Body content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

Dialog.Footer.propTypes = {
  /** Footer content */
  children: PropTypes.node,
  /** Additional CSS class */
  className: PropTypes.string,
};

Dialog.Actions.propTypes = {
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

export default Dialog;