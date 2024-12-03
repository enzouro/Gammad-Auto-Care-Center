// components/common/ErrorDialog.tsx
import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
} from '@pankod/refine-mui';
import { Error as ErrorIcon } from '@mui/icons-material';

interface ErrorDialogProps {
  open: boolean;
  errorMessage: string;
  onClose?: () => void;
}

const ErrorDialog: React.FC<ErrorDialogProps> = ({
  open,
  errorMessage,
  onClose,
}) => {
  const handleClose = onClose || (() => {});

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      aria-labelledby="error-dialog-title"
      aria-describedby="error-dialog-description"
    >
      <DialogTitle 
        id="error-dialog-title"
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1,
          color: 'error.main'
        }}
      >
        <ErrorIcon color="error" />
        Error Occurred
      </DialogTitle>
      <DialogContent>
        <DialogContentText id="error-dialog-description">
          {errorMessage}
        </DialogContentText>
      </DialogContent>
      {onClose && (
        <DialogActions>
          <Button onClick={handleClose} color="primary" variant="contained">
            Close
          </Button>
        </DialogActions>
      )}
    </Dialog>
  );
};
export default ErrorDialog;
