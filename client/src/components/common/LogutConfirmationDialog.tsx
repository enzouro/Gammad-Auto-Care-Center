// components/common/LogoutConfirmationDialog.tsx
import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Typography,
} from '@pankod/refine-mui';
import { Logout } from '@mui/icons-material';

interface LogoutConfirmationDialogProps {
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

const LogoutConfirmationDialog: React.FC<LogoutConfirmationDialogProps> = ({
  open,
  onConfirm,
  onCancel,
}) => {
  return (
    <Dialog
      open={open}
      onClose={onCancel}
      aria-labelledby="logout-dialog-title"
      aria-describedby="logout-dialog-description"
    >
      <DialogTitle id="logout-dialog-title">
        <Typography variant="h6">Confirm Logout</Typography>
      </DialogTitle>
      <DialogContent>
        <DialogContentText id="logout-dialog-description">
          Are you sure you want to log out? Any unsaved changes may be lost.
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onCancel} color="inherit">
          Cancel
        </Button>
        <Button
          onClick={onConfirm}
          color="primary"
          variant="contained"
          startIcon={<Logout />}
        >
          Logout
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default LogoutConfirmationDialog;
