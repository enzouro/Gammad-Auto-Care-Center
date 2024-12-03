// components/common/RestoreConfirmationDialog.tsx
import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
} from '@pankod/refine-mui';
import { Restore } from '@mui/icons-material';

interface RestoreConfirmationDialogProps {
  open: boolean;
  title?: string;
  contentText: string;
  onConfirm: () => void;
  onCancel: () => void;
}

const RestoreConfirmationDialog: React.FC<RestoreConfirmationDialogProps> = ({
  open,
  title = 'Confirm Restore',
  contentText,
  onConfirm,
  onCancel,
}) => {
  return (
    <Dialog
      open={open}
      onClose={onCancel}
      aria-labelledby="restore-dialog-title"
      aria-describedby="restore-dialog-description"
    >
      <DialogTitle id="restore-dialog-title">
        {title}
      </DialogTitle>
      <DialogContent>
        <DialogContentText id="restore-dialog-description">
          {contentText}
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onCancel} color="primary">
          Cancel
        </Button>
        <Button 
          onClick={onConfirm} 
          color="success" 
          variant="contained"
          startIcon={<Restore />}
        >
          Restore
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RestoreConfirmationDialog;