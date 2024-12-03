// components/common/DeleteConfirmationDialog.tsx
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
import { Delete, DeleteForever } from '@mui/icons-material';

interface DeleteConfirmationDialogProps {
  open: boolean;
  isDeleted: boolean;
  contentText: string;
  onConfirm: () => void;
  onCancel: () => void;
}

const DeleteConfirmationDialog: React.FC<DeleteConfirmationDialogProps> = ({
  open,
  isDeleted,
  contentText,
  onConfirm,
  onCancel,
}) => {
  const dialogConfig = isDeleted ? {
    title: 'Confirm Permanent Deletion',
    description: `Are you sure you want to permanently delete ${contentText}? This action cannot be undone.`,
    confirmButton: {
      text: 'Delete Permanently',
      icon: <DeleteForever />,
      color: 'error' as const
    }
  } : {
    title: 'Move to Trash',
    description: `Are you sure you want to move ${contentText} to trash? You can restore it later.`,
    confirmButton: {
      text: 'Move to Trash',
      icon: <Delete />,
      color: 'warning' as const
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onCancel}
      aria-labelledby="delete-dialog-title"
      aria-describedby="delete-dialog-description"
    >
      <DialogTitle id="delete-dialog-title">
        <Typography variant="h6">
          {dialogConfig.title}
        </Typography>
      </DialogTitle>
      <DialogContent>
        <DialogContentText id="delete-dialog-description">
          {dialogConfig.description}
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onCancel} color="inherit">
          Cancel
        </Button>
        <Button 
          onClick={onConfirm} 
          color={dialogConfig.confirmButton.color}
          variant="contained"
          startIcon={dialogConfig.confirmButton.icon}
        >
          {dialogConfig.confirmButton.text}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DeleteConfirmationDialog;