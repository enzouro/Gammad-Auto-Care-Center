import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  CircularProgress
} from '@pankod/refine-mui';
import { Pending as PendingIcon } from '@mui/icons-material';

interface LoadingDialogProps {
  open: boolean;
  loadingMessage: string;
}

const LoadingDialog: React.FC<LoadingDialogProps> = ({
  open,
  loadingMessage,
}) => {
  return (
    <Dialog
      open={open}
      aria-labelledby="loading-dialog-title"
      aria-describedby="loading-dialog-description"
    >
      <DialogTitle 
        id="loading-dialog-title"
        sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 1,
          color: 'primary.main'
        }}
      >
        <PendingIcon color="primary" />
        Loading
      </DialogTitle>
      <DialogContent 
        sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center',
          gap: 2 
        }}
      >
        <CircularProgress color="primary" />
        <DialogContentText 
          id="loading-dialog-description"
          sx={{ textAlign: 'center' }}
        >
          {loadingMessage}
        </DialogContentText>
      </DialogContent>
    </Dialog>
  );
};

export default LoadingDialog;