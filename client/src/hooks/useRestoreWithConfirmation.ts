// client/src/hooks/useRestoreWithConfirmation.ts
import { useState } from "react";
import { useNotification, useInvalidate } from "@pankod/refine-core";
import { useNavigate } from "@pankod/refine-react-router-v6";

interface RestoreConfirmationState {
  open: boolean;
  ids: string[];
  seq: string;
}

interface ErrorState {
  open: boolean;
  message: string;
}

interface UseRestoreWithConfirmationProps {
  resource: string;
  restoreEndpoint?: string;
  redirectPath?: string;
  onSuccess?: () => void;
  onError?: (error: any) => void;
}

const useRestoreWithConfirmation = ({
  resource,
  restoreEndpoint,
  redirectPath,
  onSuccess,
  onError,
}: UseRestoreWithConfirmationProps) => {
  const invalidate = useInvalidate();
  const { open } = useNotification();
  const navigate = useNavigate();
  const [restoreConfirmation, setRestoreConfirmation] = useState<RestoreConfirmationState>({
    open: false,
    ids: [],
    seq: '',
  });
  const [error, setError] = useState<ErrorState>({
    open: false,
    message: '',
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleRestoreClick = (id: string, seq: string) => {
    setRestoreConfirmation({
      open: true,
      ids: [id],
      seq,
    });
  };

  const handleTableRestore = (ids: string[], rows: any[]) => {
    if (ids.length === 1) {
      const item = rows.find((row) => row.id === ids[0]);
      setRestoreConfirmation({
        open: true,
        ids,
        seq: item?.seq || '',
      });
    } else {
      setRestoreConfirmation({
        open: true,
        ids,
        seq: `${ids.length} items`,
      });
    }
  };

  const confirmRestore = async () => {
    if (restoreConfirmation.ids.length === 0) return;
  
    setIsLoading(true);
    try {
      const ids = restoreConfirmation.ids.join(',');
      const response = await fetch(
        `http://localhost:8080/api/v1/${resource}/${ids}/restore`, 
        {
          method: 'PATCH',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${localStorage.getItem('token')}`,
          },
        }
      );
  
      if (!response.ok) {
        throw new Error('Failed to restore items');
      }
  
      setRestoreConfirmation({ open: false, ids: [], seq: '' });
      
      open?.({
        type: 'success',
        message: 'Success',
        description: `Successfully restored ${restoreConfirmation.ids.length} item(s).`,
      });
  
      if (redirectPath) {
        navigate(redirectPath);
      }
  
      invalidate({
        resource,
        invalidates: ["list"],
      });
  
      onSuccess?.();
    } catch (error) {
      handleRestoreError(error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleRestoreError = (error: any) => {
    setRestoreConfirmation({ open: false, ids: [], seq: '' });
    const errorMessage = error instanceof Error ? error.message : 'An error occurred while restoring items';
    
    setError({
      open: true,
      message: errorMessage,
    });
  
    open?.({
      type: 'error',
      message: 'Error',
      description: errorMessage,
    });
  
    onError?.(error);
  };

  const cancelRestore = () => {
    setRestoreConfirmation({ open: false, ids: [], seq: '' });
  };

  const closeErrorDialog = () => {
    setError({ open: false, message: '' });
  };

  return {
    restoreConfirmation,
    error,
    isLoading,
    handleRestoreClick,
    handleTableRestore,
    confirmRestore,
    cancelRestore,
    closeErrorDialog,
  };
};

export default useRestoreWithConfirmation;