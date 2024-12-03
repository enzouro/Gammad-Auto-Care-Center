// hooks/useDeleteWithConfirmation.ts
import { useState } from 'react';
import { useDelete } from '@pankod/refine-core';
import { useNavigate } from '@pankod/refine-react-router-v6';

interface DeleteConfirmationState {
  open: boolean;
  id: string | null;
  seq: string;
  isDeleted: boolean;
}

interface ErrorState {
  open: boolean;
  message: string;
}

interface UseDeleteWithConfirmationProps {
  resource: string;
  redirectPath?: string;
  onSuccess?: () => void;
  onError?: (error: any) => void;
}

const useDeleteWithConfirmation = ({
  resource,
  redirectPath,
  onSuccess,
  onError,
}: UseDeleteWithConfirmationProps) => {
  const { mutate: deleteMutation } = useDelete();
  const navigate = useNavigate();
  const [deleteConfirmation, setDeleteConfirmation] = useState<DeleteConfirmationState>({
    open: false,
    id: null,
    seq: '',
    isDeleted: false
  });
  const [error, setError] = useState<ErrorState>({
    open: false,
    message: ''
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleDeleteClick = (id: string, seq: string, isDeleted: boolean) => {
    setDeleteConfirmation({
      open: true,
      id,
      seq,
      isDeleted
    });
  };

  const handleTableDelete = (ids: string[], rows: any[]) => {
    // Check if all selected items have the same deletion status
    const firstItemDeleted = rows.find(row => row.id === ids[0])?.deleted || false;
    const allSameStatus = rows.every(row => (row.deleted || false) === firstItemDeleted);

    if (!allSameStatus) {
      setError({
        open: true,
        message: 'Please select items with the same deletion status'
      });
      return;
    }

    setDeleteConfirmation({
      open: true,
      id: ids.join(','),
      seq: `${ids.length} items`,
      isDeleted: firstItemDeleted
    });
  };

  const confirmDelete = () => {
    if (deleteConfirmation.id) {
      setIsLoading(true);
      deleteMutation(
        {
          resource,
          id: deleteConfirmation.id,
        },
        {
          onSuccess: () => {
            setDeleteConfirmation({ open: false, id: null, seq: '', isDeleted: false });
            if (redirectPath) {
              navigate(redirectPath);
            }
            onSuccess?.();
            setIsLoading(false);
          },
          onError: (error: any) => {
            console.error('Delete error:', error);
            setDeleteConfirmation({ open: false, id: null, seq: '', isDeleted: false });
            
            let errorMessage = 'An error occurred while deleting the item.';
            if (error.response?.data?.message) {
              errorMessage = error.response.data.message;
            } else if (error.message) {
              errorMessage = error.message;
            }
            
            setError({
              open: true,
              message: errorMessage
            });
            
            onError?.(error);
            setIsLoading(false);
          }
        }
      );
    }
  };

  const cancelDelete = () => {
    setDeleteConfirmation({ open: false, id: null, seq: '', isDeleted: false });
  };

  const closeErrorDialog = () => {
    setError({ open: false, message: '' });
  };

  return {
    deleteConfirmation,
    error,
    isLoading,
    handleDeleteClick,
    handleTableDelete,
    confirmDelete,
    cancelDelete,
    closeErrorDialog,
  };
};

export default useDeleteWithConfirmation;