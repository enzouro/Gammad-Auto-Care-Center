import React, { useContext, useState } from 'react';
import { useList, useUpdate, useDelete } from '@pankod/refine-core';
import {
  DataGrid,
  GridColDef,
  Toolbar,
  Typography,
  Box,
  Stack,
  Checkbox,
  Paper,
  CircularProgress,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@pankod/refine-mui';
import DeleteIcon from '@mui/icons-material/Delete';
import { ColorModeContext } from 'contexts';
import CustomIconButton from 'components/common/CustomIconButton';
import useDynamicHeight from 'hooks/useDynamicHeight';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const UserManagement = () => {
  const { data, isLoading, isError, refetch } = useList({
    resource: 'user-management',
  });

  const { mutate: updateUser } = useUpdate();
  const { mutate: deleteUser } = useDelete();
  const { mode } = useContext(ColorModeContext);
  const containerHeight = useDynamicHeight();
  const [selectionModel, setSelectionModel] = useState<string[]>([]);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [toggleDialogOpen, setToggleDialogOpen] = useState(false);
  const [currentUser, setCurrentUser] = useState<any>(null);

  const users = data?.data.map((user) => ({
    ...user,
    id: user._id, // Map `_id` to `id`
  })) ?? [];

  const handleDeleteUsers = () => {
    selectionModel.forEach((id) => {
      deleteUser({ resource: 'user-management', id });
    });
    refetch();
    setSelectionModel([]);
    setDeleteDialogOpen(false);
  };

  const handleAllowedToggle = () => {
    if (currentUser) {
      updateUser(
        {
          resource: 'user-management',
          id: currentUser._id,
          values: { isAllowed: !currentUser.isAllowed },
        },
        { onSuccess: () => refetch() }
      );
      setToggleDialogOpen(false);
    }
  };

  const openDeleteDialog = () => {
    setDeleteDialogOpen(true);
  };

  const openToggleDialog = (user: any) => {
    setCurrentUser(user);
    setToggleDialogOpen(true);
  };

  const CustomTableToolbar = () => (
    <Toolbar
      sx={{
        pl: { sm: 2 },
        pr: { xs: 1, sm: 1 },
        ...(selectionModel.length > 0 && { bgcolor: 'rgba(25, 118, 210, 0.08)' }),
        display: 'flex',
        justifyContent: 'space-between',
      }}
    >
      <Typography
        sx={{ flex: '1 1 100%' }}
        color="inherit"
        variant="subtitle1"
        component="div"
      >
        {selectionModel.length > 0
          ? `${selectionModel.length} selected`
          : 'All Records'}
      </Typography>
      {selectionModel.length > 0 && (
        <Stack direction="row" spacing={1}>
          <CustomIconButton
            title={`Delete ${selectionModel.length > 1 ? `(${selectionModel.length})` : ''}`}
            icon={<DeleteIcon />}
            backgroundColor="error.light"
            color="error.dark"
            handleClick={openDeleteDialog}
          />
        </Stack>
      )}
    </Toolbar>
  );

  const columns: GridColDef[] = [
    {
      field: 'avatar',
      headerName: 'Avatar',
      width: 100,
      renderCell: (params) => (
        <img
          src={params.value}
          alt="avatar"
          style={{ width: 40, height: 40, borderRadius: '50%' }}
        />
      ),
    },
    { field: 'name', headerName: 'Name', flex: 1 },
    { field: 'email', headerName: 'Email', flex: 1 },

    {
      field: 'isAdmin',
      headerName: 'Role',
      width: 100,
      valueGetter: (params) => (params.row.isAdmin ? 'Admin' : 'User'),
    },
    {
      field: 'isAllowed',
      headerName: 'Is Allowed',
      width: 120,
      renderCell: (params) => (
        <Checkbox
          checked={params.row.isAllowed}
          onChange={() => openToggleDialog(params.row)}
        />
      ),
    },
  ];

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading users data..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading users data"
      />
    );
  }

  return (
    <Paper elevation={3} sx={{     
      height: containerHeight,
        display: 'flex',
        flexDirection: 'column',
        m: 2,
      overflow: 'hidden'}}>
      <Typography
        variant="h4"
        sx={{
          p: 2,
          fontWeight: 600,
        }}
      >
        User Management
      </Typography>
      <Box sx={{ height: 660, width: '100%' }}>
        <DataGrid
          rows={users}
          columns={columns}
          checkboxSelection
          disableSelectionOnClick
          autoHeight={false}
          selectionModel={selectionModel}
          onSelectionModelChange={(newSelectionModel) =>
            setSelectionModel(newSelectionModel as string[])
          }
          components={{
            Toolbar: CustomTableToolbar,
          }}
          sx={{
            height: containerHeight,
          '& .MuiDataGrid-main': {
            overflow: 'hidden'
            },
            '& ::-webkit-scrollbar': {
              width: '10px',
              height: '10px',
            },
          '& .MuiDataGrid-row:hover': {
            backgroundColor: 'rgba(0, 0, 0, 0.04)'
            },
            '& .MuiDataGrid-cell': {
              padding: '8px',
              whiteSpace: 'normal',
            wordWrap: 'break-word'
          },
                  /* Dark Mode */
          '& .MuiDataGrid-columnHeaders': {
            backgroundColor: mode === 'light' ? '#f5f5f5' : '#333333',
            borderBottom: mode === 'light' ? '2px solid #e0e0e0' : '2px solid #444444',
            color: mode === 'light' ? 'inherit' : '#f5f5f5'
          },
          '& .MuiDataGrid-columnHeader': {
            padding: '8px',
            fontWeight: 'bold'
          }
          }}
        />
      </Box>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Confirm Deletion</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete the selected users? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)} color="primary">
            Cancel
          </Button>
          <Button
            onClick={handleDeleteUsers}
            color="error"
            variant="contained"
            startIcon={<DeleteIcon />}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Toggle Allowed Dialog */}
      <Dialog open={toggleDialogOpen} onClose={() => setToggleDialogOpen(false)}>
        <DialogTitle>Change User Permission</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to {currentUser?.isAllowed ? 'disallow' : 'allow'} this user?
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setToggleDialogOpen(false)} color="primary">
            Cancel
          </Button>
          <Button onClick={handleAllowedToggle} color="primary" variant="contained">
            Confirm
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default UserManagement;
