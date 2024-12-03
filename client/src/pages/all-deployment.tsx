import { Add } from '@mui/icons-material';
import { useTable, useUpdate } from '@pankod/refine-core';
import { Alert, Box, Button, ButtonGroup, CircularProgress, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, FormControl, GridColDef, GridRenderCellParams, MenuItem, Paper, Select, Snackbar, Stack, Switch, TextField, Typography } from '@pankod/refine-mui';
import { useNavigate } from '@pankod/refine-react-router-v6';
import CustomButton from 'components/common/CustomButton';
import CustomTable from 'components/common/CustomTable';
import DeleteConfirmationDialog from 'components/common/DeleteConfirmationDialog';
import ErrorDialog from 'components/common/ErrorDialog';
import RestoreConfirmationDialog from 'components/common/RestoreConfirmationDialog';
import useDeleteWithConfirmation from 'hooks/useDeleteWithConfirmation';
import useDynamicHeight from 'hooks/useDynamicHeight';
import useRestoreWithConfirmation from 'hooks/useRestoreWithConfirmation';
import LoadingDialog from 'components/common/LoadingDialog';
import { useMemo, useState } from 'react';



const AllDeployments = () => {
  const { mutate: updateDeployment } = useUpdate();
  const containerHeight = useDynamicHeight();
  const navigate = useNavigate();

  // States
  const [searchTerm, setSearchTerm] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [deletedFilter, setDeletedFilter] = useState('active');

  const [dialogState, setDialogState] = useState({
    open: false,
    message: '',
    title: '',
    action: () => {}
  });
  const [errorDialog, setErrorDialog] = useState({
    open: false,
    message: ''
  });

  // Use both delete and restore hooks
  const {
    deleteConfirmation,
    error: deleteError,
    handleTableDelete,
    confirmDelete,
    cancelDelete,
    isLoading: isDeleteLoading,
    closeErrorDialog: closeDeleteErrorDialog,
  } = useDeleteWithConfirmation({
    resource: 'deployments',
    redirectPath: '/deployments',
  });
  
  const {
    restoreConfirmation,
    error: restoreError,
    handleTableRestore,
    confirmRestore,
    cancelRestore,
    isLoading: isRestoreLoading,
    closeErrorDialog: closeRestoreErrorDialog,
  } = useRestoreWithConfirmation({
    resource: 'deployments',
    redirectPath: '/deployments',
  });

  const {
    tableQueryResult: { data, isLoading, isError }
  } = useTable({
    resource: 'deployments',
    hasPagination: false,
  });

  const allDeployments = data?.data ?? [];

  // Handle dialog close
  const handleDialogClose = () => {
    setDialogState((prev: any) => ({ ...prev, open: false }));
  };

  const handleErrorDialogClose = () => {
    setErrorDialog(prev => ({ ...prev, open: false }));
  };

  // Filter the data based on search term and date range
  const filteredRows = useMemo(() => {
    return allDeployments.filter((deployment) => {
      const deploymentDate = new Date(deployment.date);
      const matchesSearch =
        !searchTerm ||
        deployment.clientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        deployment.vehicleModel.toLowerCase().includes(searchTerm.toLowerCase()) ||
        deployment.seq.toString().includes(searchTerm);

      const matchesDateRange =
        (!startDate || deploymentDate >= new Date(startDate)) &&
        (!endDate || deploymentDate <= new Date(endDate));

      const matchesDeletedFilter =
        (deletedFilter === 'active' && !deployment.deleted) || 
        (deletedFilter === 'deleted' && deployment.deleted);

      return matchesSearch && matchesDateRange && matchesDeletedFilter;
    });
  }, [allDeployments, searchTerm, startDate, endDate, deletedFilter]);

  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'error' as 'error' | 'warning' | 'info' | 'success'
  });

  const handleSnackbarClose = (_event?: React.SyntheticEvent | Event, reason?: string) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleStatusChange = (id: string, field: 'deploymentStatus' | 'releaseStatus', newValue: boolean) => {
    const statusName = field === 'deploymentStatus' ? 'deployment' : 'release';
    const action = newValue ? 'enable' : 'disable';

    // Find the current deployment
    const currentDeployment = allDeployments.find(dep => dep._id === id);
    if (!currentDeployment) return;

    // Check conditions ONLY for release status
    if (field === 'releaseStatus' && newValue) {
      // Show snackbar if deployment status is off
      // if (!currentDeployment.deploymentStatus) {
      //   setSnackbar({
      //     open: true,
      //     message: "Cannot release when Deployment Status is OFF.",
      //     severity: 'error'
      //   });
      //   return;
      // }
      
      // Show snackbar if vehicle is not repaired
      if (currentDeployment.repairStatus !== 'Repaired') {
        setSnackbar({
          open: true,
          message: "Cannot release when the vehicle is not \"Repaired\".",
          severity: 'error'
        });
        return;
      }
    }

    const today = new Date().toLocaleDateString('en-CA');
    
    // Prepare update data
    const updateData = {
      repairStatus: currentDeployment.repairStatus,
      repairedDate: currentDeployment.repairedDate,
      [field]: newValue,
      ...(field === 'deploymentStatus' && {
        deploymentDate: newValue ? today : null
      }),
      ...(field === 'releaseStatus' && {
        releaseDate: newValue ? today : null
      })
    };

    // Set up confirmation dialog
    setDialogState({
      open: true,
      title: `Confirm ${statusName} Status Change`,
      message: `Are you sure you want to ${action} the ${statusName} status?`,
      action: () => {
        updateDeployment(
          {
            resource: 'deployments',
            id,
            values: updateData,
          },
          {
            onSuccess: () => {
            },
            onError: (error) => {
              setErrorDialog({
                open: true,
                message: `Failed to update ${statusName} status.`
              });
              console.error('Update error:', error);
            },
          },
        );
      }
    });
  };

  const handleRepairStatusChange = (id: string, field: 'repairStatus', newStatus: string) => {
    const statusName = 'repair';
    
    setDialogState({
      open: true,
      title: 'Confirm Status Change',
      message: `Are you sure you want to update the ${statusName} status?`,
      action: () => {
        const today = new Date().toLocaleDateString('en-CA');
        
        updateDeployment(
          {
            resource: 'deployments',
            id,
            values: {
              repairStatus: newStatus,
              repairedDate: newStatus === 'Repaired' ? today : null,
            },
          },
          {
            onSuccess: () => {
            },
            onError: (error) => {
              setErrorDialog({
                open: true,
                message: 'Failed to update repair status.'
              });
              console.error('Update error:', error);
            },
          }
        );
      }
    });
  };

  const columns: GridColDef[] = [
    { field: 'seq', headerName: 'Seq', flex: 1, sortable: true },
    { field: 'date', headerName: 'Date', flex: 1 },
    { field: 'clientName', headerName: 'Client Name', flex: 1 },
    { field: 'vehicleModel', headerName: 'Vehicle Model', flex: 1 },
    { field: 'arrivalDate', headerName: 'Arrival Date', flex: 1 },
    { field: 'partsCount', headerName: 'Parts Count', flex: 1 },
    {
      field: 'deploymentStatus',
      headerName: 'Deployed',
      flex: 1,
      renderCell: (params: GridRenderCellParams) => (
        <Switch
          checked={params.value}
          onChange={(e) => handleStatusChange(params.row.id, 'deploymentStatus', e.target.checked)}
          color="primary"
        />
      )
    },
    {
      field: 'deploymentDate',
      headerName: 'Deployed Date',
      flex: 1,
    },
    {
      field: 'repairStatus',
      headerName: 'Status',
      flex: 1,
      align: 'center',
      renderCell: (params: GridRenderCellParams) => (
        <FormControl
          size="small"
          sx={{
            width: '100px', // Take full width of cell
            minWidth: 'unset', // Remove minimum width
            align: 'center',
            display: 'flex-row',
            flexWrap: 'wrap',
          }}
        >
          <Select
            value={params.value}
            onChange={(e) => handleRepairStatusChange(params.row.id, 'repairStatus', e.target.value)}
            sx={{
              '& .MuiSelect-select': {
                align: 'center',
                display: 'flex',
               flexWrap: 'wrap',
               fontSize: () => {
                switch (params.value) {
                  case 'Pending':
                    return '0.950rem'; // Default size
                  case 'In Progress':
                    return '0.780rem'; // Smallest size
                  case 'Repaired':
                    return '0.950rem';
                  case 'Cancelled':
                    return '0.868rem'; // Medium size
                  default:
                    return '0.950rem';
                }
               },
               color: () =>{
                switch (params.value) {
                  case 'Pending':
                    return '#FF9800'; // Default size
                  case 'In Progress':
                    return '#2196F3'; // Smallest size
                  case 'Cancelled':
                    return '#F44336';
                  case 'Repaired':
                    return '#4CAF50'; // Medium size
                  default:
                    return '#FF9800';
                }
               }
              },
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'transparent' // Optional: remove border for cleaner look
              },
              '&:hover .MuiOutlinedInput-notchedOutline': {
                borderColor: 'primary.main' // Show border on hover
              },
            }}
          >
            <MenuItem value="Pending">Pending</MenuItem>
            <MenuItem value="In Progress">In Progress</MenuItem>
            <MenuItem value="Repaired">Repaired</MenuItem>
            <MenuItem value="Cancelled">Cancelled</MenuItem>
          </Select>
        </FormControl>
      )
    },
    {
      field: 'repairedDate',
      headerName: 'Repaired Date',
      flex: 1,
    },
    {
      field: 'releaseStatus',
      headerName: 'Released',
      flex: 1,
      renderCell: (params: GridRenderCellParams) => (
        <Switch
          checked={params.value}
          onChange={(e) => handleStatusChange(params.row.id, 'releaseStatus', e.target.checked)}
          color="primary"
        />
      )
    },
    {
      field: 'releaseDate',
      headerName: 'Released Date',
      flex: 1,
    },
  ];

  const handleView = (id: string) => {
    navigate(`/deployments/show/${id}`);
  };

  const handleEdit = (id: string) => {
    navigate(`/deployments/edit/${id}`);
  };

  const rows = filteredRows.map((deployment) => ({
    id: deployment._id,
    _id: deployment._id,
    seq: deployment.seq,
    date: new Date(deployment.date).toLocaleDateString(),
    clientName: deployment.clientName,
    vehicleModel: deployment.vehicleModel,
    arrivalDate: new Date(deployment.arrivalDate).toLocaleDateString(),
    partsCount: deployment.parts?.length > 0 ? deployment.parts.length : 'TBA',
    deploymentStatus: deployment.deploymentStatus,
    deploymentDate: deployment.deploymentDate ? new Date(deployment.deploymentDate).toLocaleDateString() : 'N/A',
    releaseStatus: deployment.releaseStatus,
    releaseDate: deployment.releaseDate ? new Date(deployment.releaseDate).toLocaleDateString() : 'N/A',
    repairStatus: deployment.repairStatus,
    repairedDate: deployment.repairedDate ? new Date(deployment.repairedDate).toLocaleDateString() : 'N/A',
    deleted: deployment.deleted || false,
  }));

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading deployments data..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading deployments data"
      />
    );
  }

  return (
    <Paper
      elevation={3}
      sx={{
        height: containerHeight,
        display: 'flex',
        flexDirection: 'column',
        m: 2,
        overflow: 'hidden'
      }}
    >
      <Typography
        variant="h4"
        sx={{
          p: 2,
          fontWeight: 600,
        }}
      >
        {!allDeployments.length ? 'There are no deployments' : 'All Deployments'}
      </Typography>

      <Box sx={{
        p: 2,
        display: 'flex',
        flexDirection: { xs: 'column', md: 'row' },
        gap: 2,
        alignItems: { xs: 'stretch', md: 'center' },
        justifyContent: 'space-between'
      }}>
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          sx={{ flex: 1, alignItems: 'center' }}
        >
          <TextField
            size="small"
            label="Search"
            placeholder="Search by client, vehicle, or sequence"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            sx={{ minWidth: '300px' }}
          />
          <TextField
            size="small"
            label="Start Date"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            InputLabelProps={{ shrink: true }}
          />
          <TextField
            size="small"
            label="End Date"
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            InputLabelProps={{ shrink: true }}
          />
          <ButtonGroup>
            <Button
              variant={deletedFilter === 'active' ? 'contained' : 'outlined'}
              onClick={() => setDeletedFilter('active')}
              size="small"
              sx={{
                height: '40px',
                backgroundColor: deletedFilter === 'active' ? 'primary.light' : 'inherit',
                color: deletedFilter === 'active' ? 'primary.contrastText' : 'inherit',
                '&:hover': {
                  backgroundColor: 'primary.main',
                },
              }}
            >
              Active
            </Button>
            <Button
              variant={deletedFilter === 'deleted' ? 'contained' : 'outlined'}
              onClick={() => setDeletedFilter('deleted')}
              size="small"
              sx={{
                height: '40px',
                backgroundColor: deletedFilter === 'deleted' ? 'error.light' : 'inherit',
                color: deletedFilter === 'deleted' ? 'error.contrastText' : 'inherit',
                '&:hover': {
                  backgroundColor: 'error.main',
                },
              }}
            >
              Deleted
            </Button>
          </ButtonGroup>
        </Stack>

        <CustomButton
          title="Add"
          backgroundColor="primary.light"
          color="primary.dark"
          icon={<Add />}
          handleClick={() => navigate(`/deployments/create`)}
        />
      </Box>

      <Box sx={{
        flex: 1,
        width: '100%',
        overflow: 'hidden'
      }}>
        <CustomTable
          rows={rows}
          columns={columns}
          containerHeight="100%"
          onView={handleView}
          onEdit={handleEdit}
          onDelete={(ids) => handleTableDelete(ids, rows)}
          onRestore={(ids) => handleTableRestore(ids, rows)}
          initialSortModel={[{ field: 'seq', sort: 'desc' }]}
          />
      </Box>
      {/* Confirmation Dialog */}
      <Dialog
        open={dialogState.open}
        onClose={handleDialogClose}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          {dialogState.title}
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            {dialogState.message}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleDialogClose}>Cancel</Button>
          <Button onClick={() => {
            dialogState.action();
            handleDialogClose();
          }} autoFocus>
            Confirm
          </Button>
        </DialogActions>
      </Dialog>

      {/* Error Dialog */}
      <Dialog
        open={errorDialog.open}
        onClose={handleErrorDialogClose}
        aria-labelledby="error-dialog-title"
      >
        <DialogTitle id="error-dialog-title">
          Error
        </DialogTitle>
        <DialogContent>
          <Alert severity="error">
            {errorDialog.message}
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleErrorDialogClose}>Close</Button>
        </DialogActions>
      </Dialog>

       {/* Snackbar for Release status errors */}
       <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleSnackbarClose} 
          severity={snackbar.severity} 
          sx={{ width: '80%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

       {/* Delete Confirmation Dialog */}
       <DeleteConfirmationDialog
        open={deleteConfirmation.open}
        isDeleted={deleteConfirmation.isDeleted}
        contentText={deleteConfirmation.seq}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
        />

        {/* Restore Confirmation Dialog */}
        <RestoreConfirmationDialog
          open={restoreConfirmation.open}
          contentText={`Are you sure you want to restore ${restoreConfirmation.seq}?`}
          onConfirm={confirmRestore}
          onCancel={cancelRestore}
        />

        {/* Loading Dialogs */}
        <LoadingDialog 
          open={isDeleteLoading} 
          loadingMessage="Please wait..." 
        />
        <LoadingDialog 
          open={isRestoreLoading} 
          loadingMessage="Please wait..." 
        />

        {/* Error Dialogs */}
        <ErrorDialog
          open={deleteError.open}
          errorMessage={deleteError.message}
          onClose={closeDeleteErrorDialog}
        />
        <ErrorDialog
          open={restoreError.open}
          errorMessage={restoreError.message}
          onClose={closeRestoreErrorDialog}
        />
    </Paper>
  );
};

export default AllDeployments;