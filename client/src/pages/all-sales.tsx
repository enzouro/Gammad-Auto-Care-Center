import { useState, useMemo } from 'react';
import { useTable } from '@pankod/refine-core';
import { 
  GridColDef, 
  Box, 
  Paper, 
  Typography, 
  CircularProgress, 
  Stack,
  TextField,
  Button,
  ButtonGroup
} from '@pankod/refine-mui';
import { Add } from '@mui/icons-material';
import { useNavigate } from '@pankod/refine-react-router-v6';
import CustomButton from 'components/common/CustomButton';
import CustomTable from 'components/common/CustomTable';
import useDynamicHeight from 'hooks/useDynamicHeight';
import DeleteConfirmationDialog from 'components/common/DeleteConfirmationDialog';
import RestoreConfirmationDialog from 'components/common/RestoreConfirmationDialog';
import useDeleteWithConfirmation from 'hooks/useDeleteWithConfirmation';
import useRestoreWithConfirmation from 'hooks/useRestoreWithConfirmation';
import ErrorDialog from 'components/common/ErrorDialog';
import LoadingDialog from 'components/common/LoadingDialog';

const AllSales = () => {
  const navigate = useNavigate();
  const containerHeight = useDynamicHeight();
  
  // Search and filter states
  const [searchTerm, setSearchTerm] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [deletedFilter, setDeletedFilter] = useState('active');
  
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
    resource: 'sales',
    redirectPath: '/sales',
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
    resource: 'sales',
    redirectPath: '/sales',
  });

  const { 
    tableQueryResult: { data, isLoading, isError }
  } = useTable({
    resource: 'sales',
    hasPagination: false,
  });

  const allSales = data?.data ?? [];

  // Rest of the existing filtering and data preparation logic remains the same
  const filteredRows = useMemo(() => {
    return allSales.filter((sale) => {
      const saleDate = new Date(sale.date);
      const matchesSearch = 
        !searchTerm || 
        sale.clientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        sale.tin.toString().includes(searchTerm) ||
        sale.seq.toString().includes(searchTerm);
        
      const matchesDateRange = 
        (!startDate || saleDate >= new Date(startDate)) &&
        (!endDate || saleDate <= new Date(endDate));

      const matchesDeletedFilter = 
        (deletedFilter === 'active' && !sale.deleted) || 
        (deletedFilter === 'deleted' && sale.deleted);

      return matchesSearch && matchesDateRange && matchesDeletedFilter;
    });
  }, [allSales, searchTerm, startDate, endDate, deletedFilter]);

  const columns: GridColDef[] = [
    { field: 'seq', headerName: 'Seq', flex: 1, sortable: true},
    { field: 'date', headerName: 'Date', flex: 1 },
    { field: 'clientName', headerName: 'Client Name', flex: 1 },
    { field: 'tin', headerName: 'TIN', flex: 1 },
    { field: 'amount', headerName: 'Amount', type: 'number', flex: 1 },
    { field: 'netOfVAT', headerName: 'Net of VAT', type: 'number', flex: 1 },
    { field: 'outputVAT', headerName: 'Output VAT', type: 'number', flex: 1 },
  ];

  const handleView = (id: string) => {
    navigate(`/sales/show/${id}`);
  };

  const handleEdit = (id: string) => {
    navigate(`/sales/edit/${id}`);
  };
  
  const rows = filteredRows.map((sale) => ({
    id: sale._id,
    _id: sale._id,
    seq: sale.seq,
    date: new Date(sale.date).toLocaleDateString(),
    clientName: sale.clientName,
    tin: sale.tin,
    amount: sale.amount,
    netOfVAT: sale.netOfVAT,
    outputVAT: sale.outputVAT,
    deleted: sale.deleted || false,
  }));

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading sales data..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading sales data"
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
        {!allSales.length ? 'No Sales Records' : 'All Sales'}
      </Typography>
      
      <Box sx={{ 
        p: 2,
        display: 'flex', 
        flexDirection: {xs: 'column', md: 'row'},
        gap: 2,
        alignItems: {xs: 'stretch', md: 'center'},
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
            placeholder="Search by client, TIN, or sequence"
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
                backgroundColor: deletedFilter === 'active' ? 'primary.light' : 'inherit', // Apply primary.light
                color: deletedFilter === 'active' ? 'primary.contrastText' : 'inherit',   // Ensure contrast text color
                '&:hover': {
                  backgroundColor: 'primary.main', // Add hover effect
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
          handleClick={() => navigate(`/sales/create`)}
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

export default AllSales;