// AllExpenses.tsx
/* eslint-disable */
import { useMemo, useState } from 'react';
import { useTable } from '@pankod/refine-core';
import { GridColDef, Box, Paper, Typography, CircularProgress, TextField, Stack, Button, ButtonGroup } from '@pankod/refine-mui';
import { Add } from '@mui/icons-material';
import { useNavigate } from '@pankod/refine-react-router-v6';
import CustomButton from 'components/common/CustomButton';
import useDynamicHeight from 'hooks/useDynamicHeight';
import CustomTable from 'components/common/CustomTable';
import DeleteConfirmationDialog from 'components/common/DeleteConfirmationDialog';
import useDeleteWithConfirmation from 'hooks/useDeleteWithConfirmation';
import ErrorDialog from 'components/common/ErrorDialog';
import useRestoreWithConfirmation from 'hooks/useRestoreWithConfirmation';
import RestoreConfirmationDialog from 'components/common/RestoreConfirmationDialog';
import LoadingDialog from 'components/common/LoadingDialog';

const AllExpenses = () => {
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
    resource: 'expenses',
    redirectPath: '/expenses',
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
    resource: 'expenses',
    redirectPath: '/expenses',
  });

  const { 
    tableQueryResult: { data, isLoading, isError }
  } = useTable({
    resource: 'expenses',
    hasPagination: false,
  });

  const allExpenses = data?.data ?? [];

  // Filter the data based on search term and date range
  const filteredRows = useMemo(() => {
    return allExpenses.filter((expense) => {
      const expenseDate = new Date(expense.date);
      const matchesSearch = 
        !searchTerm || 
        expense.supplierName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        expense.tin.toString().includes(searchTerm) ||
        expense.seq.toString().includes(searchTerm);
        
      const matchesDateRange = 
        (!startDate || expenseDate >= new Date(startDate)) &&
        (!endDate || expenseDate <= new Date(endDate));

        const matchesDeletedFilter = 
        (deletedFilter === 'active' && !expense.deleted) || 
        (deletedFilter === 'deleted' && expense.deleted);

      return matchesSearch && matchesDateRange && matchesDeletedFilter;
    });
  }, [allExpenses, searchTerm, startDate, endDate, deletedFilter]);

  const columns: GridColDef[] = [
    { field: 'seq', headerName: 'Seq', flex: 1, sortable: true  },
    { field: 'date', headerName: 'Date', flex: 1 },
    { field: 'supplierName', headerName: 'Supplier Name', flex: 1 },
    { field: 'ref', headerName: 'Reference', flex: 1 },
    { field: 'tin', headerName: 'TIN', flex: 1 },
    { field: 'address', headerName: 'Address', flex: 1 },
    { field: 'description', headerName: 'Description', flex: 1 },
    { field: 'amount', headerName: 'Amount', type: 'number', flex: 1 },
    { field: 'netOfVAT', headerName: 'Net of VAT', type: 'number', flex: 1 },
    { field: 'inputVAT', headerName: 'Input VAT', type: 'number', flex: 1 },
    { 
      field: 'isNonVat', 
      headerName: 'Non-VAT', 
      flex: 1,
      renderCell: (params) => (
        <Typography
          variant="body2"
          sx={{
            color: params.row.isNonVat ? 'warning.main' : 'text.secondary',
            fontWeight: params.row.isNonVat ? 'bold' : 'normal'
          }}
        >
          {params.row.isNonVat ? 'No VAT' : 'N/A'}
        </Typography>
      )
    },
    { 
      field: 'noValidReceipt', 
      headerName: 'No Receipt', 
      flex: 1,
      renderCell: (params) => (
        <Typography
          variant="body2"
          sx={{
            color: params.row.noValidReceipt ? 'warning.main' : 'text.secondary',
            fontWeight: params.row.noValidReceipt ? 'bold' : 'normal'
          }}
        >
          {params.row.noValidReceipt ? 'No RCPT' : 'N/A'}
        </Typography>
      )
    },
  ];

  const handleView = (id: string) => {
    navigate(`/expenses/show/${id}`);
  };

  const handleEdit = (id: string) => {
    navigate(`/expenses/edit/${id}`);
  };

  const rows = filteredRows.map((expense) => ({
    id: expense._id,
    _id: expense._id,
    seq: expense.seq,
    date: new Date(expense.date).toLocaleDateString(),
    supplierName: expense.supplierName,
    ref: expense.ref,
    tin: expense.tin,
    address: expense.address,
    description: expense.description,
    amount: expense.amount?.toFixed(2),
    netOfVAT: expense.netOfVAT?.toFixed(2),
    inputVAT: expense.inputVAT?.toFixed(2),
    isNonVat: expense.isNonVat,
    noValidReceipt: expense.noValidReceipt,
    deleted: expense.deleted || false,

  }));

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading expenses data..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading expenses data"
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
        {!allExpenses.length ? 'No Expenses Records' : 'All Expenses'}
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
          sx={{ flex: 1 }}
        >
          <TextField
            size="small"
            label="Search"
            placeholder="Search by supplier, TIN, or sequence"
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
          handleClick={() => navigate(`/expenses/create`)}
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

export default AllExpenses;
