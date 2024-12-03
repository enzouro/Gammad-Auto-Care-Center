//client\src\pages\all-procurements.tsx
/* eslint-disable */
import { useMemo, useState } from 'react';
import { GridColDef, Box, Paper, Typography, CircularProgress, TextField, Stack, Button, ButtonGroup } from '@pankod/refine-mui';
import { Add } from '@mui/icons-material';
import { useNavigate } from '@pankod/refine-react-router-v6';
import { useTable } from '@pankod/refine-core';
import CustomButton from 'components/common/CustomButton';
import useDynamicHeight from 'hooks/useDynamicHeight';
import CustomTable from 'components/common/CustomTable';
import DeleteConfirmationDialog from 'components/common/DeleteConfirmationDialog';
import useDeleteWithConfirmation from 'hooks/useDeleteWithConfirmation';
import ErrorDialog from 'components/common/ErrorDialog';
import RestoreConfirmationDialog from 'components/common/RestoreConfirmationDialog';
import useRestoreWithConfirmation from 'hooks/useRestoreWithConfirmation';
import LoadingDialog from 'components/common/LoadingDialog';

const AllProcurements = () => {
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
    resource: 'procurements',
    redirectPath: '/procurements',
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
    resource: 'procurements',
    redirectPath: '/procurements',
  });

  const { 
    tableQueryResult: { data, isLoading, isError }
  } = useTable({
    resource: 'procurements',
    hasPagination: false,
  });

  const allProcurements = data?.data ?? [];

  // Filter the data based on search term and date range
  const filteredRows = useMemo(() => {
    return allProcurements.filter((procurement) => {
      const procurementDate = new Date(procurement.date);
      const matchesSearch = 
        !searchTerm || 
        procurement.supplierName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        procurement.tin.toString().includes(searchTerm) ||
        procurement.seq.toString().includes(searchTerm);
        
      const matchesDateRange = 
        (!startDate || procurementDate >= new Date(startDate)) &&
        (!endDate || procurementDate <= new Date(endDate));

        const matchesDeletedFilter = 
        (deletedFilter === 'active' && !procurement.deleted) || 
        (deletedFilter === 'deleted' && procurement.deleted);

      return matchesSearch && matchesDateRange && matchesDeletedFilter;
    });
  }, [allProcurements, searchTerm, startDate, endDate, deletedFilter]);

  const columns: GridColDef[] = [
  { field: 'seq', headerName: 'Seq', flex: 1, sortable: true },
  { field: 'date', headerName: 'Date', flex: 1 },
  { field: 'supplierName', headerName: 'Supplier', flex: 1 },
  { field: 'reference', headerName: 'Ref', flex: 1 },
  { field: 'tin', headerName: 'TIN', flex: 1 },
  { field: 'address', headerName: 'Address', flex: 1 },
  { field: 'partName', headerName: 'Part', flex: 1 },
  { field: 'brandName', headerName: 'Brand', flex: 1 },
  { field: 'quantityBought', headerName: 'Quantity', type: 'number', flex: 1 },
  { field: 'amount', headerName: 'Amount', type: 'number', flex: 1 },
  { field: 'netOfVAT', headerName: 'Net of VAT', type: 'number', flex: 1 },
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
    navigate(`/procurements/show/${id}`);
  };

  const handleEdit = (id: string) => {
    navigate(`/procurements/edit/${id}`);
  };

  const rows = filteredRows.map((procurement) => ({
    id: procurement._id,
    _id: procurement._id,
    seq: procurement.seq,
    date: new Date(procurement.date).toLocaleDateString(),
    supplierName: procurement.supplierName,
    reference: procurement.reference,
    tin: procurement.tin,
    address: procurement.address,
    partName: procurement.part?.partName,
    brandName: procurement.part?.brandName,
    quantityBought: procurement.quantityBought,
    amount: procurement.amount?.toFixed(2),
    netOfVAT: procurement.netOfVAT?.toFixed(2),
    inputVAT: procurement.inputVAT?.toFixed(2),
    isNonVat: procurement.isNonVat,
    noValidReceipt: procurement.noValidReceipt,
    deleted: procurement.deleted || false,
  }));

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading procurements data..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading procurements data"
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
              {!allProcurements.length ? 'There are no procurements' : 'All Procurements'}
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
        sx={{ flex: 1, alignItems: 'center'  }}
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
        handleClick={() => navigate(`/procurements/create`)}
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

export default AllProcurements;