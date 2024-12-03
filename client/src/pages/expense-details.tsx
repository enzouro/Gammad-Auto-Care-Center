import { Delete, Edit } from '@mui/icons-material';
import {  useDelete, useGetIdentity, useShow } from '@pankod/refine-core';
import { Box, Button, CircularProgress, Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle, Paper, Stack, Typography } from '@pankod/refine-mui';
import { useNavigate, useParams } from '@pankod/refine-react-router-v6';
import CustomButton from 'components/common/CustomButton';
import DeleteConfirmationDialog from 'components/common/DeleteConfirmationDialog';
import useDeleteWithConfirmation from 'hooks/useDeleteWithConfirmation';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const ExpenseDetails = () => {
  const navigate = useNavigate();
  const { data: user } = useGetIdentity();
  const { id } = useParams();
  const { queryResult } = useShow();

  const { data, isLoading, isError } = queryResult;
  const expenseDetails = data?.data ?? {};

  const {
    deleteConfirmation,
    handleDeleteClick,
    confirmDelete,
    cancelDelete,
  } = useDeleteWithConfirmation({
    resource: 'expenses',
    redirectPath: '/expenses'  // Specify where to navigate after successful deletion
  });

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading expenses details..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading expenses details"
      />
    );
  }

  return (
    <Paper 
      elevation={2} 
      sx={{ 
        padding: '24px',
        margin: '24px auto',
        maxWidth: '1000px',
        borderRadius: '16px',
        transition: 'transform 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-2px)'
        }
      }}
    >
      <Typography 
          variant="h4" 
          sx={{ 
          textAlign: 'left',
          mb: 4,
          fontWeight: 600,
          }}
      >
        Expense Details
      </Typography>

      <Box display="flex" flexDirection={{ xs: 'column', lg: 'row' }} gap={4}>
        {/* Main Details Section */}
        <Box flex={1} maxWidth={764}>
          <Stack 
            spacing={2.5} 
            sx={{
            px: 3,
            mb: 4,
            '& .MuiTypography-root': {
              position: 'relative',
              paddingBottom: '8px',
              borderBottom: '1px solid rgba(0,0,0,0.06)',
              '&:last-child': {
                borderBottom: 'none'
              }
            }
            }}
          >
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Sequence:</strong> {expenseDetails.seq}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Date:</strong> {new Date(expenseDetails.date).toLocaleDateString()}
            </Typography>
            {!expenseDetails.noValidReceipt && (
              <>
                <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Supplier Name:</strong> {expenseDetails.supplierName}
                </Typography>
                <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Reference:</strong> {expenseDetails.ref}
                </Typography>
                <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>TIN:</strong> {expenseDetails.tin}
                </Typography>
                <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Address:</strong> {expenseDetails.address}
                </Typography>
              </>
            )}
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Description:</strong> {expenseDetails.description}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Amount:</strong> {expenseDetails.amount?.toFixed(2)}
            </Typography>
            {!expenseDetails.isNonVat && !expenseDetails.noValidReceipt && (
              <>
                <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Net of VAT:</strong> {expenseDetails.netOfVAT?.toFixed(2)}
                </Typography>
                <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>Input VAT:</strong> {expenseDetails.inputVAT?.toFixed(2)}
                </Typography>
              </>
            )}
          </Stack>

          <Box display="flex" gap={2} mt={3}>
            <CustomButton
              title="Edit"
              backgroundColor="warning.light"
              color="warning.dark"
              icon={<Edit />}
              handleClick={() => navigate(`/expenses/edit/${expenseDetails._id}`)}
            />
            <CustomButton
              title="Delete"
              backgroundColor="error.light"
              color="error.dark"
              icon={<Delete />}
              handleClick={() => handleDeleteClick(expenseDetails._id, expenseDetails.seq, expenseDetails.deleted)} // Pass the id and seq
            />
          </Box>
        </Box>

        {/* Creator Information Section */}
        <Box 
          maxWidth={326} 
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center',
            mt: { xs: 3, lg: 0 },
            p: 3,
            borderLeft: { xs: 'none', lg: '1px solid rgba(0,0,0,0.08)' },
            borderTop: { xs: '1px solid rgba(0,0,0,0.08)', lg: 'none' }
          }}
        >
          <Typography 
            fontSize={18} 
            fontWeight={600} 
            mb={2}
            color="primary.main"
          >
            Created By
          </Typography>
          <Box 
            component="img"
            src={expenseDetails.creator.avatar}
            alt="Creator Avatar"
            sx={{
              width: 50,
              height: 50,
              borderRadius: '50%',
              objectFit: 'cover',
              border: '2px solid',
              borderColor: 'primary.light'
            }}
          />
          <Box textAlign="center">
            <Typography fontSize={16} fontWeight={600}>
              {expenseDetails.creator.name}
            </Typography>
            <Typography 
              fontSize={14} 
              sx={{ color: 'text.secondary' }}
            >
              {expenseDetails.creator.email}
            </Typography>
          </Box>
        </Box>
      </Box>


       {/* Delete Confirmation Dialog */}
       <DeleteConfirmationDialog
        open={deleteConfirmation.open}
        isDeleted={deleteConfirmation.isDeleted}
        contentText={deleteConfirmation.seq}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
    </Paper>
  );
};

export default ExpenseDetails;