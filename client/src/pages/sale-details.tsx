import { Delete, Edit } from '@mui/icons-material';
import { useDelete, useGetIdentity, useShow } from '@pankod/refine-core';
import { Box, CircularProgress, Button, Typography, Paper, Stack, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions } from '@pankod/refine-mui';
import { useNavigate, useParams } from '@pankod/refine-react-router-v6';
import CustomButton from 'components/common/CustomButton';
import DeleteConfirmationDialog from 'components/common/DeleteConfirmationDialog';
import useDeleteWithConfirmation from 'hooks/useDeleteWithConfirmation';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const SaleDetails = () => {
  const navigate = useNavigate();
  const { data: user } = useGetIdentity();
  const { id } = useParams();
  const { queryResult } = useShow();

  const { data, isLoading, isError } = queryResult;
  const saleDetails = data?.data ?? {};

  const {
    deleteConfirmation,
    handleDeleteClick,
    confirmDelete,
    cancelDelete,
  } = useDeleteWithConfirmation({
    resource: 'sales',
    redirectPath: '/sales'  // Specify where to navigate after successful deletion
  });

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading sales details..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading sales details"
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
        Sale Details
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
              <strong>Sequence:</strong> {saleDetails.seq}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Date:</strong> {saleDetails.date}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Client Name:</strong> {saleDetails.clientName}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>TIN:</strong> {saleDetails.tin}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Amount:</strong> {saleDetails.amount}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Net of VAT:</strong> {saleDetails.netOfVAT}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Output VAT:</strong> {saleDetails.outputVAT}
            </Typography>
          </Stack>

          <Box display="flex" gap={2} mt={3}>
            <CustomButton
              title="Edit"
              backgroundColor="warning.light"
              color="warning.dark"
              icon={<Edit />}
              handleClick={() => navigate(`/sales/edit/${saleDetails._id}`)}
            />
            <CustomButton
              title="Delete"
              backgroundColor="error.light"
              color="error.dark"
              icon={<Delete />}
              handleClick={() => handleDeleteClick(saleDetails._id, saleDetails.seq, saleDetails.deleted)}
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
            src={saleDetails.creator.avatar}
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
              {saleDetails.creator.name}
            </Typography>
            <Typography 
              fontSize={14} 
              sx={{ color: 'text.secondary' }}
            >
              {saleDetails.creator.email}
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

export default SaleDetails;