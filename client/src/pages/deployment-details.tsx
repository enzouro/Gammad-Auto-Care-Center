import { Delete, Edit } from '@mui/icons-material';
import { useDelete, useGetIdentity, useShow } from '@pankod/refine-core';
import { Box, CircularProgress, Button, Paper, Stack, Tooltip, Typography, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions } from '@pankod/refine-mui';
import { useNavigate, useParams } from '@pankod/refine-react-router-v6';
import CustomButton from 'components/common/CustomButton';
import DeleteConfirmationDialog from 'components/common/DeleteConfirmationDialog';
import useDeleteWithConfirmation from 'hooks/useDeleteWithConfirmation';
import { Key, useState } from 'react';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const DeploymentDetails = () => {
  const navigate = useNavigate();
  const { data: user } = useGetIdentity();
  const { id } = useParams();
  const { mutate } = useDelete();
  const { queryResult } = useShow();

  const {
    deleteConfirmation,
    handleDeleteClick,
    confirmDelete,
    cancelDelete,
  } = useDeleteWithConfirmation({
    resource: 'deployments',
    redirectPath: '/deployments'  // Specify where to navigate after successful deletion
  });

  const { data, isLoading, isError } = queryResult;
  const deploymentDetails = data?.data ?? {};

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading deployment details..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading deployment details"
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
        Deployment Details
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
              <strong>Seq:</strong> {deploymentDetails.seq}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Client Name:</strong> {deploymentDetails.clientName}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Date:</strong> {deploymentDetails.date}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Vehicle Model:</strong> {deploymentDetails.vehicleModel}
            </Typography>
            {/* Replace the single part section with this */}
          <Box>
            <Typography fontSize={16} fontWeight="bold" mb={2}>Parts Used:</Typography>
            {deploymentDetails.parts && deploymentDetails.parts.length > 0 ? (
              deploymentDetails.parts.map((partEntry: { part: { partName: any; brandName: any; }; quantityUsed: any; }, index: Key | null | undefined) => (
                <Box 
                  key={index} 
                  sx={{ 
                    ml: 2, 
                    mb: 2, 
                    pb: 1, 
                    borderBottom: index !== deploymentDetails.parts.length - 1 ? '1px solid rgba(0,0,0,0.06)' : 'none'
                  }}
                >
                  <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <strong>Part Name:</strong> {partEntry.part.partName}
                  </Typography>
                  <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <strong>Brand Name:</strong> {partEntry.part.brandName}
                  </Typography>
                  <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <strong>Quantity Used:</strong> {partEntry.quantityUsed}
                  </Typography>
                </Box>
              ))
            ) : (
              <Typography fontSize={16} color="text.secondary">
                No parts used in this deployment
              </Typography>
            )}
          </Box>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Deployment Status:</strong> {deploymentDetails.deploymentStatus ? 'Yes' : 'No'}
            </Typography>
            {deploymentDetails.deploymentStatus && (
              <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <strong>Deployment Date:</strong> {deploymentDetails.deploymentDate}
              </Typography>
            )}
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Repair Status:</strong> {deploymentDetails.repairStatus}
            </Typography>
            {deploymentDetails.repairStatus == 'Repaired' && (
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Repaired Date:</strong> {deploymentDetails.repairedDate}
            </Typography>)}
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Track Code:</strong> {deploymentDetails.trackCode}
            </Typography>
            <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <strong>Release Status:</strong> {deploymentDetails.releaseStatus ? 'Yes' : 'No'}
            </Typography>
            {deploymentDetails.releaseStatus && (
              <Typography fontSize={16} sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <strong>Release Date:</strong> {deploymentDetails.releaseDate}
              </Typography>
            )}
          </Stack>

          <Box display="flex" gap={2} mt={3}>
            <CustomButton
              title="Edit"
              backgroundColor="warning.light"
              color="warning.dark"
              icon={<Edit />}
              handleClick={() => navigate(`/deployments/edit/${deploymentDetails._id}`)}
            />
            <CustomButton
              title="Delete"
              backgroundColor="error.light"
              color="error.dark"
              icon={<Delete />}
              handleClick={() => handleDeleteClick(deploymentDetails._id, deploymentDetails.seq, deploymentDetails.deleted)} // Pass the id and seq
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
              src={deploymentDetails.creator.avatar}
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
                {deploymentDetails.creator.name}
              </Typography>
              <Typography 
                fontSize={14} 
                sx={{ color: 'text.secondary' }}
              >
                {deploymentDetails.creator.email}
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

export default DeploymentDetails;