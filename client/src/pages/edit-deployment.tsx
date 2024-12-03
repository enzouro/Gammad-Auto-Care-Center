// src/pages/edit-deployment.tsx
import React, { useEffect } from 'react';
import { useGetIdentity, useOne, useList } from '@pankod/refine-core';
import { FieldValues, useForm } from '@pankod/refine-react-hook-form';
import { useNavigate, useParams } from '@pankod/refine-react-router-v6';
import DeploymentForm from 'components/common/DeploymentForm';
import { Box, CircularProgress, Typography } from '@pankod/refine-mui';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const EditDeployment = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: user } = useGetIdentity();
  const isError = false;

  const { data: deploymentData, isLoading: isDeploymentsLoading } = useOne({
    resource: 'deployments',
    id: id as string,
    queryOptions: {
      enabled: !!id, // Only fetch when id is available
    }
  });

  const { data: partsData, isLoading: isPartsLoading} = useList({
    resource: 'parts',
    queryOptions: {
      enabled: true, // Always fetch parts
    }
  });

  const {
    refineCore: { onFinish, formLoading },
    register,
    handleSubmit,
    setValue,
    reset,
  } = useForm({
    refineCoreProps: {
      resource: 'deployments',
      id: id as string,
      redirect: false,
      onMutationSuccess: () => {
        navigate('/deployments');
      },
    },
  });

  // Populate form when data is available
  useEffect(() => {
    if (deploymentData?.data) {
      const deployment = deploymentData.data;
      
      // Reset form with deployment data
      reset({
        seq: deployment.seq,
        date: deployment.date,
        clientName: deployment.clientName,
        vehicleModel: deployment.vehicleModel,
        arrivalDate: deployment.arrivalDate,
        deploymentStatus: deployment.deploymentStatus,
        deploymentDate: deployment.deploymentDate,
        releaseStatus: deployment.releaseStatus,
        releaseDate: deployment.releaseDate,
      });
    }
  }, [deploymentData, reset]);
  const onFinishHandler = async (data: FieldValues) => {
    // Format the data before submission
    const formattedData = {
      ...data,
      email: user.email,
      // Ensure dates are in the correct format
      date: data.date,
      arrivalDate: data.arrivalDate,
      deploymentDate: data.deploymentDate || null,
      releaseDate: data.releaseDate || null,
    };
    try {
      await onFinish(formattedData);
    } catch (error) {
      console.error('Error updating deployment:', error);
      // Handle error appropriately
    }
  };

  // Handle loading state
  if (formLoading || isDeploymentsLoading || isPartsLoading) {
    return (
      <LoadingDialog 
        open={formLoading}
        loadingMessage="Loading deployment form..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading deployment form"
      />
    );
  }
  
  // Check if we have the required data
  if (!deploymentData?.data || !partsData?.data) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100vh">
        <Typography variant="h6">
          No data available
        </Typography>
      </Box>
    );
  }
  return (
    <DeploymentForm
      type="Edit"
      register={register}
      onFinish={onFinish}
      formLoading={formLoading}
      handleSubmit={handleSubmit}
      onFinishHandler={onFinishHandler}
      existingParts={partsData.data}
      initialValues={deploymentData.data}
    />
  );
};

export default EditDeployment;