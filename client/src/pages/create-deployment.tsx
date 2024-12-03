// src/pages/create-deployment.tsx
import { useGetIdentity, useList } from '@pankod/refine-core';
import { Box, CircularProgress } from '@pankod/refine-mui';
import { FieldValues, useForm } from '@pankod/refine-react-hook-form';
import { useNavigate } from '@pankod/refine-react-router-v6';
import DeploymentForm from 'components/common/DeploymentForm';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

// Define a type for the parts
interface Part {
  _id: number;
  partName: string;
  brandName: string;
}

const CreateDeployment = () => {
  const navigate = useNavigate();
  const { data: user } = useGetIdentity()
  const isError = false;

  const { data: partsResponse, isLoading: isPartLoading } = useList<Part>({
    resource: 'parts'
  });

  // Extract parts from response or fallback to an empty array
  const parts = partsResponse?.data || [];

  const { refineCore: { onFinish, formLoading }, register, handleSubmit } = useForm();

  const onFinishHandler = async (data: FieldValues) => {
    await onFinish({
        ...data,
        partName: data.partName, // Use the selected partName
        brandName: data.brandName, // Corrected to `brandName`
        email: user.email,
      });

        navigate('/deployments');
  };
  
  if (formLoading || isPartLoading) {
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

  return (
        <DeploymentForm
          type="Create"
          onFinish={onFinish}
          register={register}
          formLoading={formLoading}
          handleSubmit={handleSubmit}
          onFinishHandler={onFinishHandler}
          existingParts={parts}
        />
  );
};

export default CreateDeployment;