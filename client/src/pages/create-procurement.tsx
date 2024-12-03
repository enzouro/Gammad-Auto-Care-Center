// client\src\pages\create-procurement.tsx
import { useGetIdentity, useList } from '@pankod/refine-core';
import { Box, CircularProgress } from '@pankod/refine-mui';
import { FieldValues, useForm } from '@pankod/refine-react-hook-form';
import { useNavigate } from '@pankod/refine-react-router-v6';
import ProcurementForm from 'components/common/ProcurementForm';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

// Define a type for the parts
interface Part {
    _id: number;
    partName: string;
    brandName: string;
}

const CreateProcurement = () => {
  const navigate = useNavigate();
  const { data: user } = useGetIdentity();
  const isError = false;

  // Fetch existing parts
  const { data: partsResponse, isLoading: isPartLoading } = useList<Part>({
    resource: 'parts', // Adjust resource name based on your API
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

    navigate('/procurements');
  };

  if (formLoading || isPartLoading) {
    return (
      <LoadingDialog 
        open={formLoading}
        loadingMessage="Loading procurement form..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading procurement form"
      />
    );
  }

  return (
    <ProcurementForm
      type="Create"
      register={register}
      onFinish={onFinish}
      formLoading={formLoading}
      handleSubmit={handleSubmit}
      onFinishHandler={onFinishHandler}
      existingParts={parts} // Pass the fetched parts to the form
    />
  );
};

export default CreateProcurement;
