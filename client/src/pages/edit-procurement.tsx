// client\src\pages\edit-procurement.tsx
/* eslint-disable */
import React from 'react';
import { useGetIdentity, useOne, useList } from '@pankod/refine-core';
import { FieldValues, useForm } from '@pankod/refine-react-hook-form';
import { useNavigate, useParams } from '@pankod/refine-react-router-v6';
import ProcurementForm from 'components/common/ProcurementForm';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const EditProcurement = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: user } = useGetIdentity();
  const isError = false;

  const { data: procurementData, isLoading: isProcurementLoading } = useOne({
    resource: 'procurements',
    id: id as string,
  });

  const { data: partsData, isLoading: isPartsLoading } = useList({
    resource: 'parts',
  });

  const {
    refineCore: { onFinish, formLoading },
    register,
    handleSubmit,
    setValue,
  } = useForm({
    refineCoreProps: {
      resource: 'procurements',
      id: id as string,
      redirect: false,
      onMutationSuccess: () => {
        navigate('/procurements');
      },
    },
  });

  const onFinishHandler = async (data: FieldValues) => {
    await onFinish({
      ...data,
      email: user.email,
    });
  };

  if (formLoading || isProcurementLoading || isPartsLoading) {
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
        errorMessage="Error loading procurment form"
      />
    );
  }

  return (
    <ProcurementForm
      type="Edit"
      register={register}
      onFinishHandler={onFinishHandler}
      onFinish={onFinish}
      formLoading={formLoading}
      handleSubmit={handleSubmit}
      existingParts={partsData?.data || []}
      initialValues={procurementData?.data}
    />
  );
};

export default EditProcurement;
