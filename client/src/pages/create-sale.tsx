// src/pages/create-sale.tsx
import React from 'react';
import { useGetIdentity } from '@pankod/refine-core';
import { Box, CircularProgress } from '@pankod/refine-mui';
import { FieldValues, useForm } from '@pankod/refine-react-hook-form';
import { useNavigate } from '@pankod/refine-react-router-v6';
import SaleForm from 'components/common/SaleForm';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const CreateSale = () => {
  const navigate = useNavigate();
  const { data: user } = useGetIdentity();
  const isError = false;

  const {
    refineCore: { onFinish, formLoading },
    register,
    handleSubmit,
  } = useForm();

  const onFinishHandler = async (data: FieldValues) => {
    await onFinish({
      ...data,
      email: user.email,
    });
    navigate('/sales');
  };

  if (formLoading) {
    return (
      <LoadingDialog 
        open={formLoading}
        loadingMessage="Loading sales form..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading sales form"
      />
    );
  }

  return (
    <SaleForm
      type="Create"
      register={register}
      onFinish={onFinish}
      formLoading={formLoading}
      handleSubmit={handleSubmit}
      onFinishHandler={onFinishHandler}
    />
  );
};

export default CreateSale;

