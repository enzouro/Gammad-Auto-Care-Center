/* eslint-disable */
// src/pages/edit-expense.tsx
import React from 'react';
import { useGetIdentity, useOne } from '@pankod/refine-core';
import { FieldValues, useForm } from '@pankod/refine-react-hook-form';
import { useNavigate, useParams } from '@pankod/refine-react-router-v6';
import ExpenseForm from 'components/common/ExpenseForm';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const EditExpense = () => {
  const navigate = useNavigate();
  const { id } = useParams();
  const { data: user } = useGetIdentity();
  const isError = false;

  const { data: expenseData, isLoading: isExpenseLoading } = useOne({
    resource: 'expenses',
    id: id as string,
  });

  const {
    refineCore: { onFinish, formLoading }, register, handleSubmit, setValue, } = useForm({
    refineCoreProps: {
      resource: 'expenses',
      id: id as string,
      redirect: false,
      onMutationSuccess: () => {
        navigate('/expenses');
      },
    },
  });

  const onFinishHandler = async (data: FieldValues) => {
    await onFinish({
      ...data,
      email: user.email,
    });
  };

  if (formLoading || isExpenseLoading) {
    return (
      <LoadingDialog 
        open={formLoading}
        loadingMessage="Loading expenses form..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading expenses form"
      />
    );
  }

  return (
    <ExpenseForm
      type="Edit"
      register={register}
      onFinish={onFinish}
      formLoading={formLoading}
      handleSubmit={handleSubmit}
      onFinishHandler={onFinishHandler}
      initialValues={expenseData?.data}
    />
  );
};

export default EditExpense;
