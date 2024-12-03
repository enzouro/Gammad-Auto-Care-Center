import {
  Box,
  FormControl,
  InputLabel,
  OutlinedInput,
  Paper,
  Typography
} from '@pankod/refine-mui';

import { FormPropsSale } from 'interfaces/common';
import { useState } from 'react';
import { Close, Publish } from '@mui/icons-material';
import { useNavigate } from '@pankod/refine-react-router-v6';
import CustomButton from './CustomButton';
import useNextSequence from 'hooks/useNextSequence';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';
import { TextField } from '@mui/material';

const getTodayDate = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const SaleForm = ({ type, register, handleSubmit, formLoading, onFinishHandler, initialValues }: FormPropsSale) => {
  const [amount, setAmount] = useState(initialValues?.amount || 0);
  const [netOfVAT, setNetOfVAT] = useState(initialValues?.netOfVAT || 0);
  const [outputVAT, setOutputVAT] = useState(initialValues?.outputVAT || 0);
  const navigate = useNavigate();
  const isError = false;

  // Use the custom hook for sequence logic
  const { currentSeq, isLoading: sequenceLoading } = useNextSequence({
    resource: "sales",
    type: type as "Create" | "Edit",
    initialValues,
  });

  // Function to calculate VAT components
  const calculateVATComponents = (totalAmount: number) => {
    const netAmount = totalAmount * (100/112);
    const vatAmount = totalAmount * (12/112);

    return {
      netAmount: Number(netAmount.toFixed(2)),
      vatAmount: Number(vatAmount.toFixed(2))
    };
  };

  const handleAmountChange = (e: { target: { value: string; }; }) => {
    const newAmount = parseFloat(e.target.value) || 0;
    setAmount(newAmount);

    // Calculate VAT components
    const { netAmount, vatAmount } = calculateVATComponents(newAmount);
    setNetOfVAT(netAmount);
    setOutputVAT(vatAmount);
  };

  const onSubmit = (data: any) => {
    const updatedData = { 
      ...data,
      seq: currentSeq,
      amount: parseFloat(amount),
      outputVAT: parseFloat(outputVAT),
      netOfVAT: parseFloat(netOfVAT),
      // Ensure empty strings are allowed for clientName and tin
      clientName: data.clientName || '',
      tin: data.tin || ''
    };
    
    onFinishHandler(updatedData);
  };

  if (formLoading || sequenceLoading || currentSeq === null) {
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
    <Paper 
      elevation={2} 
      sx={{ 
        padding: '32px',
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
        {type} a Sale
      </Typography>

      <form
        style={{ 
          width: '100%', 
          display: 'flex', 
          flexDirection: 'column', 
          gap: '24px' 
        }}
        onSubmit={handleSubmit(onSubmit)}
      >
        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' }, 
          gap: 2,
          '& .MuiFormControl-root': { flex: 1 }
        }}>
          <FormControl>
            <InputLabel htmlFor="seq">Sequence Number</InputLabel>
            <OutlinedInput
              id="seq"
              type="number"
              label="Sequence Number"
              value={currentSeq}
              disabled
              {...register('seq')}
            />
          </FormControl>

          <FormControl>
            <InputLabel htmlFor="date">Date</InputLabel>
            <OutlinedInput
              id="date"
              type="date"
              label="Date"
              {...register('date')}
              defaultValue={initialValues?.date || getTodayDate()}
            />
          </FormControl>
        </Box>

        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' }, 
          gap: 2,
          '& .MuiFormControl-root': { flex: 1 }
        }}>
          <TextField
            label="Client Name"
            variant="outlined"
            {...register('clientName')}
            defaultValue={initialValues?.clientName || ''}
          />
          <TextField
            label="TIN"
            variant="outlined"
            {...register('tin')}
            defaultValue={initialValues?.tin || ''}
          />
        </Box>

        <TextField
          label="Amount"
          variant="outlined"
          required
          type="number"
          value={amount}
          onChange={handleAmountChange}
          inputProps={{ step: "0.01", required: true }}
        />

        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' }, 
          gap: 2,
          alignItems: 'center',
          '& .MuiFormControl-root': { flex: 1 }
        }}>
          <TextField
            label="Net of VAT"
            variant="outlined"
            type="number"
            value={netOfVAT.toFixed(2)}
            InputProps={{ readOnly: true }}
          />
          <TextField
            label="Output VAT"
            variant="outlined"
            type="number"
            value={outputVAT.toFixed(2)}
            InputProps={{ readOnly: true }}
            inputProps={{ step: "0.01" }}
          />
        </Box>

        <Box display="flex" justifyContent="center" gap={2} mt={3}>
          <CustomButton
            type="submit"
            title="Publish"
            backgroundColor="primary.light"
            color="primary.dark"
            icon={<Publish />}
          />
          <CustomButton
            title="Close"
            backgroundColor="error.light"
            color="error.dark"
            icon={<Close />}
            handleClick={() => navigate('/sales')}
          />
        </Box>
      </form>
    </Paper>
  );
};

export default SaleForm;