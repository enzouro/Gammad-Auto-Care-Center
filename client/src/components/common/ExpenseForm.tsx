import { useState } from 'react';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Paper,
  OutlinedInput,
  CircularProgress,
  TextField,
  FormControlLabel,
  Checkbox} from '@pankod/refine-mui';
import { FormPropsExpense } from 'interfaces/common';
import { useNavigate } from '@pankod/refine-react-router-v6';
import { Close, Publish } from '@mui/icons-material';
import CustomButton from './CustomButton';
import useNextSequence from 'hooks/useNextSequence';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

const getTodayDate = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const ExpenseForm = ({ type, register, handleSubmit, formLoading, onFinishHandler, initialValues }: FormPropsExpense) => {
  const [noValidReceipt, setNoValidReceipt] = useState(initialValues?.noValidReceipt || false);
  const [isNonVat, setIsNonVat] = useState(initialValues?.isNonVat || false);
  const [amount, setAmount] = useState(initialValues?.amount || 0);
  const [netOfVAT, setNetOfVAT] = useState(initialValues?.netOfVAT || 0);
  const [inputVAT, setInputVAT] = useState(initialValues?.inputVAT || 0);
  const navigate = useNavigate();
  const isError = false;

  // Use the custom hook for sequence logic
  const { currentSeq, isLoading: sequenceLoading } = useNextSequence({
    resource: "expenses",
    type: type as "Create" | "Edit", // Assert type explicitly
    initialValues,
  });

  // Function to calculate VAT components
  const calculateVATComponents = (totalAmount: number, isNoValidReceipt: boolean, isNonVAT: boolean) => {
    if (isNoValidReceipt) {
      return {
        netAmount: 0,  // Set to 0 when no valid receipt
        vatAmount: 0
      };
    }

    if (isNonVAT) {
      return {
        netAmount: totalAmount,  // Set to total amount when non-VAT
        vatAmount: 0
      };
    }

    // For valid VAT receipts:
    // Net Amount = Amount × (100/112)
    // VAT = Amount × (12/112)
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

    // Calculate VAT components based on current state
    const { netAmount, vatAmount } = calculateVATComponents(newAmount, noValidReceipt, isNonVat);
    setNetOfVAT(netAmount);
    setInputVAT(vatAmount);
  };

  const handleNoValidReceiptChange = (e: { target: { checked: any; }; }) => {
    const checked = e.target.checked;
    setNoValidReceipt(checked);
    
    if (checked) {
      // When no valid receipt is checked, set non-VAT to true and values to 0
      setIsNonVat(true);
      setNetOfVAT(0);  // Always set to 0 when no valid receipt
      setInputVAT(0);
    } else {
      // When unchecking no valid receipt, reset non-VAT and recalculate
      setIsNonVat(false);
      const { netAmount, vatAmount } = calculateVATComponents(amount, false, false);
      setNetOfVAT(netAmount);
      setInputVAT(vatAmount);
    }
  };

  const handleNonVatChange = (e: { target: { checked: any; }; }) => {
    const checked = e.target.checked;
    setIsNonVat(checked);
    
    if (checked) {
      // When non-VAT is checked and no valid receipt is not checked
      if (!noValidReceipt) {
        setNetOfVAT(amount);
      }
      // Always set input VAT to 0 when non-VAT
      setInputVAT(0);
    } else {
      // When unchecking non-VAT, recalculate based on current receipt status
      const { netAmount, vatAmount } = calculateVATComponents(amount, noValidReceipt, false);
      setNetOfVAT(netAmount);
      setInputVAT(vatAmount);
    }
  };

  const onSubmit = (data: any) => {
    const updatedData = { 
      ...data,
      isNonVat,
      seq: currentSeq,
      noValidReceipt,
      amount: parseFloat(amount),
      inputVAT: parseFloat(inputVAT),
      netOfVAT: parseFloat(netOfVAT)
    };
    
    if (noValidReceipt) {
      updatedData.supplierName = "N/A";
      updatedData.ref = "N/A";
      updatedData.tin = "N/A";
      updatedData.address = "N/A";
      updatedData.inputVAT = 0;
      updatedData.isNonVat = true;
      updatedData.netOfVAT = 0;  // Always 0 when no valid receipt
    } else if (isNonVat) {
      updatedData.inputVAT = 0;
      updatedData.netOfVAT = updatedData.amount;
    }
    
    onFinishHandler(updatedData);
  };

  if (formLoading || sequenceLoading || currentSeq === null) {
    return (
      <LoadingDialog 
        open={formLoading}
        loadingMessage="Loading expense form..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading expense form"
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
        {type} an Expense
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
              {...register('date', { required: true })}
              defaultValue={initialValues?.date || getTodayDate()}
            />
          </FormControl>
        </Box>

        <Box sx={{ 
          display: 'flex', 
          gap: 2,
          alignItems: 'center',
        }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={noValidReceipt}
                onChange={handleNoValidReceiptChange}
              />
            }
            label="No Valid Receipt"
          />
        </Box>

        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' }, 
          gap: 2,
          '& .MuiFormControl-root': { flex: 1 }
        }}>
        {!noValidReceipt && (
          <>
            <TextField
              label="Supplier's Company Name"
              variant="outlined"
              {...register('supplierName', { required: !noValidReceipt })}
              defaultValue={initialValues?.supplierName || ''}
            />
            <TextField
              label="Reference"
              variant="outlined"
              {...register('ref', { required: !noValidReceipt })}
              defaultValue={initialValues?.ref || ''}
            />
            <TextField
              label="TIN"
              variant="outlined"
              {...register('tin', { required: !noValidReceipt })}
              defaultValue={initialValues?.tin || ''}
            />
            <TextField
              label="Address"
              variant="outlined"
              {...register('address', { required: !noValidReceipt })}
              defaultValue={initialValues?.address || ''}
            />
          </>
        )}
        </Box>

        <TextField
          label="Description"
          variant="outlined"
          {...register('description', { required: true })}
          defaultValue={initialValues?.description || ''}
        />

        <TextField
          label="Amount"
          variant="outlined"
          required
          type="number"
          value={amount}
          onChange={handleAmountChange}
          inputProps={{ step: "0.01", required: true }}
        />

        <FormControlLabel
          control={
            <Checkbox
              checked={isNonVat}
              onChange={handleNonVatChange}
              disabled={noValidReceipt}
            />
          }
          label="Non-VAT"
        />

        <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' }, 
          gap: 2,
          alignItems: 'center',
          '& .MuiFormControl-root': { flex: 1 }
        }}>

        {!noValidReceipt && (
        <>
          <TextField
            label="Net of VAT"
            variant="outlined"
            type="number"
            value={netOfVAT.toFixed(2)}
            InputProps={{ readOnly: true }}
          />
          <TextField
            label="Input VAT"
            variant="outlined"
            type="number"
            value={inputVAT.toFixed(2)}
            InputProps={{ readOnly: true }}
            inputProps={{ step: "0.01" }}
          />
        </>
        )}
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
            handleClick={() => navigate('/expenses')}
          />
        </Box>
      </form>
    </Paper>
  );
};

export default ExpenseForm;