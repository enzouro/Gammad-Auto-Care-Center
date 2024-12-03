//client\src\components\common\ProcurementForm.tsx
import { Close, Publish } from '@mui/icons-material';
import {
  Box,
  Checkbox,
  CircularProgress,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  OutlinedInput,
  Paper,
  Select,
  SelectChangeEvent,
  TextField,
  Typography
} from '@pankod/refine-mui';
import { useNavigate } from '@pankod/refine-react-router-v6';
import { FormPropsProcurement } from 'interfaces/common';
import { useEffect, useState } from 'react';
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

const ProcurementForm = ({ 
  type, 
  register, 
  handleSubmit, 
  formLoading, 
  onFinishHandler,
  existingParts,
  initialValues 
}: FormPropsProcurement) => {
  const [selectedPart, setSelectedPart] = useState('');
  const [newPartName, setNewPartName] = useState('');
  const [newBrandName, setNewBrandName] = useState('');
  const [noValidReceipt, setNoValidReceipt] = useState(initialValues?.noValidReceipt || false);
  const [isNonVat, setIsNonVat] = useState(initialValues?.isNonVat || false);
  const [amount, setAmount] = useState(initialValues?.amount || 0);
  const [netOfVAT, setNetOfVAT] = useState(initialValues?.netOfVAT || 0);
  const [inputVAT, setInputVAT] = useState(initialValues?.inputVAT || 0);
  const navigate = useNavigate();
  
  const isError = false;

  const [formErrors, setFormErrors] = useState<{
    selectedPart?: string;
    supplierName?: string;
    reference?: string;
    tin?: string;
    address?: string;
    description?: string;
    quantityBought?: string;
    amount?: string;
  }>({});
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  // Use the custom hook for sequence logic
  const { currentSeq, isLoading: sequenceLoading } = useNextSequence({
    resource: "procurements",
    type: type as "Create" | "Edit", // Assert type explicitly
    initialValues,
  });

  useEffect(() => {
    if (initialValues && initialValues.partName && initialValues.brandName) {
      setSelectedPart(`${initialValues.partName}|${initialValues.brandName}`);
    }
  }, [initialValues]);

  const handlePartChange = (event: SelectChangeEvent<string>) => {
    setSelectedPart(event.target.value as string);

    setFormErrors(prev => ({ ...prev, selectedPart: undefined }));
  };

  // Helper function to calculate VAT and net amount
  const calculateVATComponents = (totalAmount: number, isNoReceipt: boolean, isNonVAT: boolean) => {
    if (isNoReceipt) {
      return { netAmount: 0, vatAmount: 0 };
    }
    if (isNonVAT) {
      return { netAmount: totalAmount, vatAmount: 0 };
    }
    const netAmount = totalAmount * (100 / 112);
    const vatAmount = totalAmount * (12 / 112);
    return { netAmount: Number(netAmount.toFixed(2)), vatAmount: Number(vatAmount.toFixed(2)) };
  };

  const handleAmountChange = (e: { target: { value: string } }) => {
    const newAmount = parseFloat(e.target.value) || 0;
    setAmount(newAmount);
    const { netAmount, vatAmount } = calculateVATComponents(newAmount, noValidReceipt, isNonVat);
    setNetOfVAT(netAmount);
    setInputVAT(vatAmount);

    setFormErrors(prev => ({ ...prev, amount: undefined }));
  };

  const handleNoValidReceiptChange = (e: { target: { checked: any } }) => {
    const checked = e.target.checked;
    setNoValidReceipt(checked);

    if (checked) {
      setIsNonVat(true);
      setNetOfVAT(0);
      setInputVAT(0);
    } else {
      const { netAmount, vatAmount } = calculateVATComponents(amount, false, isNonVat);
      setNetOfVAT(netAmount);
      setInputVAT(vatAmount);
    }
  };

  const handleNonVatChange = (e: { target: { checked: any } }) => {
    const checked = e.target.checked;
    setIsNonVat(checked);

    if (checked && !noValidReceipt) {
      setNetOfVAT(amount);
      setInputVAT(0);
    } else if (!checked) {
      const { netAmount, vatAmount } = calculateVATComponents(amount, noValidReceipt, false);
      setNetOfVAT(netAmount);
      setInputVAT(vatAmount);
    }
  };

  const validateForm = (data: Record<string, any>): boolean => {
    const errors: typeof formErrors = {};

    // Validate Part Selection
    if (!selectedPart) {
      errors.selectedPart = 'Please select a part or add a new part';
    } else if (selectedPart === 'new') {
      if (!newPartName.trim()) {
        errors.selectedPart = 'New Part Name is required';
      }
      if (!newBrandName.trim()) {
        errors.selectedPart = errors.selectedPart || 'New Brand Name is required';
      }
    }

    // Validate Supplier Details if receipt is valid
    if (!noValidReceipt) {
      if (!data.supplierName?.trim()) {
        errors.supplierName = 'Supplier Name is required';
      }
      if (!data.reference?.trim()) {
        errors.reference = 'Reference is required';
      }
      if (!data.tin?.trim()) {
        errors.tin = 'TIN is required';
      }
      if (!data.address?.trim()) {
        errors.address = 'Address is required';
      }
    }

    // Validate Description
    if (!data.description?.trim()) {
      errors.description = 'Description is required';
    }

    // Validate Quantity Bought
    if (!data.quantityBought || data.quantityBought <= 0) {
      errors.quantityBought = 'Quantity must be greater than 0';
    }

    // Validate Amount
    if (amount <= 0) {
      errors.amount = 'Amount must be greater than 0';
    }

    // Set errors and return validation status
    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  };


  const onSubmit = (data: Record<string, any>) => {
    if (!validateForm(data)) {
      setOpenSnackbar(true);
      setSnackbarMessage('Please fill in all required fields correctly');
      return;
    }

    const updatedData: Record<string, any> = {
      ...data,
      seq: currentSeq // Add the sequence number
    };
    if (selectedPart === 'new') {
      updatedData.partName = newPartName;
      updatedData.brandName = newBrandName;
    } else {
      const [partName, brandName] = selectedPart.split('|');
      updatedData.partName = partName;
      updatedData.brandName = brandName;
    }

    const formData = {
      ...updatedData,
      isNonVat,
      noValidReceipt,
      amount,
      inputVAT,
      netOfVAT,
      supplierName: noValidReceipt ? "N/A" : updatedData.supplierName,
      reference: noValidReceipt ? "N/A" : updatedData.reference,
      tin: noValidReceipt ? "N/A" : updatedData.tin,
      address: noValidReceipt ? "N/A" : updatedData.address,
    };

    if (noValidReceipt) {
      formData.inputVAT = 0;
      formData.netOfVAT = 0;
    } else if (isNonVat) {
      formData.inputVAT = 0;
      formData.netOfVAT = amount;
    }

    onFinishHandler(formData);
  };

  if (formLoading || sequenceLoading || currentSeq === null) {
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
    <Paper
      elevation={2}
      sx={{
        padding: '32px',
        margin: '24px auto',
        maxWidth: '1000px',
        borderRadius: '16px',
        transition: 'transform 0.2s ease-in-out',
        '&:hover': {
          transform: 'translateY(-2px)',
        },
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
        {type} a Procurement
      </Typography>
  
      <form
        style={{
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          gap: '24px',
        }}
        onSubmit={handleSubmit(onSubmit)}
      >
        {/* Section for Sequence Number and Date */}
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            gap: 2,
            '& .MuiFormControl-root': { flex: 1 },
          }}
        >
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
  
        {/* Conditional section for Supplier Details */}
        {!noValidReceipt && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              gap: 2,
              '& .MuiFormControl-root': { flex: 1 },
            }}
          >
            <FormControl error={!!formErrors.supplierName}>
                <InputLabel htmlFor="supplierName">Supplier Name</InputLabel>
                <OutlinedInput
                  id="supplierName"
                  label="Supplier Name"
                  {...register('supplierName', { required: true })}
                  defaultValue={initialValues?.supplierName || ""}
                  error={!!formErrors.supplierName}
                />
                {formErrors.supplierName && (
                  <Typography color="error" variant="caption">
                    {formErrors.supplierName}
                  </Typography>
                )}
              </FormControl>
  
            <FormControl>
              <InputLabel htmlFor="reference">Reference</InputLabel>
              <OutlinedInput
                id="reference"
                label="Reference"
                {...register('reference', { required: true })}
                defaultValue={initialValues?.reference || ""}
              />
            </FormControl>
  
            <FormControl>
              <InputLabel htmlFor="tin">TIN</InputLabel>
              <OutlinedInput
                id="tin"
                label="TIN"
                {...register('tin', { required: true })}
                defaultValue={initialValues?.tin || ""}
              />
            </FormControl>
  
            <FormControl>
              <InputLabel htmlFor="address">Address</InputLabel>
              <OutlinedInput
                id="address"
                label="Address"
                {...register('address', { required: true })}
                defaultValue={initialValues?.address || ""}
              />
            </FormControl>
          </Box>
        )}
  
        {/* Part & Brand Selection */}
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            gap: 2,
            '& .MuiFormControl-root': { flex: 1 },
          }}
        >
          <FormControl  error={!!formErrors.selectedPart}>
            <InputLabel htmlFor="part">Part & Brand</InputLabel>
            <Select
              value={selectedPart}
              onChange={handlePartChange}
              input={<OutlinedInput label="Part & Brand" />}
              error={!!formErrors.selectedPart}
            >
              {existingParts
                .filter(part => !part.deleted == true) // Filter out deleted parts
                .map((part) => (
                <MenuItem key={part._id} value={`${part.partName}|${part.brandName}`}>
                  {`${part.partName} - ${part.brandName} (Available: ${part.qtyLeft})`}
                </MenuItem>
              ))}
              <MenuItem value="new">Add New Part</MenuItem>
            </Select>
          </FormControl>
  
          {/* Conditionally show New Part and Brand fields */}
          {selectedPart === 'new' && (
            <>
              <FormControl error={!!formErrors.selectedPart}>
                <InputLabel htmlFor="newPartName">New Part Name</InputLabel>
                <OutlinedInput
                  id="newPartName"
                  label="New Part Name"
                  value={newPartName}
                  onChange={(e) => setNewPartName(e.target.value)}
                />
              </FormControl>
              <FormControl error={!!formErrors.selectedPart}>
                <InputLabel htmlFor="newBrandName">New Brand Name</InputLabel>
                <OutlinedInput
                  id="newBrandName"
                  label="New Brand Name"
                  value={newBrandName}
                  onChange={(e) => setNewBrandName(e.target.value)}
                />
              </FormControl>
            </>
          )}
        </Box>
  
        {/* Additional details for Description and Quantity */}
        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', sm: 'row' },
            gap: 2,
            '& .MuiFormControl-root': { flex: 1 },
          }}
        >
          <FormControl>
            <InputLabel htmlFor="description">Description</InputLabel>
            <OutlinedInput
              id="description"
              label="Description"
              {...register('description', { required: true })}
              defaultValue={initialValues?.description || ""}
            />
          </FormControl>
  
          <FormControl>
            <InputLabel htmlFor="quantityBought">Quantity Bought</InputLabel>
            <OutlinedInput
              id="quantityBought"
              type="number"
              label="Quantity Bought"
              {...register('quantityBought', { required: true })}
              defaultValue={initialValues?.quantityBought || 0}
            />
          </FormControl>
        </Box>
  
        {/* Amount Input */}
        <TextField
          label="Amount"
          variant="outlined"
          type="number"
          required
          value={amount}
          onChange={handleAmountChange}
          inputProps={{ step: "0.01", required: true  }}
          error={!!formErrors.supplierName}
        />
  
        {/* Non-VAT Checkbox */}
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
  
        {/* Net of VAT and Input VAT */}
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
  
        {/* Action Buttons */}
        <Box display="flex" justifyContent="center" gap={2} mt={3}>
          <CustomButton
            type="submit" // This ensures the button submits the form
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
            handleClick={() => navigate('/procurements')}
          />
        </Box>
      </form>
    </Paper>
  );
};
  
  export default ProcurementForm;
  