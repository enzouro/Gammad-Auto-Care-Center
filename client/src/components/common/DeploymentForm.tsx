//client\src\components\common\DeploymentForm.tsx
import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  FormControl, 
  InputLabel,
  Paper,
  OutlinedInput,
  CircularProgress,
  Select,
  MenuItem,
  SelectChangeEvent,
  TextField,
  FormHelperText,
  FormControlLabel,
  Checkbox,
  Button} from '@pankod/refine-mui';
import { FormPropsDeployment } from 'interfaces/common';
import { useNavigate } from '@pankod/refine-react-router-v6';
import { Add, Close, Delete, Publish } from '@mui/icons-material';
import CustomButton from './CustomButton';
import CustomIconButton from './CustomIconButton';
import { customRandom } from 'nanoid';
import useNextSequence from 'hooks/useNextSequence';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

interface PartEntry {
  partId: string;
  quantityUsed: number;
}

const nanoid = customRandom('1234567890QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm', 7, size => {
  return (new Uint8Array(size)).map(() => 256 * Math.random()); 
});

const getTodayDate = () => {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, '0');
  const day = String(today.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const DeploymentForm = ({ type, register, handleSubmit, formLoading, onFinishHandler, existingParts, initialValues }: FormPropsDeployment) => {
  const [partsEntries, setPartsEntries] = useState<PartEntry[]>([{ partId: '', quantityUsed: 0 }]);
  const [deploymentStatus, setDeploymentStatus] = useState<boolean>(false);
  const [releaseStatus, setReleaseStatus] = useState<boolean>(false);
  const [availableQuantities, setAvailableQuantities] = useState<{[key: string]: number}>({});
  const [trackCode, setTrackCode] = useState<string>(initialValues?.trackCode || nanoid);
  const [repairStatus, setRepairStatus] = useState<string>(initialValues?.repairStatus || 'Pending');

  const navigate = useNavigate();
  const isError = false;

  // Use the custom hook for sequence logic
  const { currentSeq, isLoading: sequenceLoading } = useNextSequence({
    resource: "deployments",
    type: type as "Create" | "Edit", // Assert type explicitly
    initialValues,
  });

  useEffect(() => {
    if (initialValues) {
      // Initialize parts entries
      if (initialValues.parts && Array.isArray(initialValues.parts)) {
        const initialParts = initialValues.parts.map((p: any) => ({
          partId: `${p.part.partName}|${p.part.brandName}`,
          quantityUsed: parseInt(p.quantityUsed) || 0
        }));
        setPartsEntries(initialParts.length > 0 ? initialParts : [{ partId: '', quantityUsed: 0 }]);
        
        // Initialize available quantities
        const quantities: {[key: string]: number} = {};
        initialValues.parts.forEach((p: any) => {
          if (p.part) {
            const partKey = `${p.part.partName}|${p.part.brandName}`;
            quantities[partKey] = (p.part.qtyLeft || 0) + (parseInt(p.quantityUsed) || 0);
          }
        });
        setAvailableQuantities(quantities);
      }
      // Initialize repairStatus
      setDeploymentStatus(!!initialValues.deploymentStatus);
      setReleaseStatus(!!initialValues.releaseStatus);
      setTrackCode(initialValues.trackCode || '');
      setRepairStatus(initialValues.repairStatus || 'Pending');

    }
  }, [initialValues]);

  const handlePartChange = (index: number) => (event: SelectChangeEvent<string>) => {
    const value = event.target.value;
    const newPartsEntries = [...partsEntries];
    newPartsEntries[index].partId = value;
    newPartsEntries[index].quantityUsed = 0;
    setPartsEntries(newPartsEntries);

    const part = existingParts.find(p => `${p.partName}|${p.brandName}` === value);
    if (part) {
      setAvailableQuantities(prev => ({
        ...prev,
        [value]: part.qtyLeft
      }));
    }
  };
  const handleQuantityChange = (index: number) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const newPartsEntries = [...partsEntries];
    newPartsEntries[index].quantityUsed = parseInt(event.target.value) || 0;
    setPartsEntries(newPartsEntries);
  };
  const addPartEntry = () => {
    setPartsEntries([...partsEntries, { partId: '', quantityUsed: 0 }]);
  };
  const removePartEntry = (index: number) => {
    if (partsEntries.length > 1) {
      const newPartsEntries = partsEntries.filter((_, i) => i !== index);
      setPartsEntries(newPartsEntries);
    } else if (partsEntries.length <= 1) {
      setPartsEntries([{ partId: '', quantityUsed: 0 }]);
    }
  };

  const onSubmit = (data: Record<string, any>) => {
    const updatedData: Record<string, any> = {
      ...data,
      seq: currentSeq,
    };

    // Filter out empty entries and format parts data
    const validParts = partsEntries.filter((entry) =>
      entry.partId && 
      entry.quantityUsed > 0 && 
      entry.quantityUsed <= (availableQuantities[entry.partId] || 0)
    );
    // Allow submission even if no parts are selected
    updatedData.parts = validParts.map((entry) => ({
      part: entry.partId,
      quantityUsed: entry.quantityUsed,
    }));

    updatedData.deploymentStatus = deploymentStatus;
    updatedData.releaseStatus = releaseStatus;
    updatedData.repairStatus = repairStatus;
    updatedData.deploymentDate = deploymentStatus ? data.deploymentDate : null;
    updatedData.releaseDate = releaseStatus ? data.releaseDate : null;
    updatedData.repairedDate = repairStatus ? data.repairedDate : null;
  

    onFinishHandler(updatedData);
  };

  if (formLoading || sequenceLoading || currentSeq === null) {
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
        {type} a Deployment
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
          flexDirection: { xs: 'column', sm: 'row' }, 
          gap: 2,
          '& .MuiFormControl-root': { flex: 1 }
        }}>
          <FormControl sx={{ flex: 1 }}>
            <FormHelperText sx={{ fontWeight: 500, margin: '10px 0', fontSize: 16 }}>Client Name</FormHelperText>
            <TextField
              required
              variant="outlined"
              color="info"
              {...register('clientName', { required: true })}
              defaultValue={initialValues?.clientName || ""}
            />
          </FormControl>

          <FormControl sx={{ flex: 1 }}>
            <FormHelperText sx={{ fontWeight: 500, margin: '10px 0', fontSize: 16 }}>Vehicle Model</FormHelperText>
            <TextField
              required
              variant="outlined"
              color="info"
              {...register('vehicleModel', { required: true })}
              defaultValue={initialValues?.vehicleModel || ""}
            />
          </FormControl>

            <FormControl sx={{ flex: 1 }}>
              <FormHelperText sx={{ fontWeight: 500, margin: '10px 0', fontSize: 16 }}>Arrival Date</FormHelperText>
              <TextField
                required
                type="date"
                variant="outlined"
                color="info"
                {...register('arrivalDate', { required: true })}
                defaultValue={initialValues?.arrivalDate || getTodayDate()}
              />
            </FormControl>
          </Box>

            {/* Parts Section */}
      <Typography variant="h6" sx={{ mt: 3, mb: 2 }}>Parts Selection</Typography>
      
      {partsEntries.map((entry, index) => (
        <Box key={index} sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', sm: 'row' }, 
          gap: 2,
          mb: 2,
          alignItems: 'center'
        }}>
          <FormControl sx={{ flex: 2 }}>
          <InputLabel htmlFor="part">Part & Brand</InputLabel>
            <Select
              value={entry.partId}
              onChange={handlePartChange(index)}
              input={<OutlinedInput label="Part & Brand" />}
            >
              {existingParts
                .filter(part => !part.deleted == true) // Filter out deleted parts
                .map((part) => (
                <MenuItem key={part._id} value={`${part.partName}|${part.brandName}`}>
                  {`${part.partName} - ${part.brandName} (Available: ${part.qtyLeft})`}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl sx={{ flex: 1 }}>
            <TextField
              type="number"
              value={entry.quantityUsed}
              onChange={handleQuantityChange(index)}
              inputProps={{ 
                min: 0, 
                max: availableQuantities[entry.partId] || 0 
              }}
              label={`Quantity (Max: ${availableQuantities[entry.partId] || 0})`}
            />
          </FormControl>
          <CustomIconButton
            title="Remove"
            icon={<Delete />}
            backgroundColor="error.light"
            color="error.dark"
            handleClick={() => removePartEntry(index)}
          />
        </Box>
      ))}
        <Button
          onClick={addPartEntry}
          startIcon={<Add />}
          sx={{ mt: 2, mb: 4 }}
          variant="outlined"
        >
          Add Another Part
        </Button>

          <Box sx={{ 
            display: 'flex', 
            flexDirection: { xs: 'column', sm: 'row' }, 
            gap: 2,
            '& .MuiFormControl-root': { flex: 1 }
          }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={deploymentStatus}
                onChange={(e) => setDeploymentStatus(e.target.checked)}
              />
            }
            label="Deployed"
          />

          {deploymentStatus && (
            <FormControl>
              <FormHelperText sx={{ fontWeight: 500, fontSize: 16 }}>Deployment Date</FormHelperText>
              <TextField
                type="date"
                {...register('deploymentDate', { required: deploymentStatus })}
                defaultValue={initialValues?.deploymentDate?.split('T')[0] || getTodayDate()}
              />
            </FormControl>
          )}

          <FormControlLabel
            control={
              <Checkbox
                checked={releaseStatus}
                onChange={(e) => setReleaseStatus(e.target.checked)}
                disabled={!deploymentStatus}
              />
            }
            label="Released"
          />

          {releaseStatus && (
            <FormControl>
              <FormHelperText sx={{ fontWeight: 500, fontSize: 16 }}>Release Date</FormHelperText>
              <TextField
                type="date"
                {...register('releaseDate', { required: releaseStatus })}
                defaultValue={initialValues?.releaseDate?.split('T')[0] || getTodayDate()}
              />
            </FormControl>
          )}
        </Box>

        <Box sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', sm: 'row' }, 
        gap: 2,
        '& .MuiFormControl-root': { flex: 1 }
      }}>

        <FormControl>
          <FormHelperText sx={{ fontWeight: 500, fontSize: 16 }}>Status</FormHelperText>
          <Select
            value={repairStatus}
            {...register('repairStatus')}
            onChange={(e) => setRepairStatus(e.target.value as string)}
          >
            <MenuItem value="Pending">Pending</MenuItem>
            <MenuItem value="In Progress">In Progress</MenuItem>
            <MenuItem value="Repaired">Repaired</MenuItem>
            <MenuItem value="Cancelled">Cancelled</MenuItem>
          </Select>
        </FormControl>

        {repairStatus === 'Repaired' && (
          <FormControl>
            <FormHelperText sx={{ fontWeight: 500, fontSize: 16 }}>Repaired Date</FormHelperText>
            <TextField
              type="date"
              {...register('repairedDate')}
              defaultValue={initialValues?.repairedDate?.split('T')[0] || getTodayDate()}
            />
          </FormControl>
        )}

        <FormControl>
          <FormHelperText sx={{ fontWeight: 500, fontSize: 16 }}>Track Code</FormHelperText>
          <TextField
            required
            variant="outlined"
            color="info"
            {...register('trackCode', { required: true })}
            value={trackCode}
            onChange={(e) => setTrackCode(e.target.value)}
          />
        </FormControl>
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
            handleClick={() => navigate('/deployments')}
          />
        </Box>
      </form>
    </Paper>
  );
};

export default DeploymentForm;