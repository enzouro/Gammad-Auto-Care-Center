//client\src\pages\client-portal.tsx
import React, { useState } from 'react';
import { 
  TextField,
  Button,
  Box,
  Alert,
  CircularProgress,
  Paper,
  Grid,
  Typography,
  Chip,
  Card,
  CardContent,
  CardHeader,
  Divider,
  Stack,
  styled,
  Theme
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import TimelineIcon from '@mui/icons-material/Timeline';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import LocalShippingIcon from '@mui/icons-material/LocalShipping';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import Header from 'components/client-portal/Header';
import Footer from 'components/client-portal/Footer';

// Interfaces
interface Part {
  partName: string;
  brandName: string;
  quantityUsed: number;
}

// First, let's update the SearchResult interface to include estimated release time
interface SearchResult {
  seq: number;
  date: string;
  clientName: string;
  vehicleModel: string;
  arrivalDate: string;
  parts: Part[];
  releaseStatus: boolean;
  releaseDate: string | null;
  repairStatus: string;
  repairedDate: string | null;
  trackCode: string;
  estimatedReleaseDate?: string; // New field
}

interface StatusChipProps {
  status: string;
}

const StatusChip = styled(Chip)<StatusChipProps>(({ theme, status }) => {
  // Define color pairs for each status (background: lighter shade, text: darker shade)
  const statusColors = {
    'Pending': {
      background: '#FFF3E0', // Light Orange
      text: '#E65100'        // Dark Orange
    },
    'In Progress': {
      background: '#E3F2FD', // Light Blue
      text: '#1565C0'        // Dark Blue
    },
    'Repaired': {
      background: '#E8F5E9', // Light Green
      text: '#2E7D32'        // Dark Green
    },
    'Cancelled': {
      background: '#FFEBEE', // Light Red
      text: '#C62828'        // Dark Red
    },
    'Released': {
      background: '#E8F5E9', // Light Green
      text: '#2E7D32'        // Dark Green
    }
  };

  const currentStatus = statusColors[status as keyof typeof statusColors] || statusColors.Pending;

  return {
    backgroundColor: currentStatus.background,
    color: currentStatus.text,
    '& .MuiChip-label': {
      fontWeight: 600,
    },
  };
});

interface TimelinePointProps {
  dotColor?: string;
}

const TimelinePoint = styled(Box)<TimelinePointProps>(({ theme, dotColor }) => ({
  width: 12,
  height: 12,
  borderRadius: '50%',
  backgroundColor: dotColor || theme.palette.primary.main,
  marginRight: theme.spacing(1),
}));

// Create a new component for displaying the estimated release date
const EstimatedRelease: React.FC<{ 
  arrivalDate: string; 
  estimatedReleaseDate: string;
}> = ({ arrivalDate, estimatedReleaseDate }) => {
  return (
    <Box sx={{ mt: 2 }}>
      <Alert 
        severity="info" 
        icon={<TimelineIcon />}
        sx={{
          '& .MuiAlert-message': {
            width: '100%'
          }
        }}
      >
        <Box>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            Estimated Timeline
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              label={`Arrival: ${new Date(arrivalDate).toLocaleDateString()}`}
              size="small"
              color="primary"
              variant="outlined"
            />
            <ArrowForwardIcon sx={{ color: 'text.secondary' }} />
            <Chip
              label={`Estimated Release: ${new Date(estimatedReleaseDate).toLocaleDateString()}`}
              size="small"
              color="primary"
              variant="outlined"
            />
          </Box>
        </Box>
      </Alert>
    </Box>
  );
};

// Update the SearchResult component to include the estimated release date
const SearchResult = ({ searchResult }: { searchResult: SearchResult | null }) => {
  if (!searchResult) return null;

  const getStatusText = () => {
    if (searchResult.repairStatus === 'Cancelled') return 'Cancelled';
    if (searchResult.releaseStatus) return 'Released';
    if (searchResult.repairStatus === 'Repaired') return 'Repaired';
    if (searchResult.repairStatus === 'In Progress') return 'In Progress';
    return 'Pending';
  };

  return (
    <Card elevation={3}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h5">Service Details</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="subtitle1" color="text.secondary">
                Vehicle Status:
              </Typography>
              <StatusChip
                label={getStatusText()}
                status={getStatusText()}
                theme={undefined}
              />
            </Box>
          </Box>
        }
        subheader={
          <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
            <Chip
              label={`Track Code: ${searchResult.trackCode}`}
              variant="outlined"
              size="small"
            />
            <Chip
              label={`Sequence: ${searchResult.seq}`}
              variant="outlined"
              size="small"
            />
          </Box>
        }
      />

      <Divider />

      <CardContent>
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Client Information
            </Typography>
            <Typography variant="h6">{searchResult.clientName}</Typography>
            <Typography variant="body2" color="text.secondary">
              Vehicle: {searchResult.vehicleModel}
            </Typography>
          </Grid>

          <Grid item xs={12} md={6}>
            <Stack spacing={2}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <TimelinePoint dotColor="#1976d2" />
                <Typography variant="body2">
                  <strong>Arrival Date:</strong> {searchResult.arrivalDate}
                </Typography>
              </Box>
              {searchResult.repairedDate && (
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TimelinePoint dotColor="#2e7d32" />
                  <Typography variant="body2">
                    <strong>Repair Completed:</strong> {searchResult.repairedDate}
                  </Typography>
                </Box>
              )}
              {searchResult.releaseDate && (
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <TimelinePoint dotColor="#9c27b0" />
                  <Typography variant="body2">
                    <strong>Release Date:</strong> {searchResult.releaseDate}
                  </Typography>
                </Box>
              )}
            </Stack>
          </Grid>

          {searchResult.estimatedReleaseDate && !searchResult.releaseStatus && (
            <Grid item xs={12}>
              <EstimatedRelease
                arrivalDate={searchResult.arrivalDate}
                estimatedReleaseDate={searchResult.estimatedReleaseDate}
              />
            </Grid>
          )}

          <Grid item xs={12}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Parts Used
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {searchResult.parts.map((part, index) => (
                <Chip
                  key={index}
                  label={`${part.partName} - ${part.brandName} (${part.quantityUsed})`}
                  variant="outlined"
                  size="small"
                />
              ))}
            </Box>
          </Grid>
        </Grid>
      </CardContent>

      <Divider />

      {searchResult.repairStatus === 'Repaired' && !searchResult.releaseStatus && (
        <Box sx={{ px: 2, pb: 2 }}>
          <Alert severity="info" icon={<LocalShippingIcon />}>
            Your vehicle is ready for release. Please proceed to our service center for vehicle pickup.
          </Alert>
        </Box>
      )}
    </Card>
  );
};

// Update the ClientPortal component to fetch turnaround estimates
const ClientPortal: React.FC = () => {
  const [trackCode, setTrackCode] = useState('');
  const [searchResult, setSearchResult] = useState<SearchResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

// Update the handleSearch function
const handleSearch = async () => {
  try {
    setIsLoading(true);
    setError(null);
    setSearchResult(null);

    const response = await fetch(
      `http://localhost:8080/api/v1/clientPortal/search?trackCode=${encodeURIComponent(trackCode)}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    
    if (data.success) {
      try {
        const turnaroundResponse = await fetch(
          'http://localhost:8080/api/forecasting/turnaround',
          {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          }
        );

        if (turnaroundResponse.ok) {
          const turnaroundData = await turnaroundResponse.json();
          // Add null check and access avgTurnaroundTime safely
          const avgTurnaroundTime = turnaroundData?.historical?.efficiencyMetrics?.avgTurnaroundTime || 72; // Default to 72 hours if not available
          
          const arrivalDate = new Date(data.data.arrivalDate);
          const estimatedReleaseDate = new Date(arrivalDate.getTime() + (avgTurnaroundTime * 60 * 60 * 1000));
          
          setSearchResult({
            ...data.data,
            estimatedReleaseDate: estimatedReleaseDate.toISOString()
          });
        } else {
          // If turnaround data fetch fails, still show vehicle data
          setSearchResult(data.data);
        }
      } catch (turnaroundError) {
        // If turnaround calculation fails, still show vehicle data
        console.error('Turnaround calculation error:', turnaroundError);
        setSearchResult(data.data);
      }
    } else {
      setError(data.message || 'Error searching for vehicle');
    }
  } catch (error: any) {
    console.error('Search error:', error);
    setError(error.message || 'An unexpected error occurred');
  } finally {
    setIsLoading(false);
  }
};

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && trackCode) {
      handleSearch();
    }
  };

  return (
    <Box 
      sx={{ 
        display: 'flex',
        flexDirection: 'column',
        minHeight: '100vh',
      }}
    >
      <Header />
      
      <Box 
        component="main"
        sx={{ 
          flexGrow: 1,
          p: 4,
          maxWidth: '1200px',
          width: '100%',
          mx: 'auto',
        }}
      >
        <Typography variant="h3" align="center" gutterBottom>
          Vehicle Service Tracker
        </Typography>
        <Typography variant="subtitle1" color="text.secondary" align="center" sx={{ mb: 4 }}>
          Enter your tracking code to check your vehicle's service status
        </Typography>

        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'center', 
          gap: 2, 
          mb: 4,
          maxWidth: '500px',
          mx: 'auto',
        }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Enter tracking code"
            value={trackCode}
            onChange={(e) => setTrackCode(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <Button
            variant="contained"
            onClick={handleSearch}
            disabled={isLoading || !trackCode}
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
          >
            {isLoading ? "Searching..." : "Track"}
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 4, maxWidth: '500px', mx: 'auto' }}>
            {error}
          </Alert>
        )}

        {searchResult && <SearchResult searchResult={searchResult} />}
      </Box>
      
      <Footer />
    </Box>
  );
};

export default ClientPortal;