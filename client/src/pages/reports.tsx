// client\src\pages\reports.tsx
import React, { useContext, useState } from 'react';
import axios from 'axios';
import { 
  Container, 
  Paper, 
  Typography, 
  Button, 
  Grid, 
  Box, 
  CircularProgress, 
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { Dialog, DialogContent, DialogTitle, FilledInput, IconButton, List, ListItem, ListItemText, Stack, Tooltip } from '@pankod/refine-mui';
import useDynamicHeight from 'hooks/useDynamicHeight';
import { FileDownload, FileUpload, ImportExport, ImportExportOutlined } from '@mui/icons-material';
import { ColorModeContext } from 'contexts';

// Update interfaces to match the actual backend response
interface SummarySectionData {
  tableHeader: string[];
  tableData: Array<{[key: string]: number}>;
}

interface ReportData {
  salesSummary: SummarySectionData;
  expensesSummary: SummarySectionData;
  procurementSummary: SummarySectionData;
  deploymentSummary: SummarySectionData;
  filterCriteria?: {
    month?: string;
    year?: string;
  };
}

// Enum for data types
enum DataType {
    Procurement = 'Procurement',
    Deployments = 'Deployments',
    Sales = 'Sales',
    Expenses = 'Expenses'
}

const ReportsPage: React.FC = () => {

    const containerHeight = useDynamicHeight();
    const { mode } = useContext(ColorModeContext);
    // State for filters
    const [report, setReport] = useState<ReportData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
    // New states for import/export dialog
    const [importExportDialogOpen, setImportExportDialogOpen] = useState(false);
    const [importExportMode, setImportExportMode] = useState<'import' | 'export'>('import');
    const [selectedDataType, setSelectedDataType] = useState<DataType | null>(null);

    // State for filters
    const [selectedMonth, setSelectedMonth] = useState<string>('ALL');
    const [selectedYear, setSelectedYear] = useState<string>('ALL');

    // Generate current and past years (last 5 years)
    const currentYear = new Date().getFullYear();
    const years = ['ALL', ...Array.from({ length: 5 }, (_, i) => (currentYear - i).toString())];
    
    // Months list
    const months = [
        'ALL', 
        'January', 'February', 'March', 
        'April', 'May', 'June', 
        'July', 'August', 'September', 
        'October', 'November', 'December'
    ];

      // New method to handle file upload
      const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file || !selectedDataType) return;

        const formData = new FormData();
        formData.append('file', file);
        
        try {
            setLoading(true);
            const response = await axios.post(`https://gammad-auto-care-center.onrender.com/api/v1/data/import/${selectedDataType.toLowerCase()}`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
            
            // Handle successful import
            alert(`${selectedDataType} imported successfully`);
            setImportExportDialogOpen(false);
        } catch (err) {
            setError('Failed to import data');
        } finally {
            setLoading(false);
        }
    };

        // New method to handle data export
        const handleDataExport = async (dataType: DataType) => {
            try {
                setLoading(true);
                const response = await axios.get(`https://gammad-auto-care-center.onrender.com/api/v1/data/export/${dataType.toLowerCase()}`, {
                    responseType: 'blob'
                });
                
                // Create a link to download the file
                const url = window.URL.createObjectURL(new Blob([response.data]));
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', `${dataType.toLowerCase()}_export.csv`);
                document.body.appendChild(link);
                link.click();
                
                setImportExportDialogOpen(false);
            } catch (err) {
                setError('Failed to export data');
            } finally {
                setLoading(false);
            }
        };

        const openImportExportDialog = (mode: 'import' | 'export') => {
            setImportExportMode(mode);
            setImportExportDialogOpen(true);
        };
    
        // Render import/export dialog
        const renderImportExportDialog = () => {
            const dataTypes = Object.values(DataType);
    
            return (
                <Dialog 
                    open={importExportDialogOpen} 
                    onClose={() => setImportExportDialogOpen(false)}
                    fullWidth
                    maxWidth="xs"
                >
                    <DialogTitle>
                        {importExportMode === 'import' ? 'Import Data' : 'Export Data'}
                    </DialogTitle>
                    <DialogContent>
                        <List>
                            {dataTypes.map((type) => (
                                <ListItem 
                                    key={type} 
                                    button 
                                    onClick={() => {
                                        setSelectedDataType(type);
                                        if (importExportMode === 'export') {
                                            handleDataExport(type);
                                        }
                                    }}
                                >
                                    {importExportMode === 'import' ? (
                                        <>
                                            <ListItemText primary={`Import ${type}`} />
                                            <input
                                                type="file"
                                                accept=".csv,.xlsx"
                                                style={{ display: 'none' }}
                                                id={`import-${type}`}
                                                onChange={handleFileUpload}
                                            />
                                            <label htmlFor={`import-${type}`}>
                                                <Button 
                                                    component="span" 
                                                    variant="contained" 
                                                    color="primary"
                                                    onClick={() => {}}
                                                >
                                                    Choose File
                                                </Button>
                                            </label>
                                        </>
                                    ) : (
                                        <ListItemText primary={`Export ${type}`} />
                                    )}
                                </ListItem>
                            ))}
                        </List>
                    </DialogContent>
                </Dialog>
            );
        };
    


    const generateReport = async () => {
        setLoading(true);
        setError(null);
        try {
            const params: { month?: string; year?: string } = {};
        
            if (selectedMonth !== 'ALL') {
                // Convert month name to its numerical index
                params.month = (months.indexOf(selectedMonth) + 1).toString();
            }
            
            if (selectedYear !== 'ALL') {
                params.year = selectedYear;
            }
    
            const { data } = await axios.get<ReportData>('https://gammad-auto-care-center.onrender.com/api/v1/reports/generate', { params });
            setReport(data);
        } catch (err) {
            setError('Failed to generate report');
            
        } finally {
            setLoading(false);
        }
    };

    const handleMonthChange = (event: SelectChangeEvent) => {
        setSelectedMonth(event.target.value);
    };

    const handleYearChange = (event: SelectChangeEvent) => {
        setSelectedYear(event.target.value);
    };

    const renderSummarySection = (title: string, summarySectionData: SummarySectionData) => {
        // If no data, return null
        if (!summarySectionData.tableData || summarySectionData.tableData.length === 0) {
            return null;
        }

        // Get the first (and likely only) data object
        const data = summarySectionData.tableData[0];

        return (
            <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                    {title}
                </Typography>
                <TableContainer>
                    <Table>
                        <TableHead>
                            <TableRow>
                                {Object.keys(data).map((key) => (
                                    <TableCell key={key}>
                                        {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                                    </TableCell>
                                ))}
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            <TableRow>
                                {Object.entries(data).map(([key, value]) => (
                                    <TableCell key={key}>
                                        {typeof value === 'number' ? value.toFixed(2) : value}
                                    </TableCell>
                                ))}
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableContainer>
            </Paper>
        );
    };

    return (
        <Paper
            elevation={3} 
            sx={{ 
                height: '100%', // Changed from specific pixel heights
                minHeight: containerHeight, // Ensure minimum height
                display: 'flex',
                flexDirection: 'column',

            }}
        >
            <Container 
                maxWidth="md" 
                sx={{ 
                    mt: 4, 
                    flex: 1, // Allow container to grow
                    display: 'flex',
                    flexDirection: 'column',
                    overflow: 'hidden',
                }}
            >
                <Box 
                justifyContent={'space-between'}
                flex='row'
                display='flex'
                >
                    <Typography variant="h4" component="h1" gutterBottom>
                        Reports Dashboard
                    </Typography>
                    <Box display='flex' flexDirection='row'>
                        <Tooltip title="Import Data" arrow>
                            <IconButton onClick={() => openImportExportDialog('import')}>
                                <FileUpload />
                            </IconButton>
                        </Tooltip>
                        <Tooltip title="Export Data" arrow>
                            <IconButton onClick={() => openImportExportDialog('export')}>
                                <FileDownload />
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>

                <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                            <InputLabel>Month</InputLabel>
                            <Select
                                value={selectedMonth}
                                label="Month"
                                onChange={handleMonthChange}
                            >
                                {months.map((month) => (
                                    <MenuItem key={month} value={month}>
                                        {month}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>
                    <Grid item xs={12} sm={6}>
                        <FormControl fullWidth>
                            <InputLabel>Year</InputLabel>
                            <Select
                                value={selectedYear}
                                label="Year"
                                onChange={handleYearChange}
                            >
                                {years.map((year) => (
                                    <MenuItem key={year} value={year}>
                                        {year}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    </Grid>
                </Grid>

                <Box sx={{ mb: 3 }}>
                    <Button 
                        variant="contained" 
                        color="primary" 
                        onClick={generateReport} 
                        disabled={loading}
                        startIcon={loading ? <CircularProgress size={20} /> : null}
                    >
                        {loading ? 'Generating...' : 'Generate Report'}
                    </Button>
                </Box>

                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}

                {report && (
                    <Paper  
                        elevation={3} 
                        sx={{ 
                            p: 2,   
                            flex: 1, // Allow this to grow and fill available space
                            display: 'flex',
                            flexDirection: 'column',
                            m: 2,
                            overflow: 'auto',// Ensure scrolling if content is too long
                            '& .MuiDataGrid-main': {
                                overflow: 'hidden',
                                '& ::-webkit-scrollbar': {
                                  width: '10px',
                                  height: '10px',
                                },
                                '& ::-webkit-scrollbar-track': {
                                  background: mode === 'light' ? '#f1f1f1' : '#2c2c2c',
                                  borderRadius: '10px',
                                },
                                '& ::-webkit-scrollbar-thumb': {
                                  background: mode === 'light' 
                                    ? 'linear-gradient(45deg, #e0e0e0, #a0a0a0)' 
                                    : 'linear-gradient(45deg, #4a4a4a, #2c2c2c)',
                                  borderRadius: '10px',
                                  transition: 'background 0.3s ease',
                                },
                                '& ::-webkit-scrollbar-thumb:hover': {
                                  background: mode === 'light'
                                    ? 'linear-gradient(45deg, #c0c0c0, #808080)'
                                    : 'linear-gradient(45deg, #5a5a5a, #3c3c3c)',
                                },
                              },
                        }}
                    >
                        <Typography variant='h3' sx={{ mb: 2 }}> 
                            Monthly Report
                        </Typography>

                        <Typography variant='h6'> 
                            Company Name: Gammad Auto Care Corporations
                        </Typography>
                        <Typography variant='h6' sx={{ mb: 2 }}> 
                            Report Period: {selectedMonth === 'ALL' && selectedYear === 'ALL' 
                                ? 'All Records' 
                                : `${selectedMonth === 'ALL' ? 'All Months' : selectedMonth} ${selectedYear === 'ALL' ? 'All Years' : selectedYear}`}
                        </Typography>
                        
                        <Stack sx={{ flex: 1, overflow: 'auto' }}>
                            <Box>
                                {renderSummarySection('Sales Summary', report.salesSummary)}
                                {renderSummarySection('Expenses Summary', report.expensesSummary)}
                                {renderSummarySection('Procurement Summary', report.procurementSummary)}
                                {renderSummarySection('Deployment Summary', report.deploymentSummary)}
                            </Box>
                        </Stack>
                    </Paper>
                )}
            </Container>
            {/* Import/Export Dialog */}
            {renderImportExportDialog()}
        </Paper>
    );
};

export default ReportsPage;