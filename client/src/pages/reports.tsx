import React, { useState } from 'react';
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
import { Stack } from '@pankod/refine-mui';

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

const ReportsPage: React.FC = () => {
    const [report, setReport] = useState<ReportData | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    
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

    const generateReport = async () => {
        setLoading(true);
        setError(null);
        try {
            // Prepare query parameters
            const params: { month?: string; year?: string } = {};
            
            // Add month if not ALL and convert to number
            if (selectedMonth !== 'ALL') {
                params.month = (months.indexOf(selectedMonth)).toString();
            }
            
            // Add year if not ALL
            if (selectedYear !== 'ALL') {
                params.year = selectedYear;
            }

            // Make API call with optional parameters
            const { data } = await axios.get<ReportData>('http://localhost:8080/api/v1/reports/generate', { params });
            setReport(data);
        } catch (err) {
            setError('Failed to generate report');
            console.error(err);
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
        <Container maxWidth="md" sx={{ mt: 4 }}>
            <Typography variant="h4" component="h1" gutterBottom>
                Reports Dashboard
            </Typography>

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
                <Paper sx={{ p: 3 }}>
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
                    
                    <Stack>
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
    );
};

export default ReportsPage;