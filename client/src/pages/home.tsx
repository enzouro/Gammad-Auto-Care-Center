import { useState, useMemo, useEffect } from 'react';
import { useList } from '@pankod/refine-core';
import { 
  Box, 
  Paper, 
  Typography, 
  Stack,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@pankod/refine-mui';
import ApexChart from '../components/common/ApexChart';
import useDynamicHeight from 'hooks/useDynamicHeight';
import LoadingDialog from 'components/common/LoadingDialog';
import ErrorDialog from 'components/common/ErrorDialog';

// Existing interfaces
interface SaleItem {
  date: string;
  amount: number;
  deleted?: boolean;
}

interface ExpenseItem {
  date: string;
  amount: number;
  deleted?: boolean;
}

interface ProcurementItem {
  date: string;
  amount: number;
  deleted?: boolean;
}

interface DeploymentItem {
  releaseStatus: boolean;
  deleted?: boolean;
}

interface PartItem {
  partName: string;
  qtyLeft: number;
  deleted?: boolean;
}

interface ChartData {
  name: string;
  data: number[];
}

const Home = () => {
  const containerHeight = useDynamicHeight();
  
  // Generate years and months arrays
  const currentYear = new Date().getFullYear();
  const currentMonth = new Date().getMonth(); // 0-11
  const years = Array.from({ length: 5 }, (_, i) => (currentYear - i).toString());
  const months = [
    'January', 'February', 'March',
    'April', 'May', 'June',
    'July', 'August', 'September',
    'October', 'November', 'December'
  ];

  // Initialize with current year and month
  const [selectedYear, setSelectedYear] = useState(currentYear.toString());
  const [selectedMonth, setSelectedMonth] = useState(months[currentMonth]);

  // Existing data fetching
  const { 
    data: salesData, 
    isLoading: salesLoading,
    isError: salesError 
  } = useList<SaleItem>({
    resource: 'sales',
    config: {
      pagination: { pageSize: 1000 }
    }
  });

  const { 
    data: expensesData, 
    isLoading: expensesLoading,
    isError: expensesError 
  } = useList<ExpenseItem>({
    resource: 'expenses',
    config: {
      pagination: { pageSize: 1000 }
    }
  });

  const { 
    data: procurementData, 
    isLoading: procurementLoading,
    isError: procurementError 
  } = useList<ProcurementItem>({
    resource: 'procurements',
    config: {
      pagination: { pageSize: 1000 }
    }
  });

  const { 
    data: deploymentsData, 
    isLoading: deploymentsLoading,
    isError: deploymentsError 
  } = useList<DeploymentItem>({
    resource: 'deployments',
    config: {
      pagination: { pageSize: 1000 }
    }
  });

  const { 
    data: partsData, 
    isLoading: partsLoading,
    isError: partsError 
  } = useList<PartItem>({
    resource: 'parts',
    config: {
      pagination: { pageSize: 1000 }
    }
  });

  // Initialize state
  const [salesChartData, setSalesChartData] = useState<ChartData>({ name: '', data: [] });
  const [salesCategories, setSalesCategories] = useState<string[]>([]);
  const [expensesChartData, setExpensesChartData] = useState<ChartData>({ name: '', data: [] });
  const [expensesCategories, setExpensesCategories] = useState<string[]>([]);
  const [procurementChartData, setProcurementChartData] = useState<ChartData>({ name: '', data: [] });
  const [procurementCategories, setProcurementCategories] = useState<string[]>([]);
  const [deploymentsStatusData, setDeploymentsStatusData] = useState<number[]>([]);
  const [partsQuantityData, setPartsQuantityData] = useState<number[]>([]);
  const [partsLabels, setPartsLabels] = useState<string[]>([]);

  // Helper function to filter data by year and month
  const filterDataByDate = <T extends { date: string, deleted?: boolean }>(
    data: T[],
    year: string,
    month: string
  ): T[] => {
    return data.filter(item => {
      if (item.deleted) return false;

      const itemDate = new Date(item.date);
      const itemYear = itemDate.getFullYear().toString();
      const itemMonth = months[itemDate.getMonth()];

      return itemYear === year && itemMonth === month;
    });
  };

  useEffect(() => {
    if (salesData?.data) {
      const filteredSales = filterDataByDate(salesData.data, selectedYear, selectedMonth);
      const groupedSales = filteredSales.reduce((acc, sale) => {
        const existingIndex = acc.findIndex(item => item.date === sale.date);
        if (existingIndex > -1) {
          acc[existingIndex].amount += sale.amount;
        } else {
          acc.push({ date: sale.date, amount: sale.amount });
        }
        return acc;
      }, [] as SaleItem[]);

      setSalesChartData({
        name: 'Total Sales Amount',
        data: groupedSales.map(item => item.amount)
      });
      setSalesCategories(groupedSales.map(item => item.date));
      console.log('Sales Data:', salesData.data); 
    }

    if (expensesData?.data) {
      const filteredExpenses = filterDataByDate(expensesData.data, selectedYear, selectedMonth);
      const groupedExpenses = filteredExpenses.reduce((acc, expense) => {
        const existingIndex = acc.findIndex(item => item.date === expense.date);
        if (existingIndex > -1) {
          acc[existingIndex].amount += expense.amount;
        } else {
          acc.push({ date: expense.date, amount: expense.amount });
        }
        return acc;
      }, [] as ExpenseItem[]);

      setExpensesChartData({
        name: 'Total Expenses',
        data: groupedExpenses.map(item => item.amount)
      });
      setExpensesCategories(groupedExpenses.map(item => item.date));
      console.log('Expenses Data:', expensesData.data);
    }

    if (procurementData?.data) {
      const filteredProcurements = filterDataByDate(procurementData.data, selectedYear, selectedMonth);
      
      // Sort procurements by date to ensure chronological order
      const sortedProcurements = filteredProcurements.sort((a, b) => 
        new Date(a.date).getTime() - new Date(b.date).getTime()
      );
    
      const groupedProcurement = sortedProcurements.reduce((acc, procurement) => {
        // Robust amount conversion
        const amount = procurement.amount !== undefined 
          ? Number(procurement.amount) 
          : 0;
    
        if (isNaN(amount)) {
          console.warn('Invalid procurement amount:', procurement);
          return acc;
        }
    
        const existingIndex = acc.findIndex(item => item.date === procurement.date);
        if (existingIndex > -1) {
          acc[existingIndex].amount += amount;
        } else {
          acc.push({ date: procurement.date, amount: amount });
        }
        return acc;
      }, [] as ProcurementItem[]);
    
      // Format dates for display
      const formattedCategories = groupedProcurement.map(item => {
        const date = new Date(item.date);
        return date.toLocaleDateString('en-US', { 
          month: 'short', 
          day: 'numeric' 
        });
      });
    
      setProcurementChartData({
        name: 'Total Procurement Amount',
        data: groupedProcurement.map(item => {
          const safeAmount = Number(item.amount) || 0;
          return isNaN(safeAmount) ? 0 : safeAmount;
        })
      });
      setProcurementCategories(formattedCategories);
    }

    if (deploymentsData?.data) {
      const filteredDeployments = deploymentsData.data.filter(item => !item.deleted);
      const deployedCount = filteredDeployments.filter(deployment => deployment.releaseStatus).length;
      const pendingCount = filteredDeployments.filter(deployment => !deployment.releaseStatus).length;
      setDeploymentsStatusData([deployedCount, pendingCount]);
    }

    if (partsData?.data) {
      const filteredParts = partsData.data.filter(item => !item.deleted);
      setPartsQuantityData(filteredParts.map(part => part.qtyLeft));
      setPartsLabels(filteredParts.map(part => part.partName));
    }
  }, [salesData, expensesData, procurementData, deploymentsData, partsData, selectedYear, selectedMonth]);

  const isLoading = salesLoading || expensesLoading || procurementLoading || deploymentsLoading || partsLoading;
  const isError = salesError || expensesError || procurementError || deploymentsError || partsError;

  if (isLoading) {
    return (
      <LoadingDialog 
        open={isLoading}
        loadingMessage="Loading dashboard data..."
      />
    );
  }

  if (isError) {
    return (
      <ErrorDialog 
        open={true}
        errorMessage="Error loading dashboard data"
      />
    );
  }

  return (
    <Paper 
      elevation={3} 
      sx={{ 
        display: 'flex',
        flexDirection: 'column',
        m: 2,

      }}
    >
      <Typography 
        variant="h4" 
        sx={{ 
          p: 2,
          fontWeight: 600,
        }}
      >
        Dashboard
      </Typography>

      {/* Filter controls */}
      <Box sx={{ 
        p: 2,
        display: 'flex', 
        flexDirection: {xs: 'column', md: 'row'},
        gap: 2,
        alignItems: {xs: 'stretch', md: 'center'},
        justifyContent: 'space-between'
      }}>
        <Stack 
          direction={{ xs: 'column', sm: 'row' }} 
          spacing={2} 
          sx={{ flex: 1, alignItems: 'center' }}
        >
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Year</InputLabel>
            <Select
              value={selectedYear}
              label="Year"
              onChange={(e) => setSelectedYear(e.target.value)}
            >
              {years.map((year) => (
                <MenuItem key={year} value={year}>{year}</MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Month</InputLabel>
            <Select
              value={selectedMonth}
              label="Month"
              onChange={(e) => setSelectedMonth(e.target.value)}
            >
              {months.map((month) => (
                <MenuItem key={month} value={month}>{month}</MenuItem>
              ))}
            </Select>
          </FormControl>
        </Stack>
      </Box>

      <Typography 
        variant="h6" 
        sx={{ 
          px: 2,
          pb: 2,
          fontWeight: 500,
        }}
      >
        {`${selectedMonth} ${selectedYear}`}
      </Typography>

      <Box sx={{ 
        flex: 1,
        width: '100%',
        overflow: 'hidden',
        display: 'grid',
        gridTemplateColumns: { 
          xs: '1fr', 
          md: 'repeat(2, 1fr)' 
        },
        gap: 2,
        p: 2,
        pt: 0
      }}>
        {salesChartData.data.length > 0 ? (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography fontSize={18} fontWeight={600} color="#11142D" sx={{ mb: 2 }}>
              Sales Analysis
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <ApexChart
                type="line"
                series={[salesChartData]}
                options={{
                  chart: { height: '100%' },
                  xaxis: {
                    categories: salesCategories,
                    labels: {
                      style: {
                        fontSize: '12px',
                      },
                    },
                  }
                }}
                colors={['#475BE8']}
              />
            </Box>
          </Paper>
        ) : (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
            <Typography variant="h6" color="textSecondary">
              No data available in {selectedMonth} {selectedYear}
            </Typography>
          </Paper>
        )}

        {expensesChartData.data.length > 0 ? (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography fontSize={18} fontWeight={600} color="#11142D" sx={{ mb: 2 }}>
              Expenses Analysis
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <ApexChart
                type="bar"
                series={[expensesChartData]}
                options={{
                  chart: { height: '100%' },
                  xaxis: {
                    categories: expensesCategories,
                    labels: {
                      style: {
                        fontSize: '12px',
                      },
                    },
                  }
                }}
                colors={['#FD8539']}
              />
            </Box>
          </Paper>
        ) : (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
            <Typography variant="h6" color="textSecondary">
              No data available in {selectedMonth} {selectedYear}
            </Typography>
          </Paper>
        )}

        {procurementChartData.data.length > 0 ? (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography fontSize={18} fontWeight={600} color="#11142D" sx={{ mb: 2 }}>
              Procurement Analysis
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <ApexChart
                type="line"
                series={[{
                  name: procurementChartData.name,
                  data: procurementChartData.data.map(val => {
                    const numVal = Number(val);
                    return isNaN(numVal) ? 0 : numVal;
                  })
                }]}
                options={{
                  chart: { 
                    height: '100%',
                  },
                  xaxis: {
                    categories: procurementCategories,
                    labels: {
                      style: {
                        fontSize: '12px',
                      },
                      rotate: -45,
                      trim: true
                    },
                    tickPlacement: 'between'
                  },
                  yaxis: {
                    title: {
                      text: 'Procurement Amount ($)'
                    },
                    labels: {
                      formatter: (value) => `$${Number(value).toLocaleString()}`,
                    },
                    min: 0,
                    forceNiceScale: true
                  },
                  dataLabels: {
                    enabled: false
                  }
                }}
                colors={['#2ED480']}
              />
            </Box>
          </Paper>
        ) : (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
            <Typography variant="h6" color="textSecondary">
              No data available in {selectedMonth} {selectedYear}
            </Typography>
          </Paper>
        )}

        {deploymentsStatusData.length > 0 ? (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography fontSize={18} fontWeight={600} color="#11142D" sx={{ mb: 2 }}>
              Deployment Status
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <ApexChart
                type="pie"
                series={deploymentsStatusData}
                options={{
                  labels: ['Deployed', 'Pending'],
                  legend: {
                    position: 'bottom'
                  }
                }}
                colors={['#475BE8', '#FD8539']}
              />
            </Box>
          </Paper>
        ) : (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
            <Typography variant="h6" color="textSecondary">
              No data available in {selectedMonth} {selectedYear}
            </Typography>
          </Paper>
        )}

        {partsQuantityData.length > 0 ? (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', flexDirection: 'column' }}>
            <Typography fontSize={18} fontWeight={600} color="#11142D" sx={{ mb: 2 }}>
              Parts Distribution
            </Typography>
            <Box sx={{ flex: 1, overflow: 'hidden' }}>
              <ApexChart
                type="donut"
                series={partsQuantityData}
                options={{
                  labels: partsLabels,
                  legend: {
                    position: 'bottom'
                  }
                }}
                colors={['#475BE8', '#FD8539', '#2ED480', '#FE6D8E']}
              />
            </Box>
          </Paper>
        ) : (
          <Paper elevation={2} sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: 300 }}>
            <Typography variant="h6" color="textSecondary">
              No data available in {selectedMonth} {selectedYear}
            </Typography>
          </Paper>
        )}
      </Box>
    </Paper>
  );
};

export default Home;