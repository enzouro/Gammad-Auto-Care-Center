import React, { useContext, useEffect, useState } from 'react';
import ReactApexCharts from 'react-apexcharts';
import { Box, Typography, CircularProgress, Button } from '@pankod/refine-mui';
import axios from 'axios';
import { ApexOptions } from 'apexcharts';
import useDynamicHeight from 'hooks/useDynamicHeight';
import { ColorModeContext, ColorModeContextProvider } from 'contexts';

interface PartSummary {
  totalUsage: number;
  averageMonthlyUsage: number;
  monthsOfData: number;
}

interface ForecastData {
  historical: number[];
  forecast: number[];
  confidence: Array<{
    lower: number;
    upper: number;
  }>;
}

interface TopPart {
  partName: string;
  totalUsage: number;
  partId: string;
}

interface PartDemandForecastData {
  topParts: TopPart[];
  forecasts: Record<string, ForecastData>;
  summaries: Record<string, PartSummary>;
  errors: Record<string, string>;
}

interface PartDemandForecastChartProps {
  endpoint: string;
  title: string;
}

const PartDemandForecastChart: React.FC<PartDemandForecastChartProps> = ({ endpoint, title }) => {
  const [data, setData] = useState<PartDemandForecastData | null>(null);
  const [loading, setLoading] = useState(true);
  const [showChart, setShowChart] = useState(false);
  const containerHeight = useDynamicHeight();
  const { mode } = useContext(ColorModeContext);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`https://gammad-auto-care-center.onrender.com${endpoint}`);
        console.log('Fetched data:', response.data);
        setData(response.data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching forecast:', error);
        setLoading(false);
      }
    };

    if (showChart) {
      fetchForecast();
    }
  }, [endpoint, showChart]);

  const getChartOptions = (partName: string): ApexOptions => {
    const forecastData = data?.forecasts[partName];
    const historicalData = forecastData?.historical || [];
    const forecastDataLength = forecastData?.forecast?.length || 0;

    const currentDate = new Date();
    const historicalMonths = historicalData.map((_, i) => {
      const monthDate = new Date(currentDate.getFullYear(), currentDate.getMonth() - historicalData.length + i + 1);
      return monthDate.toLocaleString('default', { month: 'short', year: 'numeric' });
    });
  
    const forecastMonths = (forecastData?.forecast || []).map((_, i) => {
      const monthDate = new Date(currentDate.getFullYear(), currentDate.getMonth() + i + 1);
      return monthDate.toLocaleString('default', { month: 'short', year: 'numeric' });
    });
  
    const allCategories = [...historicalMonths, ...forecastMonths];

    return {
      chart: {
        type: 'line',
        height: 350,
        width: '100%',
        toolbar: {
          show: true,
          tools: {
            download: true,
            selection: true,
            zoom: false,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true,
          },
        },
        events: {
          mounted: (chartContext) => {
            try {
              // Ensure chartContext and zoomX method exist before calling
              if (chartContext && typeof chartContext.zoomX === 'function') {
                const historicalLength = historicalData.length;
                const forecastLength = forecastData?.forecast?.length || 0;
                
                // Start from 6 months before the forecast or the beginning of historical data
                const startIndex = Math.max(0, historicalLength - 6);
                
                chartContext.zoomX(
                  startIndex, 
                  historicalLength + forecastLength
                );
              }
            } catch (error) {
              console.warn('Zoom initialization error:', error);
            }
          }
        },
        animations: {
          enabled: true,
          easing: 'easeinout',
          speed: 800,
          animateGradually: {
            enabled: true,
            delay: 150,
          },
        },
      },
      markers: {
        size: 1,
      },
      title: {
        text: `Forecast for ${partName}`,
        align: 'left',
        style: {
          color: mode === 'dark' ? '#fff' : '#141414',
        },
      },
      
      xaxis: {
        categories: allCategories,
        title: {
          text: 'Periods',
          style: {
            color: mode === 'dark' ? '#fff' : '#141414',
          },
        },
        labels: {
          rotate: -45,
          trim: true,
          style: {
            fontSize: '12px',
            colors: mode === 'dark' ? '#fff' : '#141414',
          },
        },
        tickAmount: 5, // Limit number of ticks
      },
      yaxis: {
        title: {
          text: 'Demand Quantity',
          style: {
            color: mode === 'dark' ? '#fff' : '#141414',
          },
        },
        labels: {
          formatter: (value) => value !== undefined ? value.toLocaleString('en-US', {
            maximumFractionDigits: 0,
            style: 'currency',
            currency: 'PHP',
          }) : '',
          style: {
            fontSize: '12px',
            colors: mode === 'dark' ? '#fff' : '#141414',
          },
        },
        tickAmount: 5, // Limit number of ticks
      },
      stroke: {
        curve: 'smooth',
        width: [3, 3, 1, 1]
      },
      legend: {
        show: true,
        position: 'top',
        labels: {
          colors: mode === 'dark' ? '#fff' : '#141414',
        },
      },
      fill: {
        type: 'solid',
        opacity: [1, 1, 0.2, 0.2]
      },
      tooltip: {
        shared: true,
        intersect: false,
        theme: mode === 'dark' ? 'dark' : 'light',
        y: {
          formatter: (value) =>
            value !== undefined && value !== null
              ? value.toLocaleString('en-US', {
                  style: 'currency',
                  currency: 'PHP', // Set to PHP
                })
              : 'N/A', // Fallback for undefined or null values
        },
      },
      colors: ['#008FFB', '#FEB019', '#B3F7CA', '#B3F7CA'],
    };
  };

  const getChartSeries = (partName: string) => {
    const forecastData = data?.forecasts[partName];
    const historicalData = forecastData?.historical || [];
    const forecast = forecastData?.forecast || [];
    const confidence = forecastData?.confidence || [];

    const upperBound = confidence.map(c => c.upper);
    const lowerBound = confidence.map(c => c.lower);

    return [
      {
        name: 'Historical',
        data: historicalData
      },
      {
        name: 'Forecast',
        data: [...Array(historicalData.length).fill(null), ...forecast]
      },
      {
        name: 'Upper Bound',
        data: [...Array(historicalData.length).fill(null), ...upperBound]
      },
      {
        name: 'Lower Bound',
        data: [...Array(historicalData.length).fill(null), ...lowerBound]
      }
    ];
  };

  if (!showChart) {
    return (
      <Box className="p-4 bg-white rounded-lg shadow-md">
        <Typography variant="h6" className="mb-4">
        Part Demand Forecast
        </Typography>
        <Button
        variant="contained"
        color="primary"
        onClick={() => setShowChart(!showChart)}
        sx={{ marginBottom: 2 }}
      >
        {showChart ? 'Hide Forecast' : 'Show Forecast'}
      </Button>
      </Box>
    );
  }

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!data || !data.topParts || data.topParts.length === 0) {
    return (
      <Box sx={{ p: 2 }}>
        <Typography color="error">No forecast data available</Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h6" className="mb-4">
        {title}
      </Typography>
      <Button
        variant="contained"
        color="primary"
        onClick={() => setShowChart(!showChart)}
        sx={{ marginBottom: 2 }}
      >
        {showChart ? 'Hide Forecast' : 'Show Forecast'}
      </Button>
      {showChart && (
        <>
          {loading ? (
            <Box className="flex justify-center items-center h-64">
              <CircularProgress />
            </Box>
          ) : data ? (
            <Box>
              {data.topParts.map((part) => {
                const summary = data.summaries[part.partName];
                const error = data.errors[part.partName];
                const forecastData = data.forecasts[part.partName];

                if (loading) {
                  return (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                      <CircularProgress />
                    </Box>
                  );
                }

                if (error) {
                  return (
                    <Box key={part.partId} sx={{ mb: 4 }}>
                      <Typography color="error">
                        Error forecasting {part.partName}: {error}
                      </Typography>
                    </Box>
                  );
                }

                if (!forecastData || !forecastData.forecast || !forecastData.confidence) {
                  return (
                    <Box key={part.partId} sx={{ mb: 4 }}>
                      <Typography color="warning">
                        No forecast data available for {part.partName}
                      </Typography>
                    </Box>
                  );
                }

                return (
                  <Box key={part.partId} sx={{ mb: 6 }}>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="h6">{part.partName}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Total Usage: {summary.totalUsage.toLocaleString()}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Average Monthly Usage: {Math.round(summary.averageMonthlyUsage * 100) / 100}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Months of Historical Data: {summary.monthsOfData}
                      </Typography>
                    </Box>
                    <Box alignItems='center' justifyContent='center' display='flex-row'>
                      <ReactApexCharts
                        options={getChartOptions(part.partName)}
                        series={getChartSeries(part.partName)}
                        type="line"
                        height={350}
                        width="100%"
                      />
                    </Box>
                  </Box>
                );
              })}
            </Box>
          ) : (
            <Typography color="error">No data available</Typography>
          )}
        </>
      )}
    </Box>
  );
};

export default PartDemandForecastChart;