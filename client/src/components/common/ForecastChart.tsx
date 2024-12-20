import React, { useContext, useEffect, useState } from 'react';
import ReactApexCharts from 'react-apexcharts';
import { Box, Typography, CircularProgress, Button } from '@pankod/refine-mui';
import axios from 'axios';
import { ApexOptions } from 'apexcharts';
import useDynamicHeight from 'hooks/useDynamicHeight';
import { ColorModeContext } from 'contexts';

interface ForecastData {
  historical: number[];
  forecast: number[];
  confidence: Array<{
    lower: number;
    upper: number;
  }>;
}

interface ForecastChartProps {
  endpoint: string;
  title: string;
}

const ForecastChart: React.FC<ForecastChartProps> = ({ endpoint, title }) => {
  const [data, setData] = useState<ForecastData | null>(null);
  const [loading, setLoading] = useState(true);
  const [showChart, setShowChart] = useState(false); // State to manage chart visibility
  const containerHeight = useDynamicHeight();
  const { mode } = useContext(ColorModeContext);


  useEffect(() => {
    const fetchForecast = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`https://gammad-auto-care-center.onrender.com${endpoint}`);
        console.log('Fetched data:', response.data); // Debugging log
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

  const getChartOptions = (data: ForecastData): ApexOptions => {
    const currentDate = new Date();
    const historicalMonths = data.historical.map((_, i) => {
      const monthDate = new Date(currentDate.getFullYear(), currentDate.getMonth() - data.historical.length + i + 1);
      return monthDate.toLocaleString('default', { month: 'short', year: 'numeric' });
    });

  
    const forecastMonths = data.forecast.map((_, i) => {
      const monthDate = new Date(currentDate.getFullYear(), currentDate.getMonth() + i + 1);
      return monthDate.toLocaleString('default', { month: 'short', year: 'numeric' });
    });
  
    const allCategories = [...historicalMonths, ...forecastMonths];

    return {
      chart: {
        type: 'line',
        height: 350,
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
                const historicalLength = data.historical.length;
                const forecastLength = data.forecast.length;
                
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
      stroke: {
        width: [3, 3, 1, 1],
        dashArray: [0, 0, 0, 0],
        curve: 'smooth',
      },
      xaxis: {
        categories: allCategories,
        
        labels: {
          rotate: -45,
          trim: true,
          style: {
            colors: mode === 'dark' ? '#fff' : '#141414',
          },
        },
        title: {
          text: 'Month-Year',
          style: {
            color: mode === 'dark' ? '#fff' : '#141414',
          }
        }
      },
      yaxis: {
        title: {
          text: 'Value',
          style: {
            color: mode === 'dark' ? '#fff' : '#141414',
          },
        },
        labels: {
          formatter: (value) =>
            value !== undefined && value !== null
              ? value.toLocaleString('en-US', {
                  maximumFractionDigits: 0,
                  style: 'currency',
                  currency: 'PHP', // Set to PHP
                })
              : 'N/A', // Fallback if value is undefined or null
              style: {
                colors: mode === 'dark' ? '#fff' : '#141414',
              },
        },
        tickAmount: 5, // Limit number of ticks
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
      legend: {
        position: 'top',
        horizontalAlign: 'center',
        labels: {
          colors: mode === 'dark' ? '#fff' : '#141414',
        },
      },
      colors: ['#008FFB', '#FEB019', '#B3F7CA', '#B3F7CA'],
      fill: {
        type: ['solid', 'solid', 'solid', 'solid'],
        opacity: [1, 1, 0.4, 0.4],
      },

    };
  };

  const getSeries = (data: ForecastData) => {
    

    const historicalSeries = {
      name: 'Historical',
      type: 'line',
      data: data.historical,
    };

    const forecastSeries = {
      name: 'Forecast',
      type: 'line',
      data: [...Array(data.historical.length).fill(null), ...data.forecast],
    };

    const upperBoundSeries = {
      name: 'Upper Bound',
      type: 'line',
      data: [
        ...Array(data.historical.length).fill(null),
        ...(data.confidence ? data.confidence.map((c) => c.upper) : []),
      ],
    };

    const lowerBoundSeries = {
      name: 'Lower Bound',
      type: 'line',
      data: [
        ...Array(data.historical.length).fill(null),
        ...(data.confidence ? data.confidence.map((c) => c.lower) : []),
      ],
    };

    return [historicalSeries, forecastSeries, upperBoundSeries, lowerBoundSeries];
  };
  

  return (
    <Box className="p-4 bg-white rounded-lg shadow-md">
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
            <Box className="w-full h-96">
              <ReactApexCharts
                options={getChartOptions(data)}
                series={getSeries(data)}
                height={350}
                width="100%"
              />
            </Box>
          ) : (
            <Typography color="error">No data available</Typography>
          )}
        </>
      )}
    </Box>
  );
};

export default ForecastChart;
