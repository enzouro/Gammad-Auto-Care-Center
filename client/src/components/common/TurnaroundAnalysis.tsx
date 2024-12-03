import React, { useEffect, useState } from 'react';
import ReactApexCharts from 'react-apexcharts';
import { Box, Typography, CircularProgress, Button } from '@pankod/refine-mui';
import axios from 'axios';
import { ApexOptions } from 'apexcharts';
import useDynamicHeight from 'hooks/useDynamicHeight';

interface TurnaroundData {
  historical: {
    timeSeriesData: Array<{
      month: string;
      avgRepairTime: number;
      avgTurnaroundTime: number;
      completionRate: number;
      cancellationRate: number;
    }>;
    efficiencyMetrics: {
      avgTurnaroundTime: number;
      avgRepairTime: number;
      completionRate: number;
      trend: 'improving' | 'deteriorating' | 'stable' | 'insufficient_data';
    };
  };
  forecasts: {
    turnaroundTime: {
      forecast: number[];
      confidence: Array<{
        lower: number;
        upper: number;
      }>;
    };
    repairTime: {
      forecast: number[];
      confidence: Array<{
        lower: number;
        upper: number;
      }>;
    };
  };
  recommendations: Array<{
    priority: 'high' | 'medium' | 'low';
    area: string;
    suggestion: string;
  }>;
}

interface TurnaroundAnalysisProps {
  endpoint: string;
  title: string;
}

const TurnaroundAnalysis: React.FC<TurnaroundAnalysisProps> = ({ endpoint, title }) => {
  const [data, setData] = useState<TurnaroundData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showChart, setShowChart] = useState(false);
  const containerHeight = useDynamicHeight();

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await axios.get(`http://localhost:8080${endpoint}`);
        console.log('Fetched data:', response.data); // Debugging log
        
        // Access the nested data property
        const analysisData = response.data.data;
        
        // Enhanced validation
        if (!analysisData?.historical?.efficiencyMetrics || 
            typeof analysisData.historical.efficiencyMetrics.avgTurnaroundTime !== 'number' ||
            typeof analysisData.historical.efficiencyMetrics.completionRate !== 'number') {
          throw new Error('Invalid or incomplete data structure received from server');
        }
        
        setData(analysisData);
      } catch (err) {
        console.error('Error fetching analysis:', err);
        setError(err instanceof Error ? err.message : 'An error occurred while fetching data');
      } finally {
        setLoading(false);
      }
    };

    if (showChart) {
      fetchAnalysis();
    }
  }, [endpoint, showChart]);

  const getChartOptions = (): ApexOptions => {
    if (!data?.historical?.timeSeriesData) return {};

    const historicalMonths = data.historical.timeSeriesData.map(d => d.month) || [];
    const forecastMonths = data.forecasts?.turnaroundTime?.forecast?.map((_, i) => 
      `Forecast ${i + 1}`
    ) || [];

    return {
      chart: {
        type: 'line',
        height: containerHeight,
        toolbar: {
          show: true,
          tools: {
            download: true,
            selection: true,
            zoom: true,
            zoomin: true,
            zoomout: true,
            pan: true,
            reset: true,
          },
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
        width: [3, 3, 2, 2, 2, 2],
        dashArray: [0, 0, 0, 0, 0, 0],
        curve: 'smooth',
      },
      xaxis: {
        categories: [...historicalMonths, ...forecastMonths],
        labels: {
          rotate: -45,
          trim: true,
        },
      },
      yaxis: {
        title: {
          text: 'Hours',
        },
        labels: {
          formatter: (value) => value != null ? value.toFixed(1) : '0.0',
        },
      },
      tooltip: {
        shared: true,
        intersect: false,
        y: {
          formatter: (value) => value != null ? `${value.toFixed(1)} hours` : 'N/A',
        },
      },
      legend: {
        position: 'top',
        horizontalAlign: 'center',
      },
      colors: ['#008FFB', '#FEB019', '#00E396', '#FF4560', '#B3F7CA', '#B3F7CA'],
      fill: {
        type: ['solid', 'solid', 'solid', 'solid', 'solid', 'solid'],
        opacity: [1, 1, 1, 1, 0.4, 0.4],
      },
    };
  };

  const getSeries = () => {
    if (!data?.historical?.timeSeriesData || !data?.forecasts) return [];

    const { timeSeriesData } = data.historical;
    const { turnaroundTime, repairTime } = data.forecasts;

    const turnaroundHistorical = {
      name: 'Turnaround Time',
      type: 'line',
      data: timeSeriesData.map(d => d.avgTurnaroundTime),
    };

    const repairHistorical = {
      name: 'Repair Time',
      type: 'line',
      data: timeSeriesData.map(d => d.avgRepairTime),
    };

    const turnaroundForecast = {
      name: 'Turnaround Forecast',
      type: 'line',
      data: [...Array(timeSeriesData.length).fill(null), ...(turnaroundTime?.forecast || [])],
    };

    const repairForecast = {
      name: 'Repair Forecast',
      type: 'line',
      data: [...Array(timeSeriesData.length).fill(null), ...(repairTime?.forecast || [])],
    };

    const turnaroundConfidence = {
      name: 'Turnaround Confidence (Upper)',
      type: 'line',
      data: [
        ...Array(timeSeriesData.length).fill(null),
        ...(turnaroundTime?.confidence?.map(c => c.upper) || []),
      ],
    };

    const turnaroundConfidenceLower = {
      name: 'Turnaround Confidence (Lower)',
      type: 'line',
      data: [
        ...Array(timeSeriesData.length).fill(null),
        ...(turnaroundTime?.confidence?.map(c => c.lower) || []),
      ],
    };

    return [
      turnaroundHistorical,
      repairHistorical,
      turnaroundForecast,
      repairForecast,
      turnaroundConfidence,
      turnaroundConfidenceLower,
    ];
  };

  const renderMetrics = () => {
    if (!data?.historical?.efficiencyMetrics) return null;

    const { efficiencyMetrics } = data.historical;
    const getTrendColor = (trend: string) => {
      switch (trend) {
        case 'improving':
          return 'success.main';
        case 'deteriorating':
          return 'error.main';
        default:
          return 'warning.main';
      }
    };

    return (
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2, mb: 3 }}>
        <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="subtitle2">Average Turnaround Time</Typography>
          <Typography variant="h6">
            {(efficiencyMetrics.avgTurnaroundTime || 0).toFixed(1)} hours
          </Typography>
        </Box>
        <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="subtitle2">Completion Rate</Typography>
          <Typography variant="h6">
            {(efficiencyMetrics.completionRate || 0).toFixed(1)}%
          </Typography>
        </Box>
        <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="subtitle2">Trend</Typography>
          <Typography variant="h6" color={getTrendColor(efficiencyMetrics.trend)}>
            {efficiencyMetrics.trend.charAt(0).toUpperCase() + efficiencyMetrics.trend.slice(1)}
          </Typography>
        </Box>
      </Box>
    );
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
        {showChart ? 'Hide Analysis' : 'Show Analysis'}
      </Button>

      {showChart && (
        <>
          {loading ? (
            <Box className="flex justify-center items-center h-64">
              <CircularProgress />
            </Box>
          ) : error ? (
            <Typography color="error">{error}</Typography>
          ) : data ? (
            <>
              {renderMetrics()}
              <Box className="w-full h-96">
                <ReactApexCharts
                  options={getChartOptions()}
                  series={getSeries()}
                  height={350}
                  width="100%"
                />
              </Box>
              {data.recommendations && (
                <Box sx={{ mt: 4 }}>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    Recommendations
                  </Typography>
                  {data.recommendations.map((rec, index) => (
                    <Box
                      key={index}
                      sx={{
                        p: 2,
                        mb: 2,
                        bgcolor: 'background.paper',
                        borderRadius: 1,
                        borderLeft: 6,
                        borderColor: rec.priority === 'high' ? 'error.main' : 'warning.main',
                      }}
                    >
                      <Typography variant="subtitle1" sx={{ mb: 1 }}>
                        {rec.area.replace('_', ' ').toUpperCase()} - {rec.priority.toUpperCase()} Priority
                      </Typography>
                      <Typography variant="body2">{rec.suggestion}</Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </>
          ) : (
            <Typography color="error">No data available</Typography>
          )}
        </>
      )}
    </Box>
  );
};

export default TurnaroundAnalysis;