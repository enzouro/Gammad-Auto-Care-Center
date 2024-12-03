// ApexChart.tsx
import { ApexOptions } from 'apexcharts';
import React from 'react';
import Chart from 'react-apexcharts';

interface ApexChartProps {
  type: 'line' | 'bar' | 'area' | 'pie' | 'donut';
  series: any; // This allows for both array of objects and array of numbers
  options?: ApexOptions;
  colors?: string[];
}

const ApexChart: React.FC<ApexChartProps> = ({ type, series, options = {}, colors }) => {
    // Ensure series is valid
    const safeSeries = React.useMemo(() => {
      if (!series) return [];
      
      // For numeric series
      if (Array.isArray(series)) {
        return series.map(item => {
          if (typeof item === 'number') {
            return isNaN(item) ? 0 : item;
          }
          
          // For object series with data
          if (item && item.data) {
            return {
              ...item,
              data: item.data.map((val: any) => {
                const numVal = Number(val);
                return isNaN(numVal) ? 0 : numVal;
              })
            };
          }
          
          return item;
        });
      }
      
      return series;
    }, [series]);

  const defaultOptions: ApexOptions = {
    chart: {
      type,
      toolbar: {
        show: false,
      },
    },
    colors,
    dataLabels: {
      enabled: false,
    },
    legend: {
      position: type === 'pie' || type === 'donut' ? 'bottom' : 'top',
    },
    stroke: {
      curve: 'smooth',
    },
  };

  const mergedOptions = {
    ...defaultOptions,
    ...options,
  };

  return (
    <Chart
      options={mergedOptions}
      series={safeSeries}
      type={type}
      height={350}
    />
  );
};

export default ApexChart;