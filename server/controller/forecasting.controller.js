// server/controller/forecasting.controller.js
import {
  forecastProcurementExpenses,
  forecastSeasonalPartDemand,
  forecastSales,
  forecastExpenses,
} from '../services/forecasting.services.js';
import { analyzeTurnaroundTimes } from '../services/turnaround.services.js';

export const getProcurementForecast = async (req, res) => {
  try {
    const periods = parseInt(req.query.periods || 3);
    const forecast = await forecastProcurementExpenses(periods);
    res.status(200).json(forecast);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getPartDemandForecast = async (req, res) => {
  try {
    const periods = parseInt(req.query.periods || 6);
    const topLimit = parseInt(req.query.topLimit || 5);
    const forecast = await forecastSeasonalPartDemand(periods, topLimit);
    res.status(200).json(forecast);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getSalesForecast = async (req, res) => {
  try {
    const periods = parseInt(req.query.periods || 3);
    const forecast = await forecastSales(periods);
    res.status(200).json(forecast);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getExpensesForecast = async (req, res) => {
  try {
    const periods = parseInt(req.query.periods || 3);
    const forecast = await forecastExpenses(periods);
    res.status(200).json(forecast);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export const getTurnaroundForecast = async (req, res) => {
  try {
    const periods = parseInt(req.query.periods || 12); // Default to 12 periods for turnaround analysis
    const analysis = await analyzeTurnaroundTimes(periods);
    
    res.status(200).json({
      success: true,
      data: analysis,
      metadata: {
        periods,
        generatedAt: new Date().toISOString(),
        dataPoints: analysis.historical.timeSeriesData.length
      }
    });
  } catch (error) {
    console.error('Error in turnaround forecast:', error);
    res.status(500).json({ 
      success: false, 
      message: error.message,
      error: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
};