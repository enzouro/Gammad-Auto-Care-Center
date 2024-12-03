import expenseModel from '../mongodb/models/expense.js';
import procurementModel from '../mongodb/models/procurement.js';
import saleModel from '../mongodb/models/sale.js';
import deploymentModel from '../mongodb/models/deployment.js';
import moment from 'moment';
import * as tf from '@tensorflow/tfjs-node';

// Existing MODEL_CACHE for other forecasts
const MODEL_CACHE = new Map();
// New cache specifically for seasonal forecasts
const SEASONAL_FORECAST_CACHE = new Map();
const CACHE_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days in milliseconds

export const generateForecast = async (model, field, dateField, periods, interval = 'month', partName = '') => {
    const modelKey = `${model.modelName}_${field}_${interval}`;
    const cachedModel = MODEL_CACHE.get(modelKey);

    console.log(`Generating forecast for model: ${modelKey}, with interval: ${interval}`);

    if (cachedModel && Date.now() - cachedModel.timestamp < CACHE_DURATION) {
        console.log(`Using cached model for ${modelKey}`);
        return generatePredictions(cachedModel.model, cachedModel.historicalValues, periods, cachedModel.mean, cachedModel.std);
    }

    const data = await model
        .find({ deleted: false })
        .sort({ [dateField]: 1 })
        .select(`${field} ${dateField}`);

    if (data.length < 2) {
        console.error('Insufficient data for forecasting.');
        return {
            historical: [],
            forecast: Array(periods).fill(null),
            confidence: Array(periods).fill(null),
        };
    }

    const timeSeries = data.map(item => ({
        value: item[field],
        date: moment(item[dateField]).startOf(interval).toDate(),
    }));

    const groupedData = {};
    timeSeries.forEach(({ value, date }) => {
        const key = moment(date).format(`YYYY-MM-${interval === 'month' ? '01' : 'DD'}`);
        groupedData[key] = (groupedData[key] || 0) + value;
    });

    const sortedKeys = Object.keys(groupedData).sort();
    const historicalValues = sortedKeys.map(key => groupedData[key]);

    if (historicalValues.length < 2) {
        console.error('Insufficient historical data.');
        return {
            historical: historicalValues,
            forecast: Array(periods).fill(null),
            confidence: Array(periods).fill(null),
        };
    }

    const { normalizedValues, mean, std } = normalizeData(historicalValues);
    const inputValues = normalizedValues.slice(0, -1);
    const outputValues = normalizedValues.slice(1);

    if (inputValues.length === 0 || outputValues.length === 0) {
        console.error('Training data insufficient after normalization.');
        return {
            historical: historicalValues,
            forecast: Array(periods).fill(null),
            confidence: Array(periods).fill(null),
        };
    }

    const xs = tf.tensor2d(inputValues, [inputValues.length, 1]);
    const ys = tf.tensor2d(outputValues, [outputValues.length, 1]);

    const modelTensor = tf.sequential();
    modelTensor.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [1] }));
    modelTensor.add(tf.layers.dropout({ rate: 0.2 }));
    modelTensor.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    modelTensor.add(tf.layers.dense({ units: 1 }));

    modelTensor.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
        metrics: ['mse'],
    });

    await modelTensor.fit(xs, ys, {
        epochs: 200,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (epoch > 50 && logs.val_loss > logs.loss * 1.5) {
                    modelTensor.stopTraining = true;
                }
            },
        },
    });

    MODEL_CACHE.set(modelKey, {
        model: modelTensor,
        historicalValues,
        timestamp: Date.now(),
        mean,
        std,
    });

    return generatePredictions(modelTensor, historicalValues, periods, mean, std);
};

function normalizeData(data) {
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const std = Math.sqrt(data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length);
    return {
        normalizedValues: data.map(value => (value - mean) / std),
        mean,
        std
    };
}

function generatePredictions(model, historicalValues, periods, mean, std) {
    let current = historicalValues[historicalValues.length - 1];
    const predictions = [];

    for (let i = 0; i < periods; i++) {
        // Normalize input
        const normalizedInput = (current - mean) / std;
        const prediction = model.predict(tf.tensor2d([normalizedInput], [1, 1]));
        
        // Denormalize prediction
        current = prediction.dataSync()[0] * std + mean;
        predictions.push(Math.max(0, current)); // Ensure non-negative predictions
    }

    return {
        historical: historicalValues,
        forecast: predictions,
        confidence: calculateConfidenceIntervals(predictions)
    };
}

function calculateConfidenceIntervals(predictions) {
    const forecastMean = predictions.reduce((acc, val) => acc + val, 0) / predictions.length;
    const forecastVariance = predictions.reduce((acc, val) => acc + Math.pow(val - forecastMean, 2), 0) / predictions.length;
    const forecastStdDev = Math.sqrt(forecastVariance);

    return predictions.map(pred => ({
        lower: Math.max(0, pred - 1.96 * forecastStdDev), // 95% confidence interval
        upper: pred + 1.96 * forecastStdDev
    }));
}

// New function to get top used parts
async function getTopPartsUsage(deployments, limit = 5) {
    // Aggregate total usage by part
    const partTotalUsage = {};
    
    deployments.forEach(deployment => {
        deployment.parts.forEach(({ part, quantityUsed }) => {
            if (!part || !part.partName || !quantityUsed || isNaN(quantityUsed)) return;
            
            if (!partTotalUsage[part.partName]) {
                partTotalUsage[part.partName] = {
                    totalUsage: 0,
                    partId: part._id
                };
            }
            partTotalUsage[part.partName].totalUsage += quantityUsed;
        });
    });

    // Convert to array and sort by total usage
    return Object.entries(partTotalUsage)
        .map(([partName, data]) => ({
            partName,
            totalUsage: data.totalUsage,
            partId: data.partId
        }))
        .sort((a, b) => b.totalUsage - a.totalUsage)
        .slice(0, limit);
    }

// Enhanced seasonal part demand forecasting with top parts analysis
export const forecastSeasonalPartDemand = async (periods, topLimit = 5) => {
    // Check if there's a cached result that's still valid
    const cacheKey = `seasonal_${periods}_${topLimit}`;
    const cachedForecast = SEASONAL_FORECAST_CACHE.get(cacheKey);
    
    if (cachedForecast && Date.now() - cachedForecast.timestamp < CACHE_DURATION) {
        console.log('Using cached seasonal forecast');
        return cachedForecast.data;
    }

    const deployments = await deploymentModel
    .find({
      deleted: false,
      'parts.part': { $exists: true, $ne: null }
    })
    .populate({
      path: 'parts.part',
      match: { deleted: false }
    })
    .sort({ date: 1 });

    // Get top parts first
    const topParts = await getTopPartsUsage(deployments, topLimit);
    
    // Initialize storage for monthly usage of top parts
    const partUsageByMonth = {};
    topParts.forEach(part => {
        partUsageByMonth[part.partName] = {};
    });

    // Calculate monthly usage for top parts
    deployments.forEach(deployment => {
        if (!deployment.date) return;
        const month = moment(deployment.date).format('YYYY-MM');
        
        deployment.parts.forEach(({ part, quantityUsed }) => {
            if (!part || !part.partName || !quantityUsed || isNaN(quantityUsed)) return;
            
            // Only process if it's one of the top parts
            if (partUsageByMonth.hasOwnProperty(part.partName)) {
                if (!partUsageByMonth[part.partName][month]) {
                    partUsageByMonth[part.partName][month] = 0;
                }
                partUsageByMonth[part.partName][month] += quantityUsed;
            }
        });
    });

    // Generate forecasts for top parts
    const forecasts = {};
    const forecastErrors = {};
    const summaries = {};

    for (const topPart of topParts) {
        const partName = topPart.partName;
        const monthlyUsage = partUsageByMonth[partName];
        const timeSeriesData = Object.values(monthlyUsage);
        
        try {
            if (timeSeriesData.length >= 12) {
                forecasts[partName] = await generateSeasonalForecast(timeSeriesData, periods);
                // Add usage summary
                summaries[partName] = {
                    totalUsage: topPart.totalUsage,
                    averageMonthlyUsage: topPart.totalUsage / timeSeriesData.length,
                    monthsOfData: timeSeriesData.length
                };
            } else {
                forecastErrors[partName] = 'Insufficient historical data';
            }
        } catch (error) {
            forecastErrors[partName] = error.message;
            console.error(`Forecast error for ${partName}:`, error);
        }
    }

    const result = {
        topParts,
        forecasts,
        summaries,
        errors: forecastErrors
    };

    // Cache the result
    SEASONAL_FORECAST_CACHE.set(cacheKey, {
        data: result,
        timestamp: Date.now()
    });

    return result;
}

async function generateSeasonalForecast(historicalData, periods) {
    // Handle very short time series
    if (historicalData.length < 3) {
        return {
            historical: historicalData,
            forecast: Array(periods).fill(null),
            confidence: Array(periods).fill(null)
        };
    }

    // Simplified seasonal decomposition for smaller datasets
    const seasonalityPeriod = Math.min(12, historicalData.length);
    const seasons = Array(seasonalityPeriod).fill(0);
    
    // Calculate seasonal indices
    for (let i = 0; i < historicalData.length; i++) {
        seasons[i % seasonalityPeriod] += historicalData[i];
    }
    
    const seasonalFactors = seasons.map(s => 
        s / (Math.floor(historicalData.length / seasonalityPeriod)));

    // Remove seasonality from data
    const deseasonalizedData = historicalData.map((value, index) => 
        value / (seasonalFactors[index % seasonalityPeriod] || 1));

    try {
        // Create input sequence for the model
        const inputValues = deseasonalizedData.slice(0, -1);
        const outputValues = deseasonalizedData.slice(1);

        if (inputValues.length === 0 || outputValues.length === 0) {
            throw new Error('Insufficient data for training');
        }

        // Normalize the data
        const { normalizedValues, mean, std } = normalizeData(deseasonalizedData);

        // Create and train the model directly
        const xs = tf.tensor2d(normalizedValues.slice(0, -1), [inputValues.length, 1]);
        const ys = tf.tensor2d(normalizedValues.slice(1), [outputValues.length, 1]);

        const modelTensor = tf.sequential();
        modelTensor.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [1] }));
        modelTensor.add(tf.layers.dropout({ rate: 0.2 }));
        modelTensor.add(tf.layers.dense({ units: 16, activation: 'relu' }));
        modelTensor.add(tf.layers.dense({ units: 1 }));

        modelTensor.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError',
            metrics: ['mse'],
        });

        await modelTensor.fit(xs, ys, {
            epochs: 200,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch > 50 && logs.val_loss > logs.loss * 1.5) {
                        modelTensor.stopTraining = true;
                    }
                },
            },
        });

        // Generate predictions
        let current = deseasonalizedData[deseasonalizedData.length - 1];
        const predictions = [];
        const confidenceIntervals = [];

        for (let i = 0; i < periods; i++) {
            // Normalize input
            const normalizedInput = (current - mean) / std;
            const prediction = modelTensor.predict(tf.tensor2d([normalizedInput], [1, 1]));
            
            // Denormalize prediction
            current = prediction.dataSync()[0] * std + mean;
            predictions.push(Math.max(0, current));
        }

        // Calculate confidence intervals
        const forecastMean = predictions.reduce((acc, val) => acc + val, 0) / predictions.length;
        const forecastVariance = predictions.reduce((acc, val) => acc + Math.pow(val - forecastMean, 2), 0) / predictions.length;
        const forecastStdDev = Math.sqrt(forecastVariance);

        const confidence = predictions.map(pred => ({
            lower: Math.max(0, pred - 1.96 * forecastStdDev),
            upper: pred + 1.96 * forecastStdDev
        }));

        // Reapply seasonality to forecast and confidence intervals
        const seasonalForecast = predictions.map((value, index) => 
            Math.max(0, value * (seasonalFactors[index % seasonalityPeriod] || 1)));

        const seasonalConfidence = confidence.map((conf, index) => ({
            lower: Math.max(0, conf.lower * (seasonalFactors[index % seasonalityPeriod] || 1)),
            upper: conf.upper * (seasonalFactors[index % seasonalityPeriod] || 1)
        }));

        return {
            historical: historicalData,
            forecast: seasonalForecast,
            confidence: seasonalConfidence
        };

    } catch (error) {
        console.error('Forecast generation error:', error);
        return {
            historical: historicalData,
            forecast: Array(periods).fill(null),
            confidence: Array(periods).fill(null)
        };
    }
}

// Export other forecasting functions
export const forecastProcurementExpenses = async (periods) =>
    generateForecast(procurementModel, 'amount', 'date', periods);

export const forecastSales = async (periods) =>
    generateForecast(saleModel, 'amount', 'date', periods);

export const forecastExpenses = async (periods) =>
    generateForecast(expenseModel, 'amount', 'date', periods);
