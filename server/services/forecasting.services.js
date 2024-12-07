import expenseModel from '../mongodb/models/expense.js';        // Importing expense model
import procurementModel from '../mongodb/models/procurement.js';        // Importing procurement model
import saleModel from '../mongodb/models/sale.js';        // Importing sales model
import deploymentModel from '../mongodb/models/deployment.js';        // Importing deployment model
import moment from 'moment';        // Importing Moment.js for date handling
import * as tf from '@tensorflow/tfjs-node';        // Importing TensorFlow.js for machine learning

// Existing MODEL_CACHE for other forecasts
const MODEL_CACHE = new Map();        // Cache for general forecasts
// New cache specifically for seasonal forecasts
const SEASONAL_FORECAST_CACHE = new Map();        // Cache for seasonal part demand
const CACHE_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days in milliseconds

export const generateForecast = async (model, field, dateField, periods, interval = 'month', partName = '') => {        // Main function for generating forecasts
    const modelKey = `${model.modelName}_${field}_${interval}`;        // Unique key for model cache
    const cachedModel = MODEL_CACHE.get(modelKey);        // Check if model is cached

    console.log(`Generating forecast for model: ${modelKey}, with interval: ${interval}`);        // Log current operation

    if (cachedModel && Date.now() - cachedModel.timestamp < CACHE_DURATION) {        // If cached model is still valid
        console.log(`Using cached model for ${modelKey}`);        // Log cache hit
        return generatePredictions(cachedModel.model, cachedModel.historicalValues, periods, cachedModel.mean, cachedModel.std);        // Use cached model
    }

    const data = await model        // Fetch data from the database
        .find({ deleted: false })        // Filter out deleted records
        .sort({ [dateField]: 1 })        // Sort by date field in ascending order
        .select(`${field} ${dateField}`);        // Select only required fields

    if (data.length < 2) {        // Check if there's enough data
        console.error('Insufficient data for forecasting.');        // Log error if data is insufficient
        return {
            historical: [],        // Return empty historical data
            forecast: Array(periods).fill(null),        // Return empty forecast
            confidence: Array(periods).fill(null),        // Return empty confidence intervals
        };
    }

    const timeSeries = data.map(item => ({        // Transform data into time series format
        value: item[field],        // Map value field
        date: moment(item[dateField]).startOf(interval).toDate(),        // Round date to the interval start
    }));

    const groupedData = {};        // Object to hold grouped data
    timeSeries.forEach(({ value, date }) => {        // Loop through time series data
        const key = moment(date).format(`YYYY-MM-${interval === 'month' ? '01' : 'DD'}`);        // Format date as key
        groupedData[key] = (groupedData[key] || 0) + value;        // Sum values for each date key
    });

    const sortedKeys = Object.keys(groupedData).sort();        // Sort grouped data by keys (dates)
    const historicalValues = sortedKeys.map(key => groupedData[key]);        // Extract historical values

    if (historicalValues.length < 2) {        // Check if historical data is sufficient
        console.error('Insufficient historical data.');        // Log error if not enough historical data
        return {
            historical: historicalValues,        // Return available historical data
            forecast: Array(periods).fill(null),        // Return empty forecast
            confidence: Array(periods).fill(null),        // Return empty confidence intervals
        };
    }

    const { normalizedValues, mean, std } = normalizeData(historicalValues);        // Normalize the data for training
    const inputValues = normalizedValues.slice(0, -1);        // Input values for training
    const outputValues = normalizedValues.slice(1);        // Output values for training

    if (inputValues.length === 0 || outputValues.length === 0) {        // Check if there's enough training data
        console.error('Training data insufficient after normalization.');        // Log error if not enough data
        return {
            historical: historicalValues,        // Return historical values
            forecast: Array(periods).fill(null),        // Return empty forecast
            confidence: Array(periods).fill(null),        // Return empty confidence intervals
        };
    }

    const xs = tf.tensor2d(inputValues, [inputValues.length, 1]);        // Create tensor for input data
    const ys = tf.tensor2d(outputValues, [outputValues.length, 1]);        // Create tensor for output data

    const modelTensor = tf.sequential();        // Create a new TensorFlow model
    modelTensor.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [1] }));        // Add dense layer with 32 units
    modelTensor.add(tf.layers.dropout({ rate: 0.2 }));        // Add dropout layer to prevent overfitting
    modelTensor.add(tf.layers.dense({ units: 16, activation: 'relu' }));        // Add dense layer with 16 units
    modelTensor.add(tf.layers.dense({ units: 1 }));        // Add output layer with 1 unit

    modelTensor.compile({        // Compile the model
        optimizer: tf.train.adam(0.001),        // Use Adam optimizer with learning rate 0.001
        loss: 'meanSquaredError',        // Use mean squared error as loss function
        metrics: ['mse'],        // Track mean squared error during training
    });

    await modelTensor.fit(xs, ys, {        // Train the model
        epochs: 200,        // Train for 200 epochs
        validationSplit: 0.2,        // Use 20% of data for validation
        callbacks: {        // Callbacks for monitoring training
            onEpochEnd: (epoch, logs) => {        // On each epoch end
                if (epoch > 50 && logs.val_loss > logs.loss * 1.5) {        // Stop training if validation loss diverges
                    modelTensor.stopTraining = true;        // Stop training
                }
            },
        },
    });

    MODEL_CACHE.set(modelKey, {        // Cache the trained model
        model: modelTensor,        // Store the trained model
        historicalValues,        // Store historical values
        timestamp: Date.now(),        // Store cache timestamp
        mean,        // Store mean value for denormalization
        std,        // Store standard deviation for denormalization
    });

    return generatePredictions(modelTensor, historicalValues, periods, mean, std);        // Generate predictions using the trained model
};

function normalizeData(data) { // Function to normalize data using mean and standard deviation
    const mean = data.reduce((a, b) => a + b, 0) / data.length; // Calculate mean
    const std = Math.sqrt(data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length); // Calculate standard deviation
    return {
        normalizedValues: data.map(value => (value - mean) / std), // Normalize each value
        mean, // Return the calculated mean
        std  // Return the calculated standard deviation
    };
}

function generatePredictions(model, historicalValues, periods, mean, std) { 
    // Function to generate predictions for future periods using a given model
    let current = historicalValues[historicalValues.length - 1]; // Start with the last historical value
    const predictions = []; // Array to store predictions

    for (let i = 0; i < periods; i++) { 
        // Loop for the number of forecast periods
        const normalizedInput = (current - mean) / std; // Normalize current input
        const prediction = model.predict(tf.tensor2d([normalizedInput], [1, 1])); // Generate prediction using model

        current = prediction.dataSync()[0] * std + mean; // Denormalize prediction
        predictions.push(Math.max(0, current)); // Ensure predictions are non-negative
    }

    return {
        historical: historicalValues, // Include historical values
        forecast: predictions,        // Add forecast values
        confidence: calculateConfidenceIntervals(predictions) // Calculate confidence intervals
    };
}

function calculateConfidenceIntervals(predictions) { 
    // Calculate confidence intervals for predictions
    const forecastMean = predictions.reduce((acc, val) => acc + val, 0) / predictions.length; // Mean of predictions
    const forecastVariance = predictions.reduce((acc, val) => acc + Math.pow(val - forecastMean, 2), 0) / predictions.length; // Variance
    const forecastStdDev = Math.sqrt(forecastVariance); // Standard deviation

    return predictions.map(pred => ({
        lower: Math.max(0, pred - 1.96 * forecastStdDev), // Lower bound of 95% CI
        upper: pred + 1.96 * forecastStdDev               // Upper bound of 95% CI
    }));
}

// New function to get top used parts
async function getTopPartsUsage(deployments, limit = 5) { 
    // Aggregate total usage by part
    const partTotalUsage = {};

    deployments.forEach(deployment => { 
        // Iterate through deployments
        deployment.parts.forEach(({ part, quantityUsed }) => { 
            // Iterate through parts in each deployment
            if (!part || !part.partName || !quantityUsed || isNaN(quantityUsed)) return; // Skip invalid data

            if (!partTotalUsage[part.partName]) {
                partTotalUsage[part.partName] = {
                    totalUsage: 0,  // Initialize total usage
                    partId: part._id // Store part ID
                };
            }
            partTotalUsage[part.partName].totalUsage += quantityUsed; // Add quantity used
        });
    });

    // Convert usage object to sorted array and return top `limit` parts
    return Object.entries(partTotalUsage)
        .map(([partName, data]) => ({
            partName,
            totalUsage: data.totalUsage, // Total usage of the part
            partId: data.partId          // Part ID
        }))
        .sort((a, b) => b.totalUsage - a.totalUsage) // Sort by usage (descending)
        .slice(0, limit); // Return top parts
}

// Enhanced seasonal part demand forecasting with top parts analysis
export const forecastSeasonalPartDemand = async (periods, topLimit = 5) => {
    const cacheKey = `seasonal_${periods}_${topLimit}`; // Key for caching
    const cachedForecast = SEASONAL_FORECAST_CACHE.get(cacheKey); // Check cache
    
    if (cachedForecast && Date.now() - cachedForecast.timestamp < CACHE_DURATION) { 
        // Return cached result if valid
        console.log('Using cached seasonal forecast');
        return cachedForecast.data;
    }

    const deployments = await deploymentModel
        .find({ deleted: false, 'parts.part': { $exists: true, $ne: null } }) // Get deployments
        .populate({
            path: 'parts.part',
            match: { deleted: false }
        })
        .sort({ date: 1 }); // Sort by date (ascending)

    const topParts = await getTopPartsUsage(deployments, topLimit); // Get top parts

    const partUsageByMonth = {}; // Initialize storage for monthly usage of top parts
    topParts.forEach(part => {
        partUsageByMonth[part.partName] = {}; // Create entry for each top part
    });

    deployments.forEach(deployment => {
        if (!deployment.date) return; // Skip if no date
        const month = moment(deployment.date).format('YYYY-MM'); // Format date as year-month

        deployment.parts.forEach(({ part, quantityUsed }) => {
            if (!part || !part.partName || !quantityUsed || isNaN(quantityUsed)) return; // Skip invalid data
            
            if (partUsageByMonth.hasOwnProperty(part.partName)) { // Check if part is in top parts
                if (!partUsageByMonth[part.partName][month]) {
                    partUsageByMonth[part.partName][month] = 0; // Initialize month entry
                }
                partUsageByMonth[part.partName][month] += quantityUsed; // Add usage for the month
            }
        });
    });

    const forecasts = {}; // Store forecasts for each part
    const forecastErrors = {}; // Store errors during forecasting
    const summaries = {}; // Store usage summaries

    for (const topPart of topParts) {
        const partName = topPart.partName;
        const monthlyUsage = partUsageByMonth[partName]; // Get monthly usage for part
        const timeSeriesData = Object.values(monthlyUsage); // Convert to time series data

        try {
            if (timeSeriesData.length >= 12) { // Ensure sufficient data (at least 12 months)
                forecasts[partName] = await generateSeasonalForecast(timeSeriesData, periods); // Generate forecast
                summaries[partName] = {
                    totalUsage: topPart.totalUsage, // Total usage
                    averageMonthlyUsage: topPart.totalUsage / timeSeriesData.length, // Average monthly usage
                    monthsOfData: timeSeriesData.length // Data span in months
                };
            } else {
                forecastErrors[partName] = 'Insufficient historical data'; // Log insufficient data error
            }
        } catch (error) {
            forecastErrors[partName] = error.message; // Log error message
            console.error(`Forecast error for ${partName}:`, error);
        }
    }

    const result = {
        topParts, // Top parts data
        forecasts, // Forecast results
        summaries, // Usage summaries
        errors: forecastErrors // Forecasting errors
    };

    SEASONAL_FORECAST_CACHE.set(cacheKey, { // Cache the result
        data: result,
        timestamp: Date.now()
    });

    return result; // Return the result
};

async function generateSeasonalForecast(historicalData, periods) {  
    // Handle very short time series
    if (historicalData.length < 3) { 
        return {
            historicalDates: [], // Add historical dates
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
        const inputValues = deseasonalizedData.slice(0, -1);
        const outputValues = deseasonalizedData.slice(1);

        if (inputValues.length === 0 || outputValues.length === 0) {
            throw new Error('Insufficient data for training');
        }

        // Generate historical dates (assuming monthly data)
        const historicalDates = Array.from(
            { length: historicalData.length }, 
            (_, i) => moment().subtract(historicalData.length - i, 'months').format('MMM YYYY')
        );

        const { normalizedValues, mean, std } = normalizeData(deseasonalizedData);

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

        let current = deseasonalizedData[deseasonalizedData.length - 1];
        const predictions = [];
        const confidenceIntervals = [];

        // Generate forecast dates
        const forecastDates = Array.from(
            { length: periods }, 
            (_, i) => moment().add(i + 1, 'months').format('MMM YYYY')
        );

        for (let i = 0; i < periods; i++) {
            const normalizedInput = (current - mean) / std;
            const prediction = modelTensor.predict(tf.tensor2d([normalizedInput], [1, 1]));
            
            current = prediction.dataSync()[0] * std + mean;
            predictions.push(Math.max(0, current));
        }

        const forecastMean = predictions.reduce((acc, val) => acc + val, 0) / predictions.length;
        const forecastVariance = predictions.reduce((acc, val) => acc + Math.pow(val - forecastMean, 2), 0) / predictions.length;
        const forecastStdDev = Math.sqrt(forecastVariance);

        const confidence = predictions.map(pred => ({
            lower: Math.max(0, pred - 1.96 * forecastStdDev),
            upper: pred + 1.96 * forecastStdDev
        }));

        const seasonalForecast = predictions.map((value, index) => 
            Math.max(0, value * (seasonalFactors[index % seasonalityPeriod] || 1)));

        const seasonalConfidence = confidence.map((conf, index) => ({
            lower: Math.max(0, conf.lower * (seasonalFactors[index % seasonalityPeriod] || 1)),
            upper: conf.upper * (seasonalFactors[index % seasonalityPeriod] || 1)
        }));

        return {
            historicalDates, // Add historical dates
            forecastDates,   // Add forecast dates
            historical: historicalData,
            forecast: seasonalForecast,
            confidence: seasonalConfidence
        };

    } catch (error) {
        console.error('Forecast generation error:', error);
        return {
            historicalDates: [], // Empty dates on error
            forecastDates: [],   // Empty forecast dates on error
            historical: historicalData,
            forecast: Array(periods).fill(null),
            confidence: Array(periods).fill(null)
        };
    }
}
// Export other forecasting functions
export const forecastProcurementExpenses = async (periods) =>
    generateForecast(procurementModel, 'amount', 'date', periods); // Function for procurement expense forecasting

export const forecastSales = async (periods) =>
    generateForecast(saleModel, 'amount', 'date', periods); // Function for sales forecasting

export const forecastExpenses = async (periods) =>
    generateForecast(expenseModel, 'amount', 'date', periods); // Function for general expense forecasting
