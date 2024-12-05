//server\services\turnaround.services.js
import deploymentModel from '../mongodb/models/deployment.js'; // Import the deployment model for database operations
import moment from 'moment'; // Import moment.js for date manipulation
import * as tf from '@tensorflow/tfjs-node'; // Import TensorFlow.js for machine learning tasks

// Cache for turnaround forecasts
const TURNAROUND_FORECAST_CACHE = new Map(); // Map object to store forecast results in memory
const CACHE_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days in milliseconds

export const analyzeTurnaroundTimes = async (periods = 12) => { // Function to analyze and forecast turnaround times
    // Check cache first
    const cacheKey = `turnaround_${periods}`; // Unique key for caching based on periods
    const cachedAnalysis = TURNAROUND_FORECAST_CACHE.get(cacheKey); // Retrieve from cache if available
    
    if (cachedAnalysis && Date.now() - cachedAnalysis.timestamp < CACHE_DURATION) { // Check if cache is valid
        console.log('Using cached turnaround analysis'); // Log cache hit
        return cachedAnalysis.data; // Return cached result
    }

    try {
        // Fetch all non-deleted deployments from the database
        const deployments = await deploymentModel
            .find({ deleted: false }) // Exclude deleted records
            .sort({ arrivalDate: 1 }); // Sort deployments by arrival date

        // Initialize data structures for different metrics
        const monthlyMetrics = {}; // Object to store metrics grouped by month
        const detailedTurnaroundTimes = []; // Array for storing individual turnaround times

        deployments.forEach(deployment => { // Process each deployment
            // Skip if essential dates are missing
            if (!deployment.arrivalDate) return; // Continue to the next iteration if no arrival date

            const arrivalDate = moment(deployment.arrivalDate); // Parse arrival date
            const monthKey = arrivalDate.format('YYYY-MM'); // Generate a key for grouping by month

            // Initialize monthly metrics if not exists
            if (!monthlyMetrics[monthKey]) { 
                monthlyMetrics[monthKey] = {
                    totalDeployments: 0,
                    totalRepairTime: 0,
                    totalTurnaroundTime: 0,
                    completedDeployments: 0,
                    cancelledRepairs: 0,
                    pendingRepairs: 0,
                };
            }

            // Increment total deployments
            monthlyMetrics[monthKey].totalDeployments++;

            // Calculate repair time
            if (deployment.repairStatus && deployment.repairedDate) { // Check if repair data exists
                const repairEndDate = moment(deployment.repairedDate); // Parse repair end date
                const repairDuration = repairEndDate.diff(arrivalDate, 'hours'); // Calculate repair duration in hours

                if (deployment.repairStatus === 'cancelled') {
                    monthlyMetrics[monthKey].cancelledRepairs++; // Increment cancelled repairs count
                } else if (['pending', 'in progress'].includes(deployment.repairStatus)) {
                    monthlyMetrics[monthKey].pendingRepairs++; // Increment pending repairs count
                }

                if (repairDuration > 0) {
                    monthlyMetrics[monthKey].totalRepairTime += repairDuration; // Accumulate repair duration
                }
            }

            // Calculate total turnaround time (from arrival to release)
            if (deployment.releaseStatus && deployment.releaseDate) { // Check if release data exists
                const releaseDate = moment(deployment.releaseDate); // Parse release date
                const turnaroundTime = releaseDate.diff(arrivalDate, 'hours'); // Calculate turnaround time in hours

                if (turnaroundTime > 0) {
                    monthlyMetrics[monthKey].totalTurnaroundTime += turnaroundTime; // Accumulate turnaround time
                    monthlyMetrics[monthKey].completedDeployments++; // Increment completed deployments count

                    detailedTurnaroundTimes.push({
                        month: monthKey,
                        turnaroundTime // Record turnaround time details
                    });
                }
            }
        });

        // Calculate averages and prepare time series data
        const timeSeriesData = Object.entries(monthlyMetrics)
            .sort(([a], [b]) => a.localeCompare(b)) // Sort data by month
            .map(([month, metrics]) => ({
                month, // Month key
                avgRepairTime: metrics.totalRepairTime / (metrics.totalDeployments - metrics.pendingRepairs) || 0, // Calculate average repair time
                avgTurnaroundTime: metrics.totalTurnaroundTime / metrics.completedDeployments || 0, // Calculate average turnaround time
                completionRate: (metrics.completedDeployments / metrics.totalDeployments) * 100, // Calculate completion rate
                cancellationRate: (metrics.cancelledRepairs / metrics.totalDeployments) * 100 // Calculate cancellation rate
            }));

        // Prepare data for forecasting
        const turnaroundTimes = timeSeriesData.map(d => d.avgTurnaroundTime); // Extract average turnaround times
        const repairTimes = timeSeriesData.map(d => d.avgRepairTime); // Extract average repair times

        // Generate forecasts for turnaround and repair times
        const turnaroundForecast = await generateTimeForecast(turnaroundTimes, periods);
        const repairTimeForecast = await generateTimeForecast(repairTimes, periods);

        // Calculate efficiency metrics
        const efficiencyMetrics = calculateEfficiencyMetrics(timeSeriesData); // Analyze efficiency trends

        const result = {
            historical: { // Historical data summary
                timeSeriesData,
                efficiencyMetrics
            },
            forecasts: { // Forecast data
                turnaroundTime: turnaroundForecast,
                repairTime: repairTimeForecast
            },
            recommendations: generateRecommendations(efficiencyMetrics, turnaroundForecast) // Generate recommendations
        };

        // Cache the results
        TURNAROUND_FORECAST_CACHE.set(cacheKey, { 
            data: result, // Save result to cache
            timestamp: Date.now() // Record the cache timestamp
        });

        return result; // Return the final result
    } catch (error) {
        console.error('Error analyzing turnaround times:', error); // Log any errors
        throw error; // Re-throw error for handling elsewhere
    }
};

async function generateTimeForecast(timeSeriesData, periods) { // Function to generate time series forecast using machine learning
    if (timeSeriesData.length < 2) { // If there is insufficient data, return null forecasts
        return {
            forecast: Array(periods).fill(null), // Fill the forecast array with null values
            confidence: Array(periods).fill(null) // Fill the confidence array with null values
        };
    }

    // Simple data normalization
    const mean = timeSeriesData.reduce((acc, val) => acc + val, 0) / timeSeriesData.length; // Calculate the mean of the data
    const squaredDiffs = timeSeriesData.map(val => Math.pow(val - mean, 2)); // Calculate squared differences from the mean
    const std = Math.sqrt(squaredDiffs.reduce((acc, val) => acc + val, 0) / timeSeriesData.length) || 1; // Calculate standard deviation, default to 1 if no variance
    
    const normalizedValues = timeSeriesData.map(val => (val - mean) / std); // Normalize the data by subtracting mean and dividing by std
    
    // Prepare training data for the model
    const inputValues = normalizedValues.slice(0, -1); // Use all values except the last one as inputs
    const outputValues = normalizedValues.slice(1); // Use all values except the first one as outputs

    // Create and train the machine learning model
    const xs = tf.tensor2d(inputValues, [inputValues.length, 1]); // Convert input data into tensor format
    const ys = tf.tensor2d(outputValues, [outputValues.length, 1]); // Convert output data into tensor format

    // Create a sequential model for training
    const model = tf.sequential(); // Initialize a sequential model
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [1] })); // Add first dense layer with ReLU activation
    model.add(tf.layers.dropout({ rate: 0.2 })); // Add dropout to avoid overfitting
    model.add(tf.layers.dense({ units: 16, activation: 'relu' })); // Add second dense layer with ReLU activation
    model.add(tf.layers.dense({ units: 1 })); // Output layer with 1 unit

    model.compile({
        optimizer: tf.train.adam(0.001), // Adam optimizer with learning rate of 0.001
        loss: 'meanSquaredError' // Use mean squared error as loss function
    });

    // Train the model
    await model.fit(xs, ys, {
        epochs: 200, // Train for 200 epochs
        validationSplit: 0.2, // Use 20% of data for validation
        callbacks: {
            onEpochEnd: (epoch, logs) => { // Callback to stop training if validation loss is too high
                if (epoch > 50 && logs.val_loss > logs.loss * 1.5) {
                    model.stopTraining = true; // Stop training if validation loss grows too much
                }
            }
        }
    });

    // Generate predictions using the trained model
    let current = timeSeriesData[timeSeriesData.length - 1]; // Start predictions from the last data point
    const predictions = []; // Array to store predictions

    for (let i = 0; i < periods; i++) { // Loop through the number of periods to forecast
        const normalizedInput = (current - mean) / std; // Normalize the current value
        const prediction = model.predict(tf.tensor2d([normalizedInput], [1, 1])); // Predict the next value
        current = prediction.dataSync()[0] * std + mean; // Denormalize the prediction and update the current value
        predictions.push(Math.max(0, current)); // Ensure the prediction is not negative and store it
    }

    // Calculate confidence intervals for the predictions
    const confidence = calculateConfidenceIntervals(predictions);

    return {
        forecast: predictions, // Return the forecasted values
        confidence // Return the confidence intervals for the forecast
    };
}

function calculateEfficiencyMetrics(timeSeriesData) { // Function to calculate efficiency metrics from time series data
    const recentMonths = timeSeriesData.slice(-3); // Get data from the last 3 months

    return {
        avgTurnaroundTime: recentMonths.reduce((acc, d) => acc + d.avgTurnaroundTime, 0) / recentMonths.length, // Calculate average turnaround time
        avgRepairTime: recentMonths.reduce((acc, d) => acc + d.avgRepairTime, 0) / recentMonths.length, // Calculate average repair time
        completionRate: recentMonths.reduce((acc, d) => acc + d.completionRate, 0) / recentMonths.length, // Calculate average completion rate
        trend: calculateTrend(timeSeriesData.map(d => d.avgTurnaroundTime)) // Calculate trend based on average turnaround times
    };
}

function calculateTrend(data) { // Function to calculate the trend in the data
    if (data.length < 2) return 'insufficient_data'; // If there is not enough data, return 'insufficient_data'

    const recentAvg = data.slice(-3).reduce((a, b) => a + b, 0) / 3; // Calculate the average of the last 3 data points
    const previousAvg = data.slice(-6, -3).reduce((a, b) => a + b, 0) / 3; // Calculate the average of the 3 data points before the recent ones
    
    const percentageChange = ((recentAvg - previousAvg) / previousAvg) * 100; // Calculate the percentage change between the two periods
    
    if (percentageChange < -5) return 'improving'; // If the change is negative and exceeds 5%, return 'improving'
    if (percentageChange > 5) return 'deteriorating'; // If the change is positive and exceeds 5%, return 'deteriorating'
    return 'stable'; // Otherwise, return 'stable'
}

function generateRecommendations(metrics, forecast) { // Function to generate recommendations based on metrics and forecast
    const recommendations = []; // Initialize an empty array to store recommendations

    if (metrics.trend === 'deteriorating') { // If the trend is deteriorating, suggest improving turnaround time
        recommendations.push({
            priority: 'high',
            area: 'turnaround_time',
            suggestion: 'Consider reviewing resource allocation and workflow processes to address increasing turnaround times.'
        });
    }

    if (metrics.completionRate < 85) { // If the completion rate is less than 85%, suggest improving it
        recommendations.push({
            priority: 'high',
            area: 'completion_rate',
            suggestion: 'Implement measures to improve completion rate, such as better initial assessment and resource planning.'
        });
    }

    if (metrics.avgRepairTime > 48) { // If the average repair time exceeds 48 hours, suggest improving repair efficiency
        recommendations.push({
            priority: 'medium',
            area: 'repair_efficiency',
            suggestion: 'Consider optimizing repair processes or increasing repair capacity to reduce average repair time.'
        });
    }

    return recommendations; // Return the generated recommendations
}

function calculateConfidenceIntervals(predictions) { // Function to calculate confidence intervals for the forecasted predictions
    const forecastStdDev = Math.sqrt(
        predictions.reduce((acc, val) => { // Calculate the standard deviation of the predictions
            const diff = val - predictions.reduce((a, b) => a + b, 0) / predictions.length; // Difference from the mean
            return acc + (diff * diff); // Sum of squared differences
        }, 0) / predictions.length
    );

    return predictions.map(pred => ({ // For each prediction, calculate the confidence interval
        lower: Math.max(0, pred - 1.96 * forecastStdDev), // Lower bound of the confidence interval (using 95% confidence)
        upper: pred + 1.96 * forecastStdDev // Upper bound of the confidence interval (using 95% confidence)
    }));
}
