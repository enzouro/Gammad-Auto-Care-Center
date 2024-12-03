//server\services\turnaround.services.js
import deploymentModel from '../mongodb/models/deployment.js';
import moment from 'moment';
import * as tf from '@tensorflow/tfjs-node';

// Cache for turnaround forecasts
const TURNAROUND_FORECAST_CACHE = new Map();
const CACHE_DURATION = 7 * 24 * 60 * 60 * 1000; // 7 days in milliseconds

export const analyzeTurnaroundTimes = async (periods = 12) => {
    // Check cache first
    const cacheKey = `turnaround_${periods}`;
    const cachedAnalysis = TURNAROUND_FORECAST_CACHE.get(cacheKey);
    
    if (cachedAnalysis && Date.now() - cachedAnalysis.timestamp < CACHE_DURATION) {
        console.log('Using cached turnaround analysis');
        return cachedAnalysis.data;
    }

    try {
        // Fetch all non-deleted deployments
        const deployments = await deploymentModel
            .find({ deleted: false })
            .sort({ arrivalDate: 1 });

        // Initialize data structures for different metrics
        const monthlyMetrics = {};
        const detailedTurnaroundTimes = [];

        deployments.forEach(deployment => {
            // Skip if essential dates are missing
            if (!deployment.arrivalDate) return;

            const arrivalDate = moment(deployment.arrivalDate);
            const monthKey = arrivalDate.format('YYYY-MM');

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
            if (deployment.repairStatus && deployment.repairedDate) {
                const repairEndDate = moment(deployment.repairedDate);
                const repairDuration = repairEndDate.diff(arrivalDate, 'hours');

                if (deployment.repairStatus === 'cancelled') {
                    monthlyMetrics[monthKey].cancelledRepairs++;
                } else if (['pending', 'in progress'].includes(deployment.repairStatus)) {
                    monthlyMetrics[monthKey].pendingRepairs++;
                }

                if (repairDuration > 0) {
                    monthlyMetrics[monthKey].totalRepairTime += repairDuration;
                }
            }

            // Calculate total turnaround time (from arrival to release)
            if (deployment.releaseStatus && deployment.releaseDate) {
                const releaseDate = moment(deployment.releaseDate);
                const turnaroundTime = releaseDate.diff(arrivalDate, 'hours');

                if (turnaroundTime > 0) {
                    monthlyMetrics[monthKey].totalTurnaroundTime += turnaroundTime;
                    monthlyMetrics[monthKey].completedDeployments++;

                    detailedTurnaroundTimes.push({
                        month: monthKey,
                        turnaroundTime
                    });
                }
            }
        });

        // Calculate averages and prepare time series data
        const timeSeriesData = Object.entries(monthlyMetrics)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([month, metrics]) => ({
                month,
                avgRepairTime: metrics.totalRepairTime / (metrics.totalDeployments - metrics.pendingRepairs) || 0,
                avgTurnaroundTime: metrics.totalTurnaroundTime / metrics.completedDeployments || 0,
                completionRate: (metrics.completedDeployments / metrics.totalDeployments) * 100,
                cancellationRate: (metrics.cancelledRepairs / metrics.totalDeployments) * 100
            }));

        // Prepare data for forecasting
        const turnaroundTimes = timeSeriesData.map(d => d.avgTurnaroundTime);
        const repairTimes = timeSeriesData.map(d => d.avgRepairTime);

        // Generate forecasts
        const turnaroundForecast = await generateTimeForecast(turnaroundTimes, periods);
        const repairTimeForecast = await generateTimeForecast(repairTimes, periods);

        // Calculate efficiency metrics
        const efficiencyMetrics = calculateEfficiencyMetrics(timeSeriesData);

        const result = {
            historical: {
                timeSeriesData,
                efficiencyMetrics
            },
            forecasts: {
                turnaroundTime: turnaroundForecast,
                repairTime: repairTimeForecast
            },
            recommendations: generateRecommendations(efficiencyMetrics, turnaroundForecast)
        };

        // Cache the results
        TURNAROUND_FORECAST_CACHE.set(cacheKey, {
            data: result,
            timestamp: Date.now()
        });

        return result;
    } catch (error) {
        console.error('Error analyzing turnaround times:', error);
        throw error;
    }
};

async function generateTimeForecast(timeSeriesData, periods) {
    if (timeSeriesData.length < 2) {
        return {
            forecast: Array(periods).fill(null),
            confidence: Array(periods).fill(null)
        };
    }

    // Simple data normalization
    const mean = timeSeriesData.reduce((acc, val) => acc + val, 0) / timeSeriesData.length;
    const squaredDiffs = timeSeriesData.map(val => Math.pow(val - mean, 2));
    const std = Math.sqrt(squaredDiffs.reduce((acc, val) => acc + val, 0) / timeSeriesData.length) || 1;
    
    const normalizedValues = timeSeriesData.map(val => (val - mean) / std);
    
    // Prepare training data
    const inputValues = normalizedValues.slice(0, -1);
    const outputValues = normalizedValues.slice(1);

    // Create and train model
    const xs = tf.tensor2d(inputValues, [inputValues.length, 1]);
    const ys = tf.tensor2d(outputValues, [outputValues.length, 1]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [1] }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError'
    });

    await model.fit(xs, ys, {
        epochs: 200,
        validationSplit: 0.2,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                if (epoch > 50 && logs.val_loss > logs.loss * 1.5) {
                    model.stopTraining = true;
                }
            }
        }
    });

    // Generate predictions
    let current = timeSeriesData[timeSeriesData.length - 1];
    const predictions = [];

    for (let i = 0; i < periods; i++) {
        const normalizedInput = (current - mean) / std;
        const prediction = model.predict(tf.tensor2d([normalizedInput], [1, 1]));
        current = prediction.dataSync()[0] * std + mean;
        predictions.push(Math.max(0, current));
    }

    // Calculate confidence intervals
    const confidence = calculateConfidenceIntervals(predictions);

    return {
        forecast: predictions,
        confidence
    };
}

function calculateEfficiencyMetrics(timeSeriesData) {
    const recentMonths = timeSeriesData.slice(-3);
    
    return {
        avgTurnaroundTime: recentMonths.reduce((acc, d) => acc + d.avgTurnaroundTime, 0) / recentMonths.length,
        avgRepairTime: recentMonths.reduce((acc, d) => acc + d.avgRepairTime, 0) / recentMonths.length,
        completionRate: recentMonths.reduce((acc, d) => acc + d.completionRate, 0) / recentMonths.length,
        trend: calculateTrend(timeSeriesData.map(d => d.avgTurnaroundTime))
    };
}

function calculateTrend(data) {
    if (data.length < 2) return 'insufficient_data';
    
    const recentAvg = data.slice(-3).reduce((a, b) => a + b, 0) / 3;
    const previousAvg = data.slice(-6, -3).reduce((a, b) => a + b, 0) / 3;
    
    const percentageChange = ((recentAvg - previousAvg) / previousAvg) * 100;
    
    if (percentageChange < -5) return 'improving';
    if (percentageChange > 5) return 'deteriorating';
    return 'stable';
}

function generateRecommendations(metrics, forecast) {
    const recommendations = [];

    if (metrics.trend === 'deteriorating') {
        recommendations.push({
            priority: 'high',
            area: 'turnaround_time',
            suggestion: 'Consider reviewing resource allocation and workflow processes to address increasing turnaround times.'
        });
    }

    if (metrics.completionRate < 85) {
        recommendations.push({
            priority: 'high',
            area: 'completion_rate',
            suggestion: 'Implement measures to improve completion rate, such as better initial assessment and resource planning.'
        });
    }

    if (metrics.avgRepairTime > 48) { // If average repair time is more than 48 hours
        recommendations.push({
            priority: 'medium',
            area: 'repair_efficiency',
            suggestion: 'Consider optimizing repair processes or increasing repair capacity to reduce average repair time.'
        });
    }

    return recommendations;
}

function calculateConfidenceIntervals(predictions) {
    const forecastStdDev = Math.sqrt(
        predictions.reduce((acc, val) => {
            const diff = val - predictions.reduce((a, b) => a + b, 0) / predictions.length;
            return acc + (diff * diff);
        }, 0) / predictions.length
    );

    return predictions.map(pred => ({
        lower: Math.max(0, pred - 1.96 * forecastStdDev),
        upper: pred + 1.96 * forecastStdDev
    }));
}