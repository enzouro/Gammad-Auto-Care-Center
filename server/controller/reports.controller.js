// server\controller\report.controller.js
import dotenv from 'dotenv';
import mongoose from 'mongoose';

import Sale from '../mongodb/models/sale.js';
import Expense from '../mongodb/models/expense.js';
import Procurement from '../mongodb/models/procurement.js';
import Deployment from '../mongodb/models/deployment.js';

// Load environment variables
dotenv.config();

export const generateReport = async (req, res) => {
    try {
        // Extract month, year, and date range from query parameters
        const { month, year, startDate, endDate } = req.query;

        // Build date filter
        const dateFilter = {
            deleted: false // Only fetch non-deleted items
        };

        // Add month and year filtering if specified
        if (month && year) {
            const startOfMonth = new Date(year, month - 1, 1);
            const endOfMonth = new Date(year, month, 0);
            dateFilter.date = {
                $gte: startOfMonth,
                $lte: endOfMonth
            };
        } 
        // If specific date range is provided, use it
        else if (startDate && endDate) {
            dateFilter.date = { 
                $gte: new Date(startDate), 
                $lte: new Date(endDate) 
            };
        }

        // Sales Summary
        const sales = await Sale.find(dateFilter);
        const salesSummary = {
            tableHeader: ['Total Amount', 'Total Net of VAT', 'Total Output VAT'],
            tableData: [{
                totalAmount: sales.reduce((total, sale) => total + (sale.amount || 0), 0),
                totalNetOfVAT: sales.reduce((total, sale) => total + (sale.netOfVAT || 0), 0),
                totalOutputVAT: sales.reduce((total, sale) => total + (sale.outputVAT || 0), 0)
            }]
        };

        // Expenses Summary
        const expenses = await Expense.find(dateFilter);
        const expensesSummary = {
            tableHeader: ['Total Amount', 'Total Net of VAT', 'Non-VAT Amount', 'No Valid Receipt Amount'],
            tableData: [{
                totalAmount: expenses.reduce((total, expense) => total + (expense.amount || 0), 0),
                totalNetOfVAT: expenses.reduce((total, expense) => total + (expense.netOfVAT || 0), 0),
                nonVatAmount: expenses.reduce((total, expense) => 
                    total + (expense.isNonVat ? (expense.amount || 0) : 0), 0),
                noValidReceiptAmount: expenses.reduce((total, expense) => 
                    total + (expense.noValidReceipt ? (expense.amount || 0) : 0), 0)
            }]
        };

        // Procurement Summary
        const procurements = await Procurement.find(dateFilter);
        const procurementSummary = {
            tableHeader: ['Total Amount', 'Total Net of VAT', 'Non-VAT Amount', 'No Valid Receipt Amount'],
            tableData: [{
                totalAmount: procurements.reduce((total, procurement) => total + (procurement.amount || 0), 0),
                totalNetOfVAT: procurements.reduce((total, procurement) => total + (procurement.netOfVAT || 0), 0),
                nonVatAmount: procurements.reduce((total, procurement) => 
                    total + (procurement.isNonVat ? (procurement.amount || 0) : 0), 0),
                noValidReceiptAmount: procurements.reduce((total, procurement) => 
                    total + (procurement.noValidReceipt ? (procurement.amount || 0) : 0), 0)
            }]
        };

        // Deployment Summary
        const deployments = await Deployment.find(dateFilter).populate('parts.part');
        const deploymentSummary = {
            tableHeader: ['Total Parts Deployed', 'Average Turnaround Time (Days)'],
            tableData: [{
                totalPartsDeployed: deployments.reduce((sum, deployment) => 
                    sum + deployment.parts.reduce((qtySum, part) => qtySum + (part.quantityUsed || 0), 0)
                , 0),
                averageTurnaroundTime: (() => {
                    const validDeployments = deployments.filter(d => d.arrivalDate && d.releaseDate);
                    if (validDeployments.length === 0) return 0;

                    const totalTurnaround = validDeployments.reduce((sum, deployment) => {
                        const arrival = new Date(deployment.arrivalDate);
                        const release = new Date(deployment.releaseDate);
                        return sum + (release - arrival) / (1000 * 60 * 60 * 24); // Days difference
                    }, 0);

                    return totalTurnaround / validDeployments.length;
                })()
            }]
        };

        // Consolidated Report
        res.status(200).json({
            salesSummary,
            expensesSummary,
            procurementSummary,
            deploymentSummary,
            filterCriteria: { 
                month, 
                year, 
                startDate, 
                endDate 
            }
        });
    } catch (error) {
        console.error('Report Generation Error:', error);
        res.status(500).json({ 
            message: 'Error generating report', 
            error: error.message,
            stack: process.env.NODE_ENV === 'development' ? error.stack : undefined 
        });
    }
};