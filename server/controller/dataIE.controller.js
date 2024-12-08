import mongoose from 'mongoose';
import csv from 'csvtojson';
import { Parser } from 'json2csv';
import fs from 'fs';
import path from 'path';

// Import Models
import procurementModel from '../mongodb/models/procurement.js';
import deploymentModel from '../mongodb/models/deployment.js';
import saleModel from '../mongodb/models/sale.js';
import expenseModel from '../mongodb/models/expense.js';

// Utility function to get model based on data type
const getModelByType = (type) => {
    const models = {
        'procurements': procurementModel,
        'deployments': deploymentModel,
        'sales': saleModel,
        'expenses': expenseModel
    };
    
    const model = models[type.toLowerCase()];
    if (!model) throw new Error('Invalid model type');
    return model;
};

// Import Controller
export const importData = async (req, res) => {
    // Validate file upload
    if (!req.file) {
        return res.status(400).json({ 
            message: 'No file uploaded', 
            error: 'File is required for import' 
        });
    }

    const { type } = req.params;
    
    try {
        // Validate file type
        const allowedFileTypes = ['.csv', '.xlsx'];
        const fileExt = path.extname(req.file.originalname).toLowerCase();
        if (!allowedFileTypes.includes(fileExt)) {
            // Remove uploaded file
            fs.unlinkSync(req.file.path);
            return res.status(400).json({ 
                message: 'Invalid file type', 
                error: `Only ${allowedFileTypes.join(', ')} files are allowed` 
            });
        }

        // Get appropriate model
        const Model = getModelByType(type);

        // Convert CSV to JSON
        const jsonArray = await csv().fromFile(req.file.path);

        // Validate input data
        if (jsonArray.length === 0) {
            fs.unlinkSync(req.file.path);
            return res.status(400).json({ 
                message: 'Empty file', 
                error: 'No data found in the uploaded file' 
            });
        }

        // Start a mongoose session for transaction
        const session = await mongoose.startSession();
        session.startTransaction();

        try {
            // Bulk write operation for better performance
            const bulkOps = jsonArray.map(doc => ({
                insertOne: {
                    document: {
                        ...doc,
                        // Remove creator reference if no authentication
                        date: doc.date || new Date().toISOString().split('T')[0]
                    }
                }
            }));

            // Perform bulk write
            const result = await Model.bulkWrite(bulkOps, { session });

            // Commit transaction
            await session.commitTransaction();
            session.endSession();

            // Remove temporary file
            fs.unlinkSync(req.file.path);

            res.status(200).json({ 
                message: `${type} data imported successfully`, 
                count: jsonArray.length,
                insertedCount: result.insertedCount
            });
        } catch (bulkWriteError) {
            // Abort transaction
            await session.abortTransaction();
            session.endSession();

            // Remove temporary file
            fs.unlinkSync(req.file.path);

            // Log detailed error
            console.error('Bulk Write Error:', bulkWriteError);
            res.status(500).json({ 
                message: 'Failed to import data', 
                error: bulkWriteError.message 
            });
        }
    } catch (error) {
        // Remove temporary file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        console.error('Import error:', error);
        res.status(500).json({ 
            message: 'Failed to import data', 
            error: error.message 
        });
    }
};

// Export Controller
export const exportData = async (req, res) => {
    const { type } = req.params;

    try {
        // Get appropriate model
        const Model = getModelByType(type);

        // Fetch all non-deleted documents
        const data = await Model.find({ deleted: { $ne: true } });

        // Convert to plain JavaScript objects to remove Mongoose metadata
        const jsonData = data.map(doc => doc.toObject());

        // Determine fields dynamically based on the first document
        const fields = jsonData.length > 0 
            ? Object.keys(jsonData[0]).filter(key => 
                !['_id', '__v', 'creator', 'deleted', 'deletedAt'].includes(key)
            )
            : [];

        // Validate data export
        if (jsonData.length === 0) {
            return res.status(404).json({ 
                message: 'No data available', 
                error: `No ${type} records found for export` 
            });
        }

        // Create CSV parser
        const json2csvParser = new Parser({ fields });
        const csvData = json2csvParser.parse(jsonData);

        // Set headers for file download
        res.setHeader('Content-Type', 'text/csv');
        res.setHeader('Content-Disposition', `attachment; filename=${type.toLowerCase()}_export.csv`);
        
        // Send CSV data
        res.status(200).send(csvData);
    } catch (error) {
        console.error('Export error:', error);
        res.status(500).json({ 
            message: 'Failed to export data', 
            error: error.message 
        });
    }
};