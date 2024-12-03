// server/controllers/clientPortal.controller.js
import mongoose from 'mongoose';
import * as dotenv from 'dotenv';
import Deployment from '../mongodb/models/deployment.js';

dotenv.config();

export const searchByTrackCode = async (req, res) => {
    const session = await mongoose.startSession();
    session.startTransaction();

    try {
        const { trackCode } = req.query;
        
        // Input validation
        if (!trackCode || typeof trackCode !== 'string') {
            return res.status(400).json({
                success: false,
                message: 'Please provide a valid tracking code'
            });
        }

        // Query MongoDB using Deployment model
        const deployment = await Deployment
            .findOne({ trackCode })
            .select('seq date clientName vehicleModel arrivalDate parts releaseStatus releaseDate repairStatus repairedDate trackCode')
            .populate({
                path: 'parts.part',
                model: 'Part',
                select: 'partName brandName'
            })
            .lean();
        
        // Handle no results
        if (!deployment) {
            return res.status(404).json({
                success: false,
                message: 'No vehicle found with that tracking code'
            });
        }

        // Transform the parts data
        const transformedParts = deployment.parts.map(partItem => ({
            partName: partItem.part?.partName || 'Unknown Part',
            brandName: partItem.part?.brandName || 'Unknown Brand',
            quantityUsed: partItem.quantityUsed
        }));

        // Transform the data to match the frontend interface
        const transformedData = {
            seq: deployment.seq,
            date: deployment.date,
            clientName: deployment.clientName,
            vehicleModel: deployment.vehicleModel,
            arrivalDate: deployment.arrivalDate,
            parts: transformedParts,
            releaseStatus: deployment.releaseStatus,
            releaseDate: deployment.releaseDate,
            repairStatus: deployment.repairStatus || 'Pending',
            repairedDate: deployment.repairedDate,
            trackCode: deployment.trackCode
        };

        // Send successful response
        return res.status(200).json({
            success: true,
            data: transformedData
        });

    } catch (error) {
        console.error('Search error:', error);
        return res.status(500).json({
            success: false,
            message: 'Error searching for vehicle',
            error: process.env.NODE_ENV === 'development' ? error.message : undefined
        });
    } finally {
        session.endSession();
    }
};