import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { importData, exportData } from '../controller/dataIE.controller.js';

const router = express.Router();

// Configure multer for file upload
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        // Ensure the uploads directory exists
        const uploadDir = path.join(process.cwd(), 'uploads');
        
        try {
            // Create uploads directory if it doesn't exist
            if (!fs.existsSync(uploadDir)) {
                fs.mkdirSync(uploadDir, { recursive: true });
            }
            cb(null, uploadDir);
        } catch (err) {
            console.error('Failed to create uploads directory:', err);
            cb(err, null);
        }
    },
    filename: (req, file, cb) => {
        // Generate unique filename
        cb(null, `${req.params.type}-${Date.now()}${path.extname(file.originalname)}`);
    }
});

// File filter to accept only csv and xlsx
const fileFilter = (req, file, cb) => {
    const allowedTypes = [
        'text/csv', 
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel'
    ];
    
    if (allowedTypes.includes(file.mimetype)) {
        cb(null, true);
    } else {
        cb(new Error('Invalid file type. Only CSV and Excel files are allowed.'), false);
    }
};

// Configure multer upload
const upload = multer({ 
    storage: storage,
    fileFilter: fileFilter,
    limits: { 
        fileSize: 10 * 1024 * 1024 // 10MB file size limit
    }
});

// Import route
router.post('/import/:type', 
    upload.single('file'), 
    (req, res) => {
        // Directly call import data controller
        importData(req, res);
    }
);

// Export route
router.get('/export/:type', exportData);

// Global error handler for routes
router.use((err, req, res, next) => {
    console.error('Route Error:', err);
    
    // Handle multer file upload errors
    if (err instanceof multer.MulterError) {
        return res.status(400).json({
            message: 'File upload error',
            error: err.message
        });
    }

    // Handle other unexpected errors
    res.status(500).json({
        message: 'An unexpected error occurred',
        error: err.message
    });
});

export default router;