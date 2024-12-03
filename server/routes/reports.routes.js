// server/routes/reportRoutes.js
import express from 'express';
import { generateReport } from '../controller/reports.controller.js';

const router = express.Router();

router.get('/generate', generateReport);

export default router;
