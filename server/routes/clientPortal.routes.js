// server\routes\clientPortal.routes.js
import express from 'express';
import { searchByTrackCode } from '../controller/clientPortal.controller.js';

const router = express.Router();

router.get('/search', searchByTrackCode);

export default router;