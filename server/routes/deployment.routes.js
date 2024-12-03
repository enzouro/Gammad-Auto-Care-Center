//server\routes\deployment.routes.js
import express from 'express';
import {
  getAllDeployments,
  getDeploymentDetail,
  createDeployment,
  updateDeployment,
  deleteDeployment,
  restoreDeployment,
} from '../controller/deployment.controller.js';

const router = express.Router();

router.get('/', getAllDeployments);
router.get('/:id', getDeploymentDetail);
router.post('/', createDeployment);
router.patch('/:id', updateDeployment);
router.delete('/:id', deleteDeployment);
router.patch('/:id/restore', restoreDeployment);

export default router;