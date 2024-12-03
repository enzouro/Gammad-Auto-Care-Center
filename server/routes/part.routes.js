import express from 'express';

import {
  createPart, deletePart, getAllParts, updatePart,
} from '../controller/part.controller.js';

const router = express.Router();

router
  .route('/')
  .get(getAllParts);

router
  .route('/')
  .post(createPart);

router
  .route('/:id')
  .patch(updatePart);

router
  .route('/:id')
  .delete(deletePart);

export default router;
