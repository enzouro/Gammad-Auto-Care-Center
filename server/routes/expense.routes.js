// server\routes\expense.routes.js
import express from 'express';

import {
  createExpense, deleteExpense, getAllExpenses, getExpenseDetail, updateExpense, restoreExpense,
} from '../controller/expense.controller.js';

const router = express.Router();

router
  .route('/')
  .get(getAllExpenses);

router
  .route('/:id')
  .get(getExpenseDetail);

router
  .route('/')
  .post(createExpense);

router
  .route('/:id')
  .patch(updateExpense);

router
  .route('/:id')
  .delete(deleteExpense);

router
  .route('/:id/restore')
  .patch(restoreExpense);

export default router;