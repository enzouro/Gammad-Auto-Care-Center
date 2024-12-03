import express from 'express';

import {
  createUser,
  getAllUsers,
  getUserInfoByID
} from '../controller/user.controller.js';

const router = express.Router();

// Combined the GET and POST methods for '/'
router.route('/')
  .get(getAllUsers)  // Handles fetching all users or users by query (email, etc.)
  .post(createUser); // Handles creating a new user

// Route for getting a user by ID
router.route('/:id')
  .get(getUserInfoByID); // Handles fetching user info by their ID

export default router;
