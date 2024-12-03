// server/seeds/seedExpenses.js
import mongoose from 'mongoose';
import * as dotenv from 'dotenv';
import expenseModel from './mongodb/models/expense.js';
import User from './mongodb/models/user.js'
import connectDB from './mongodb/connect.js';

dotenv.config();

// Helper function to generate random date (2020-2024)
const generateRandomDate = (startYear, endYear) => {
  const start = new Date(`${startYear}-01-01`).getTime();
  const end = new Date(`${endYear}-12-31`).getTime();
  return new Date(start + Math.random() * (end - start)).toISOString().split('T')[0];
};

// Helper function to generate random TIN (12-digit string)
const generateTIN = () => Array(12).fill(0).map(() => Math.floor(Math.random() * 10)).join('');

// Helper function to generate random address
const generateAddress = () => {
  const streets = ['Makati Ave', 'EDSA', 'Quezon Ave', 'Ortigas Ave', 'Shaw Blvd'];
  const cities = ['Makati', 'Quezon City', 'Pasig', 'Mandaluyong', 'Taguig'];
  return `${Math.floor(Math.random() * 1000) + 1} ${streets[Math.floor(Math.random() * streets.length)]}, ${cities[Math.floor(Math.random() * cities.length)]}`;
};

// Helper function to generate random amount and VAT calculations
const generateAmountDetails = () => {
  const amount = Math.floor(Math.random() * 50000) + 5000; // Between 5,000 and 50,000
  const isNonVat = Math.random() > 0.8;
  const netOfVAT = isNonVat ? amount : amount / 1.12;
  const inputVAT = isNonVat ? 0 : amount - netOfVAT;
  return {
    amount,
    netOfVAT: parseFloat(netOfVAT.toFixed(2)),
    inputVAT: parseFloat(inputVAT.toFixed(2)),
    isNonVat,
  };
};

// Seed function
async function seedExpenses() {
  try {
    await connectDB(process.env.MONGODB_URL);
    console.log('Connected to MongoDB');

    // Get a user for creator reference
    const user = await User.findOne();
    if (!user) {
      console.log('Please ensure at least one user exists in the database');
      process.exit(1);
    }

    // Clear existing expense data
    await expenseModel.deleteMany({});
    console.log('Cleared existing expenses');

    // Generate expenses
    const expenses = [];
    for (let i = 0; i < 1000; i++) {
      const { amount, netOfVAT, inputVAT, isNonVat } = generateAmountDetails();

      expenses.push({
        seq: i + 1,
        date: generateRandomDate(2020, 2024),
        supplierName: `Supplier-${Math.floor(Math.random() * 100) + 1}`,
        ref: `EXP-${Math.floor(Math.random() * 100000)}`,
        tin: generateTIN(),
        address: generateAddress(),
        description: ['Office Supplies', 'Utilities', 'Rent', 'Equipment', 'Services'][Math.floor(Math.random() * 5)],
        amount,
        netOfVAT,
        inputVAT,
        isNonVat,
        noValidReceipt: Math.random() > 0.9,
        creator: user._id,
        deleted: false,
        deletedAt: null,
      });
    }

    // Insert expenses into the database
    await expenseModel.insertMany(expenses);
    console.log('Seeded 1000 expense records successfully');

    process.exit(0);
  } catch (error) {
    console.error('Error seeding expenses:', error);
    process.exit(1);
  }
}

seedExpenses();
