import mongoose from 'mongoose';
import dotenv from 'dotenv';
import Sale from './mongodb/models/sale.js';
import User from './mongodb/models/user.js';
import connectDB from './mongodb/connect.js';

dotenv.config();

// Helper function to generate random date (within a specific range)
const getRandomDate = () => {
  const startDate = new Date('2020-01-01').getTime(); // Start date: January 1, 2020
  const endDate = new Date('2024-12-31').getTime(); // End date: December 31, 2024
  const randomDate = new Date(startDate + Math.random() * (endDate - startDate));

  // Format the randomDate to 'YYYY-MM-DD'
  const year = randomDate.getFullYear();
  const month = String(randomDate.getMonth() + 1).padStart(2, '0'); // Month is 0-indexed
  const day = String(randomDate.getDate()).padStart(2, '0');

  return `${year}-${month}-${day}`; // Return in 'YYYY-MM-DD' format
};

// Helper function to generate TIN (Taxpayer Identification Number)
const generateTIN = () => {
  return `${Math.floor(100000000 + Math.random() * 900000000)}`;
};

// Helper function to generate client name
const generateClientName = () => {
  const prefixes = ['Local', 'Small', 'Community', 'Neighborhood', 'Family', 'Personal'];
  const suffixes = ['Business', 'Store', 'Shop', 'Enterprise', 'Services', 'Outlet'];
  
  return `${prefixes[Math.floor(Math.random() * prefixes.length)]} ${suffixes[Math.floor(Math.random() * suffixes.length)]}`;
};

// Seed function
async function seedSales() {
  try {
    await connectDB(process.env.MONGODB_URL);
    console.log('Connected to MongoDB');

    // Get a user for creator reference with more robust fetching
    let user = await User.findOne();
    
    // If no user exists, create a default user
    if (!user) {
      console.log('No existing user found. Creating a default user...');
      user = new User({
        name: 'Default Admin',
        email: 'admin@example.com',
        avatar: 'https://example.com/default-avatar.png', // Placeholder avatar URL
        isAllowed: true,
        isAdmin: true,
        allProperties: [],
        allProcurements: [],
        allSales: [],
        allDeployments: [],
        allExpenses: []
      });
      
      await user.save();
      console.log('Default user created successfully');
    }

    // Clear existing data in Sale collection
    await Sale.deleteMany({});
    console.log('Cleared existing sales data');

    // Seed sales
    const sales = [];
    let salesSeq = 1;

    for (let i = 0; i < 2000; i++) {
      // Generate amount around 1,000 with some variation
      const baseAmount = 1000;
      const variation = Math.floor(Math.random() * 200) - 100; // -100 to +100
      const amount = baseAmount + variation;
      
      // Calculate VAT (assuming 12% VAT rate)
      const vatRate = 0.12;
      const netOfVAT = amount / (1 + vatRate);
      const outputVAT = amount - netOfVAT;

      const sale = new Sale({
        seq: salesSeq++,
        date: getRandomDate(),
        clientName: generateClientName(),
        tin: generateTIN(),
        amount: Number(amount.toFixed(2)),
        netOfVAT: Number(netOfVAT.toFixed(2)),
        outputVAT: Number(outputVAT.toFixed(2)),
        creator: user._id,
        deleted: false,
        deletedAt: null
      });

      // Add the sale to the user's allSales array
      user.allSales.push(sale._id);

      sales.push(sale);
    }

    // Save all sales and update user in a single operation
    await Sale.insertMany(sales);
    await user.save();

    console.log(`Created ${sales.length} sales records`);
    console.log('Sales seeding completed successfully!');
    process.exit(0);
  } catch (error) {
    console.error('Error seeding sales data:', error);
    process.exit(1);
  }
}

seedSales();