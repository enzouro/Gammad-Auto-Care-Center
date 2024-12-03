import mongoose from 'mongoose';
import * as dotenv from 'dotenv';
import procurementModel from './mongodb/models/procurement.js';
import partModel from './mongodb/models/part.js';
import User from './mongodb/models/user.js';
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
async function seedProcurement() {
  try {
    await connectDB(process.env.MONGODB_URL);
    console.log('Connected to MongoDB');

    // Get a user for creator reference
    const user = await User.findOne();
    if (!user) {
      console.log('Please ensure at least one user exists in the database');
      process.exit(1);
    }

    // Clear existing procurement and part data
    await procurementModel.deleteMany({});
    await partModel.deleteMany({});
    console.log('Cleared existing procurement and part data');

    // Create seed parts
    const partNames = [
      'Brake Pad', 'Spark Plug', 'Oil Filter', 
      'Air Filter', 'Alternator', 'Water Pump', 
      'Timing Belt', 'Fuel Pump', 'Starter Motor', 
      'Radiator'
    ];

    const brands = [
      'Bosch', 'Denso', 'NGK', 'Mann', 
      'Mobil', 'Castrol', 'ACDelco', 'Mobil 1', 
      'Fram', 'Motorcraft'
    ];

    // Generate parts
    const parts = partNames.map((partName, index) => ({
      partName,
      brandName: brands[index],
      qtyLeft: 0,
      deleted: false,
      deletedAt: null,
      procurements: []
    }));

    const createdParts = await partModel.insertMany(parts);
    console.log('Seeded parts successfully');

    // Generate procurements
    const procurements = [];
    for (let i = 0; i < 1000; i++) {
      const part = createdParts[Math.floor(Math.random() * createdParts.length)];
      const { amount, netOfVAT, inputVAT, isNonVat } = generateAmountDetails();
      const quantityBought = Math.floor(Math.random() * 50) + 1; // 1-50 quantity

      const procurement = {
        seq: i + 1,
        date: generateRandomDate(2020, 2024),
        supplierName: `Supplier-${Math.floor(Math.random() * 100) + 1}`,
        reference: `PROC-${Math.floor(Math.random() * 100000)}`,
        tin: generateTIN(),
        address: generateAddress(),
        part: part._id,
        description: `${part.brandName} ${part.partName} Purchase`,
        quantityBought,
        amount,
        netOfVAT,
        inputVAT,
        isNonVat,
        noValidReceipt: Math.random() > 0.9,
        creator: user._id,
        deleted: false,
        deletedAt: null,
      };

      procurements.push(procurement);

      // Update part's quantity and procurements
      part.qtyLeft += quantityBought;
      part.procurements.push(procurement._id);
    }

    // Insert procurements into the database
    await procurementModel.insertMany(procurements);
    
    // Update parts with their procurements
    await Promise.all(createdParts.map(part => part.save()));

    console.log('Seeded 1000 procurement records successfully');

    process.exit(0);
  } catch (error) {
    console.error('Error seeding procurements:', error);
    process.exit(1);
  }
}

seedProcurement();