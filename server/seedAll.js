import mongoose from 'mongoose';
import dotenv from 'dotenv';
import Deployment from './mongodb/models/deployment.js';
import Part from './mongodb/models/part.js';
import User from './mongodb/models/user.js';
import connectDB from './mongodb/connect.js';

dotenv.config();

// Helper function to generate random boolean values
const getRandomBoolean = () => Math.random() > 0.5;

// Helper function to generate random date (within a specific range)
const getRandomDate = () => {
  const startDate = new Date('2022-01-01').getTime(); // Start date: January 1, 2022
  const endDate = new Date('2024-12-31').getTime(); // End date: December 31, 2023
  const randomDate = new Date(startDate + Math.random() * (endDate - startDate));

  // Format the randomDate to 'YYYY-MM-DD'
  const year = randomDate.getFullYear();
  const month = String(randomDate.getMonth() + 1).padStart(2, '0'); // Month is 0-indexed
  const day = String(randomDate.getDate()).padStart(2, '0');

  return `${year}-${month}-${day}`; // Return in 'YYYY-MM-DD' format
};

// Helper function to generate a track code
const generateTrackCode = () => {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  return Array(8).fill(0).map(() => chars[Math.floor(Math.random() * chars.length)]).join('');
};

// Seed function
async function seedDeployment() {
  try {
    await connectDB(process.env.MONGODB_URL);
    console.log('Connected to MongoDB');

    // Get a user for creator reference
    const user = await User.findOne();
    if (!user) {
      console.log('Please ensure at least one user exists in the database');
      process.exit(1);
    }

    // Fetch existing parts
    const createdParts = await Part.find();
    if (createdParts.length === 0) {
      console.log('No parts available in the database. Please seed parts first.');
      process.exit(1);
    }

    // Clear existing data in Deployment collection
    await Deployment.deleteMany({});
    console.log('Cleared existing deployment data');

    // Seed deployments
    const deployments = [];
    let deploymentSeq = 1;

    const startDate = new Date(); // Starting point: today's date
    const endDate = new Date(startDate);
    endDate.setMonth(endDate.getMonth() + 24); // 24 months from today

    for (let i = 0; i < 1000; i++) {
      // Determine if parts are available for deployment
      const partsAvailable = Math.random() > 0.3; // 70% chance parts are available
      const numParts = partsAvailable ? Math.floor(Math.random() * 3) + 1 : 0;
      const selectedParts = partsAvailable
        ? createdParts
            .sort(() => 0.5 - Math.random())
            .slice(0, numParts)
            .map(part => ({
              part: part._id,
              quantityUsed: Math.floor(Math.random() * 5) + 1, // Random quantity used between 1 and 5
            }))
        : [];

      // Generate random dates for the deployment process
      const arrivalDate = getRandomDate(startDate, endDate); // Random arrival date within the 24-month window
      const deploymentDate = Math.random() > 0.5
        ? getRandomDate(new Date(arrivalDate), new Date(arrivalDate).setDate(new Date(arrivalDate).getDate() + 7))
        : null;

      // Define the repair status and related dates
      let repairStatus = "Pending";
      let repairedDate = null;
      let releaseDate = null;

      if (deploymentDate) {
        repairStatus = Math.random() > 0.5 ? "In progress" : "Pending";

        if (Math.random() > 0.1) { // 90% chance of progressing to repaired
          repairedDate = getRandomDate(new Date(deploymentDate), new Date(deploymentDate).setDate(new Date(deploymentDate).getDate() + 14));
          repairStatus = "Repaired";
          releaseDate = getRandomDate(new Date(repairedDate), new Date(repairedDate).setDate(new Date(repairedDate).getDate() + 14));
        } else {
          repairStatus = "Cancelled";
        }
      } else {
        repairStatus = Math.random() > 0.7 ? "Cancelled" : "Pending"; // More likely to stay Pending if no deployment date
      }

      // Push the deployment record
      deployments.push({
        seq: deploymentSeq++,
        date: arrivalDate,
        clientName: `Client-${Math.floor(Math.random() * 100)}`,
        vehicleModel: `Vehicle-${Math.floor(Math.random() * 100)}`,
        arrivalDate,
        parts: selectedParts,
        deploymentStatus: deploymentDate !== null,
        deploymentDate,
        releaseStatus: releaseDate !== null,
        releaseDate,
        creator: user._id, // Referencing the existing user
        repairStatus,
        repairedDate,
        trackCode: generateTrackCode(), // Generate a unique track code
        deleted: false, // Default to false
        deletedAt: null, // Default to null
      });
    }

    const createdDeployments = await Deployment.insertMany(deployments);
    console.log(`Created ${createdDeployments.length} deployments`);
    console.log('Deployment seeding completed successfully!');
    process.exit(0);
  } catch (error) {
    console.error('Error seeding deployment data:', error);
    process.exit(1);
  }
}

seedDeployment();
