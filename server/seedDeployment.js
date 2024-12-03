import mongoose from 'mongoose';
import dotenv from 'dotenv';
import Deployment from './mongodb/models/deployment.js';
import Part from './mongodb/models/part.js';
import User from './mongodb/models/user.js';
import connectDB from './mongodb/connect.js';

dotenv.config();

// Helper function to generate random boolean values
const getRandomBoolean = () => Math.random() > 0.5;

// Helper function to generate random date within a specific range
const getRandomDate = (startDate, endDate) => {
  const start = new Date(startDate).getTime();
  const end = new Date(endDate).getTime();
  const randomDate = new Date(start + Math.random() * (end - start));

  // Format the randomDate to 'YYYY-MM-DD'
  const year = randomDate.getFullYear();
  const month = String(randomDate.getMonth() + 1).padStart(2, '0');
  const day = String(randomDate.getDate()).padStart(2, '0');

  return `${year}-${month}-${day}`;
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

    for (let i = 0; i < 1000; i++) {
      // Determine parts availability
      const partsAvailable = Math.random() > 0.3;
      const numParts = partsAvailable ? Math.floor(Math.random() * 3) + 1 : 0;
      const selectedParts = partsAvailable
        ? createdParts
            .sort(() => 0.5 - Math.random())
            .slice(0, numParts)
            .map(part => ({
              part: part._id,
              quantityUsed: Math.floor(Math.random() * 5) + 1,
            }))
        : [];

      // Generate arrival date within 2020-2024
      const arrivalYear = 2020 + Math.floor(Math.random() * 5);
      const arrivalMonth = Math.floor(Math.random() * 12) + 1;
      const arrivalDay = Math.floor(Math.random() * 28) + 1;
      const arrivalDate = `${arrivalYear}-${String(arrivalMonth).padStart(2, '0')}-${String(arrivalDay).padStart(2, '0')}`;

      // Determine deployment and repair status based on the index
      let deploymentDate = null;
      let repairStatus = "Pending";
      let repairedDate = null;
      let releaseDate = null;
      let releaseStatus = false;

      if (i < 800) {
        // First 800 deployments: mostly released
        if (Math.random() > 0.1) { // 90% chance of being released
          // 2-3 days after arrival for deployment date
          const deploymentDateObj = new Date(new Date(arrivalDate));
          deploymentDateObj.setDate(deploymentDateObj.getDate() + (Math.random() > 0.5 ? 2 : 3));
          deploymentDate = getRandomDate(arrivalDate, deploymentDateObj);

          // High chance of being repaired
          if (Math.random() > 0.1) {
            repairStatus = "Repaired";
            
            // Repaired date after deployment date
            const repairedDateObj = new Date(deploymentDate);
            repairedDateObj.setDate(repairedDateObj.getDate() + Math.floor(Math.random() * 10) + 5);
            repairedDate = getRandomDate(deploymentDate, repairedDateObj);

            // Release date 10-60 days after arrival
            const releaseDateObj = new Date(arrivalDate);
            releaseDateObj.setDate(releaseDateObj.getDate() + Math.floor(Math.random() * 51) + 10);
            releaseDate = getRandomDate(arrivalDate, releaseDateObj);
            releaseStatus = true;
          } else {
            repairStatus = "In Progress";
          }
        } else {
          repairStatus = "Cancelled";
        }
      } else {
        // Last 200 deployments: mostly repaired, some pending/in progress
        if (Math.random() > 0.2) { // 80% chance of being repaired
          // 2-3 days after arrival for deployment date
          const deploymentDateObj = new Date(new Date(arrivalDate));
          deploymentDateObj.setDate(deploymentDateObj.getDate() + (Math.random() > 0.5 ? 2 : 3));
          deploymentDate = getRandomDate(arrivalDate, deploymentDateObj);

          repairStatus = "Repaired";
          
          // Repaired date after deployment date
          const repairedDateObj = new Date(deploymentDate);
          repairedDateObj.setDate(repairedDateObj.getDate() + Math.floor(Math.random() * 10) + 5);
          repairedDate = getRandomDate(deploymentDate, repairedDateObj);

          // Release date 10-60 days after arrival
          const releaseDateObj = new Date(arrivalDate);
          releaseDateObj.setDate(releaseDateObj.getDate() + Math.floor(Math.random() * 51) + 10);
          releaseDate = getRandomDate(arrivalDate, releaseDateObj);
          releaseStatus = true;
        } else {
          // Remaining are split between pending and in progress
          repairStatus = Math.random() > 0.5 ? "Pending" : "In Progress";
          
          // Still add a deployment date 2-3 days after arrival
          const deploymentDateObj = new Date(new Date(arrivalDate));
          deploymentDateObj.setDate(deploymentDateObj.getDate() + (Math.random() > 0.5 ? 2 : 3));
          deploymentDate = getRandomDate(arrivalDate, deploymentDateObj);
        }
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
        releaseStatus,
        releaseDate,
        creator: user._id,
        repairStatus,
        repairedDate,
        trackCode: generateTrackCode(),
        deleted: false,
        deletedAt: null,
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