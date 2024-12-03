import * as dotenv from 'dotenv';
import mongoose from 'mongoose';
import Deployment from '../mongodb/models/deployment.js';
import User from '../mongodb/models/user.js';
import Part from '../mongodb/models/part.js';

dotenv.config();

const getAllDeployments = async (req, res) => {
  const {
    _end, _order, _start, _sort, searchField, searchValue = '',
  } = req.query;

  const query = {};

  // Handle search based on various fields
  if (searchValue) {
    switch (searchField) {
      case 'clientName':
      case 'vehicleModel':
        query[searchField] = { $regex: searchValue, $options: 'i' };
        break;
      case 'seq':
        // Handle numeric search for sequence
        if (!isNaN(searchValue)) {
          query.seq = parseInt(searchValue);
        }
        break;
    }
  }

  try {
    const count = await Deployment.countDocuments(query);

    const deployments = await Deployment
      .find(query)
      .limit(parseInt(_end) - parseInt(_start))
      .skip(parseInt(_start))
      .sort({ [_sort]: _order })
      .populate('parts.part'); // Populate the parts array

    res.header('x-total-count', count);
    res.header('Access-Control-Expose-Headers', 'x-total-count');

    res.status(200).json(deployments);
  } catch (err) {
    console.log(err.message);
    res.status(500).json({ message: 'Fetching deployments failed, please try again later' });
  }
};

const getDeploymentDetail = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }

    const deploymentExists = await Deployment.findOne({ _id: id })
      .populate('creator')
      .populate({
        path: 'parts.part',
        select: 'partName brandName qtyLeft' // Include any other part fields you need
      })
      .session(session);

    if (deploymentExists) {
      const response = {
        ...deploymentExists.toObject(),
        parts: deploymentExists.parts.map(partEntry => ({
          ...partEntry.toObject(),
          partName: partEntry.part?.partName || null,
          brandName: partEntry.part?.brandName || null,
          qtyLeft: partEntry.part?.qtyLeft || 0,
        }))
      };
      res.status(200).json(response);
    } else {
      res.status(404).json({ message: 'Deployment does not exist' });
    }
  } catch (err) {
    console.error('Error fetching deployment detail:', err);
    res.status(500).json({ message: 'Failed to get the deployment details, please try again later' });
  } finally {
    session.endSession();
  }
};

const createDeployment = async (req, res) => {
  const { 
    email,
    seq, 
    date, 
    clientName, 
    vehicleModel, 
    arrivalDate,
    parts, // Array of {part: "partName|brandName", quantityUsed: number}
    deploymentStatus,
    deploymentDate,
    releaseStatus,
    releaseDate,
    repairStatus,
    repairedDate,
    trackCode
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const user = await User.findOne({ email }).session(session);
    if (!user) throw new Error('User not found');
    // Process all parts
    const processedParts = await Promise.all(parts.map(async (partEntry) => {
      // Split the part string into partName and brandName
      const [partName, brandName] = partEntry.part.split('|');
      
      if (!partName || !brandName) {
        throw new Error('Invalid part format. Expected "partName|brandName"');
      }

      // Find part using partName and brandName
      const selectedPart = await Part.findOne({
        partName: partName,
        brandName: brandName
      }).session(session);

      if (!selectedPart) {
        throw new Error(`Part with name "${partName}" and brand "${brandName}" not found`);
      }

      // Validate quantity
      if (selectedPart.qtyLeft < partEntry.quantityUsed) {
        throw new Error(`Insufficient quantity available for ${partName}. Requested: ${partEntry.quantityUsed}, Available: ${selectedPart.qtyLeft}`);
      }

      return {
        part: selectedPart._id,
        quantityUsed: parseInt(partEntry.quantityUsed),
        selectedPart // Keep reference to update quantity later
      };
    }));

    // Create new deployment
    const newDeployment = new Deployment({
      seq,
      date,
      clientName,
      vehicleModel,
      arrivalDate,
      parts: processedParts.map(({ part, quantityUsed }) => ({
        part,
        quantityUsed
      })),
      deploymentStatus: deploymentStatus || false,
      deploymentDate,
      releaseStatus: releaseStatus || false,
      releaseDate,
      repairStatus: repairStatus,
      repairedDate,
      trackCode,
      creator: user._id,
    });

    // Save deployment
    await newDeployment.save({ session });

    // Update quantities for all parts
    await Promise.all(processedParts.map(async ({ selectedPart, quantityUsed }) => {
      selectedPart.qtyLeft -= quantityUsed;
      if (selectedPart.deployments) {
        selectedPart.deployments.push(newDeployment._id);
      }
      await selectedPart.save({ session });
    }));

    // Link deployment to user's deployments
    if (user.allDeployments) {
      user.allDeployments.push(newDeployment._id);
      await user.save({ session });
    }

    await session.commitTransaction();

    const populatedDeployment = await Deployment.findById(newDeployment._id)
      .populate('parts.part')
      .populate('creator');

    res.status(201).json({ 
      message: 'Deployment created successfully', 
      deployment: populatedDeployment 
    });
  } catch (error) {
    await session.abortTransaction();
    console.log('Error creating deployment:', error);
    res.status(500).json({ message: 'Failed to create deployment', error: error.message });
  } finally {
    session.endSession();
  }
};

const updateDeployment = async (req, res) => {
  const { id } = req.params;
  const {
    deploymentStatus,
    deploymentDate,
    releaseStatus,
    releaseDate,
    seq,
    date,
    clientName,
    vehicleModel,
    arrivalDate,
    parts,
    repairStatus,
    repairedDate,
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    // Find the existing deployment
    const deployment = await Deployment.findById(id)
      .populate('parts.part')
      .session(session);
    if (!deployment) {
      return res.status(404).json({ message: 'Deployment not found' });
    }

    // Check if only deployment and release statuses and dates are being updated
    const isStatusUpdateOnly = !(
      seq || date || clientName || vehicleModel || arrivalDate || parts
    );

    if (isStatusUpdateOnly) {
      // Update deployment status and date if provided
      if (deploymentStatus !== undefined) {
        deployment.deploymentStatus = deploymentStatus;
        deployment.deploymentDate = deploymentDate;

        // If deployment status is set to false, also set release status to false
        // if (deployment.deploymentStatus === false) {
        //   deployment.releaseStatus = false;
        //   deployment.releaseDate = null;
        // }
      }

      // Allow release status and date update only if deployment status is true
      if ( releaseStatus !== undefined) {
        deployment.releaseStatus = releaseStatus;
        deployment.releaseDate = releaseDate;
      } else if (releaseStatus !== undefined) {
        return res.status(400).json({ 
          message: 'Release status cannot be updated until deployment is complete.' 
        });
      }

        if (repairStatus !== 'Repaired'){
          deployment.repairStatus = repairStatus;
          deployment.repairedDate = null;
        }
        else{
          deployment.repairStatus = repairStatus;
          deployment.repairedDate = repairedDate;
        }

      await deployment.save({ session });
      await session.commitTransaction();

      res.status(200).json({
        message: 'Deployment status and dates updated successfully',
        deployment
      });
      return;
    }

    // Handle parts updates
    if (parts) {
      // Create a map of existing parts for easier lookup
      const existingPartsMap = new Map(
        deployment.parts.map(p => [p.part._id.toString(), p])
      );
      // Process each new part
      for (const newPartData of parts) {
        const [partName, brandName] = newPartData.part.split('|');
        if (!partName || !brandName) {
          return res.status(400).json({ 
            message: 'Invalid part format. Expected "partName|brandName"' 
          });
        }

        // Find the part in database
        const partDoc = await Part.findOne({
          partName: partName,
          brandName: brandName
        }).session(session);

        if (!partDoc) {
          return res.status(404).json({ 
            message: `Part with name "${partName}" and brand "${brandName}" not found` 
          });
        }

        const partId = partDoc._id.toString();
        const existingPart = existingPartsMap.get(partId);
        const newQuantity = parseInt(newPartData.quantityUsed);
        if (existingPart) {
          // Update existing part quantity
          const quantityDifference = newQuantity - existingPart.quantityUsed;
          if (partDoc.qtyLeft < quantityDifference) {
            return res.status(400).json({
              message: `Insufficient quantity available for ${partName}. Need ${quantityDifference} more, but only ${partDoc.qtyLeft} available`
            });
          }
          partDoc.qtyLeft -= quantityDifference;
          existingPartsMap.delete(partId);
        } else {
          // Add new part
          if (partDoc.qtyLeft < newQuantity) {
            return res.status(400).json({
              message: `Insufficient quantity available for ${partName}. Requested: ${newQuantity}, Available: ${partDoc.qtyLeft}`
            });
          }
          partDoc.qtyLeft -= newQuantity;
        }

        await partDoc.save({ session });
      }

      // Return quantities for removed parts
      for (const [, removedPart] of existingPartsMap) {
        const partDoc = await Part.findById(removedPart.part._id).session(session);
        partDoc.qtyLeft += removedPart.quantityUsed;
        await partDoc.save({ session });
      }

      // Update deployment parts array
      deployment.parts = await Promise.all(parts.map(async (partData) => {
        const [partName, brandName] = partData.part.split('|');
        const partDoc = await Part.findOne({
          partName: partName,
          brandName: brandName
        }).session(session);
        
        return {
          part: partDoc._id,
          quantityUsed: parseInt(partData.quantityUsed)
        };
      }));
    }

    // Update other deployment fields
    deployment.seq = seq;
    deployment.date = date;
    deployment.clientName = clientName;
    deployment.vehicleModel = vehicleModel;
    deployment.arrivalDate = arrivalDate;


    //Update repair status
    if (repairStatus !== 'Repaired'){
      deployment.repairStatus = repairStatus;
      deployment.repairedDate = null;
    }
    else{
      deployment.repairStatus = repairStatus;
      deployment.repairedDate = repairedDate;
     }

  // Update status fields if provided
    if (deploymentStatus !== undefined) {
      deployment.deploymentStatus = deploymentStatus;
      deployment.deploymentDate = deploymentDate;

      // if (deployment.deploymentStatus === false) {
      //   deployment.releaseStatus = false;
      //   deployment.releaseDate = null;
      // }
    }

    if (releaseStatus !== undefined) {
      deployment.releaseStatus = releaseStatus;
      deployment.releaseDate = releaseDate;
    }

    await deployment.save({ session });
    await session.commitTransaction();

    const populatedDeployment = await Deployment.findById(deployment._id)
      .populate('parts.part')
      .populate('creator');

    res.status(200).json({
      message: 'Deployment updated successfully',
      deployment: populatedDeployment
    });

  } catch (error) {
    await session.abortTransaction();
    console.error('Error updating deployment:', error);
    res.status(500).json({ 
      message: 'Failed to update deployment', 
      error: error.message 
    });
  } finally {
    session.endSession();
  }
};

const deleteDeployment = async (req, res) => {
  const { id } = req.params;
  const session = await mongoose.startSession();
  session.startTransaction();
  try {
    const ids = id.split(',').map(id => id.trim());
    const validIds = ids.every(id => mongoose.Types.ObjectId.isValid(id));
    if (!validIds) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }

    const deployments = await Deployment.find({ _id: { $in: ids } }).populate('parts.part').session(session);

    let actionCount = 0;
    for (const deployment of deployments) {
      if (deployment.deleted === false) {
        // Soft delete deployment
        deployment.deleted = true;
        deployment.deletedAt = new Date();
        await deployment.save({ session });

        if (deployment.parts) {
          for (const partEntry of deployment.parts) {
            if (partEntry.part) {
              const part = await Part.findById(partEntry.part).session(session);
              if (part) {
                part.qtyLeft += partEntry.quantityUsed;
                await part.save({ session });
              }
            }
          }
        }
        actionCount++;
      } else if (deployment.deleted === true) {
        // Hard delete deployment
        await Deployment.findByIdAndDelete(deployment._id, { session });

        if (deployment.parts) {
          for (const partEntry of deployment.parts) {
            if (partEntry.part) {
              await Part.findByIdAndUpdate(
                partEntry.part,
                { $pull: { deployments: deployment._id } },
                { session }
              );
            }
          }
        }
        actionCount++;
      }
    }

    if (actionCount === 0) {
      return res.status(404).json({ message: 'No deployments found to delete' });
    }

    await session.commitTransaction();
    res.status(200).json({
      message: `Successfully deleted ${actionCount} deployment(s)`,
      processedCount: actionCount,
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to delete deployments', error: error.message });
  } finally {
    session.endSession();
  }
};

const restoreDeployment = async (req, res) => {
  const { id } = req.params;
  const session = await mongoose.startSession();
  session.startTransaction();
  try {
    const ids = id.split(',').map(id => id.trim());
    const validIds = ids.every(id => mongoose.Types.ObjectId.isValid(id));
    if (!validIds) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }

    const deployments = await Deployment.find({
      _id: { $in: ids },
      deleted: true,
    }).populate('parts.part').session(session);

    if (deployments.length === 0) {
      return res.status(404).json({ message: 'No deployments found to restore' });
    }

    for (const deployment of deployments) {
      for (const partEntry of deployment.parts) {
        if (partEntry.part) {
          const part = await Part.findById(partEntry.part).session(session);
          if (!part || part.deleted === true) {
            return res.status(400).json({
              message: `Cannot restore deployment. Part "${part?.partName}" is deleted or no longer exists.`,
            });
          }
          part.qtyLeft -= partEntry.quantityUsed;
          await part.save({ session });
        }
      }

      deployment.deleted = false;
      deployment.deletedAt = undefined;
      await deployment.save({ session });
    }

    await session.commitTransaction();
    res.status(200).json({
      message: `Successfully restored ${deployments.length} deployment(s)`,
      restoredCount: deployments.length,
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to restore deployments', error: error.message });
  } finally {
    session.endSession();
  }
};

export {
  getAllDeployments,
  getDeploymentDetail,
  createDeployment,
  updateDeployment,
  deleteDeployment,
  restoreDeployment,
};