//server\controller\procurement.controller.js
import * as dotenv from 'dotenv';
import mongoose from 'mongoose';
import Procurement from '../mongodb/models/procurement.js';
import Deployment from '../mongodb/models/deployment.js';
import User from '../mongodb/models/user.js';
import Part from '../mongodb/models/part.js';

dotenv.config();

const getAllProcurements = async (req, res) => {
  const {
    _end, _order, _start, _sort, supplierName_like = '',
  } = req.query;

  const query = {};

  if (supplierName_like) {
    query.supplierName = { $regex: supplierName_like, $options: 'i' };
  }

  try {
    const count = await Procurement.countDocuments(query);

    const procurements = await Procurement
      .find(query)
      .limit(parseInt(_end) - parseInt(_start))
      .skip(parseInt(_start))
      .sort({ [_sort]: _order })
      .populate('part', 'partName brandName');

    res.header('x-total-count', count);
    res.header('Access-Control-Expose-Headers', 'x-total-count');
    res.status(200).json(procurements);
  } catch (err) {
    console.error('Error fetching procurements:', err.message);
    res.status(500).json({ message: 'Fetching procurements failed, please try again later.' });
  }
};

const getProcurementDetail = async (req, res) => {
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid ID format.' });
    }

    const procurement = await Procurement.findById(id)
      .populate('creator')
      .populate('part')
      .session(session);

    if (!procurement) {
      return res.status(404).json({ message: 'Procurement not found.' });
    }

    const response = {
      ...procurement.toObject(),
      partName: procurement.part ? procurement.part.partName : null,
      brandName: procurement.part ? procurement.part.brandName : null,
    };
    res.status(200).json(response);
  } catch (err) {
    console.error('Error fetching procurement detail:', err);
    res.status(500).json({ message: 'Failed to get the procurement details. Please try again later.' });
  } finally {
    session.endSession();
  }
};

const createProcurement = async (req, res) => {
  const {
    email, seq, date, supplierName, reference, tin, address, partName, brandName, description,
    quantityBought, amount, netOfVAT, inputVAT, isNonVat, noValidReceipt
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const user = await User.findOne({ email }).session(session);
    if (!user) throw new Error('User not found.');

    let part = await Part.findOne({ partName, brandName }).session(session);

    if (!part) {
      part = new Part({
        partName,
        brandName,
        qtyLeft: parseInt(quantityBought),
        procurements: []
      });
    } else {
      part.qtyLeft += parseInt(quantityBought);
    }
    await part.save({ session });

    let calculatedNetOfVAT, calculatedInputVAT, finalIsNonVat = isNonVat;
    if (noValidReceipt) {
      calculatedNetOfVAT = 0;
      calculatedInputVAT = 0;
      finalIsNonVat = true;
    } else if (isNonVat) {
      calculatedNetOfVAT = amount;
      calculatedInputVAT = 0;
    } else {
      calculatedNetOfVAT = netOfVAT;
      calculatedInputVAT = inputVAT;
    }

    const newProcurement = new Procurement({
      seq,
      date,
      supplierName: noValidReceipt ? "N/A" : supplierName,
      reference: noValidReceipt ? "N/A" : reference,
      tin: noValidReceipt ? "N/A" : tin,
      address: noValidReceipt ? "N/A" : address,
      part: part._id,
      description,
      quantityBought: parseInt(quantityBought),
      amount,
      netOfVAT: calculatedNetOfVAT,
      inputVAT: calculatedInputVAT,
      isNonVat: finalIsNonVat,
      noValidReceipt,
      creator: user._id,
    });

    await newProcurement.save({ session });
    part.procurements.push(newProcurement._id);
    await part.save({ session });
    user.allProcurements.push(newProcurement._id);
    await user.save({ session });

    await session.commitTransaction();
    res.status(201).json({ message: 'Procurement created successfully', procurement: newProcurement });
  } catch (error) {
    await session.abortTransaction();
    console.error('Error creating procurement:', error);
    res.status(500).json({ message: 'Failed to create procurement.', error: error.message });
  } finally {
    session.endSession();
  }
};

const updateProcurement = async (req, res) => {
  const { id } = req.params;
  const {
    seq, date, supplierName, reference, tin, address, description, partName, brandName,
    quantityBought, amount, netOfVAT, inputVAT, isNonVat, noValidReceipt
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const procurement = await Procurement.findById(id).populate('part').session(session);
    if (!procurement) return res.status(404).json({ message: 'Procurement not found.' });

    const oldPart = procurement.part;
    oldPart.qtyLeft -= procurement.quantityBought;
    await oldPart.save({ session });

    let newPart = await Part.findOne({ partName, brandName }).session(session);
    if (!newPart) {
      newPart = new Part({
        partName,
        brandName,
        qtyLeft: parseInt(quantityBought),
        procurements: []
      });
    } else {
      newPart.qtyLeft += parseInt(quantityBought);
    }
    await newPart.save({ session });

    let calculatedNetOfVAT, calculatedInputVAT, finalIsNonVat = isNonVat;
    if (noValidReceipt) {
      calculatedNetOfVAT = 0;
      calculatedInputVAT = 0;
      finalIsNonVat = true;
    } else if (isNonVat) {
      calculatedNetOfVAT = amount;
      calculatedInputVAT = 0;
    } else {
      calculatedNetOfVAT = netOfVAT;
      calculatedInputVAT = inputVAT;
    }

    procurement.set({
      seq, date, description, amount, isNonVat: finalIsNonVat, noValidReceipt,
      netOfVAT: calculatedNetOfVAT, inputVAT: calculatedInputVAT,
      supplierName: noValidReceipt ? "N/A" : supplierName,
      reference: noValidReceipt ? "N/A" : reference,
      tin: noValidReceipt ? "N/A" : tin,
      address: noValidReceipt ? "N/A" : address,
      part: newPart._id,
      quantityBought: parseInt(quantityBought)
    });

    await procurement.save({ session });
    oldPart.procurements.pull(procurement._id);
    await oldPart.save({ session });
    newPart.procurements.push(procurement._id);
    await newPart.save({ session });

    await session.commitTransaction();
    res.status(200).json({ message: 'Procurement updated successfully', procurement });
  } catch (error) {
    await session.abortTransaction();
    console.error('Error updating procurement:', error);
    res.status(500).json({ message: 'Failed to update procurement.', error: error.message });
  } finally {
    session.endSession();
  }
};

const deleteProcurement = async (req, res) => {
  const { id } = req.params;
  const session = await mongoose.startSession();
  session.startTransaction();
  try {
    const ids = id.split(',').map(id => id.trim());
    const validIds = ids.every(id => mongoose.Types.ObjectId.isValid(id));
    if (!validIds) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }

    const procurements = await Procurement.find({ _id: { $in: ids } }).populate('part').session(session);
    if (procurements.length === 0) {
      return res.status(404).json({ message: 'No procurements found to process' });
    }

    let actionCount = 0;
    for (const procurement of procurements) {
      if (procurement.deleted === false) {
        // Soft delete procurement
        procurement.deleted = true;
        procurement.deletedAt = new Date();
        await procurement.save({ session });

        if (procurement.part) {
          const part = await Part.findById(procurement.part).session(session);
          if (part) {
            part.qtyLeft -= procurement.quantityBought;

            // Check if there are any active procurements for this part
            const hasActiveProcurements = await Procurement.countDocuments({
              part: part._id,
              deleted: false,
            }).session(session) >= 1;

            // Only soft delete the part if there are no active procurements
            if (!hasActiveProcurements) {
              part.deleted = true;
              part.deletedAt = new Date();
            }
            await part.save({ session });
          }
        }
        actionCount++;
      } else if (procurement.deleted === true) {
        // Hard delete procurement
        await Procurement.findByIdAndDelete(procurement._id, { session });

        // if (procurement.part) {
        //   const part = await Part.findById(procurement.part).session(session);
        //   if (part) {
        //     // Check if there are any remaining procurements (active or deleted)
        //     const hasRemainingProcurements = await Procurement.countDocuments({
        //       part: part._id,
        //       _id: { $ne: procurement._id } // Exclude current procurement being deleted
        //     }).session(session) >= 1;

        //     // Only delete the part if there are no remaining procurements
        //     if (!hasRemainingProcurements) {
        //       await Part.findByIdAndDelete(part._id, { session });
        //     }
        //   }
        // }
        actionCount++;
      }
    }

    if (actionCount === 0) {
      return res.status(404).json({ message: 'No procurements found to delete' });
    }

    await session.commitTransaction();
    res.status(200).json({
      message: `Successfully deleted ${actionCount} procurement(s)`,
      processedCount: actionCount,
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to delete procurements', error: error.message });
  } finally {
    session.endSession();
  }
};

const restoreProcurement = async (req, res) => {
  const { id } = req.params;
  const session = await mongoose.startSession();
  session.startTransaction();
  try {
    const ids = id.split(',').map(id => id.trim());
    const validIds = ids.every(id => mongoose.Types.ObjectId.isValid(id));
    if (!validIds) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }

    const procurements = await Procurement.find({
      _id: { $in: ids },
      deleted: true,
    }).populate('part').session(session);

    if (procurements.length === 0) {
      return res.status(404).json({ message: 'No procurements found to restore' });
    }

    for (const procurement of procurements) {
      procurement.deleted = false;
      procurement.deletedAt = undefined;

      if (procurement.part) {
        const part = await Part.findById(procurement.part).session(session);
        if (part && part.deleted) { // Check if part exists AND is deleted
          part.qtyLeft += procurement.quantityBought;
          part.deleted = false;
          part.deletedAt = undefined;
          await part.save({ session });
        }
      }
      await procurement.save({ session });
    }

    await session.commitTransaction();
    res.status(200).json({
      message: `Successfully restored ${procurements.length} procurement(s)`,
      restoredCount: procurements.length,
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to restore procurements', error: error.message });
  } finally {
    session.endSession();
  }
};

export {
  getAllProcurements,
  getProcurementDetail,
  createProcurement,
  updateProcurement,
  deleteProcurement,
  restoreProcurement
};
