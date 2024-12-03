// server\controller\sale.controller.js
import * as dotenv from 'dotenv';
import mongoose from 'mongoose';
import Sale from '../mongodb/models/sale.js';
import User from '../mongodb/models/user.js';

dotenv.config();

const getAllSales = async (req, res) => {
  const {
    _end, _order, _start, _sort, clientName_like = '',
  } = req.query;

  const query = {}; // Add this condition to exclude soft-deleted sales

  if (clientName_like) {
    query.clientName = { $regex: clientName_like, $options: 'i' };
  }

  try {
    const count = await Sale.countDocuments(query);

    const sales = await Sale
      .find(query)
      .limit(parseInt(_end) - parseInt(_start))
      .skip(parseInt(_start))
      .sort({ [_sort]: _order })
      .populate('creator', 'name email');

    res.header('x-total-count', count);
    res.header('Access-Control-Expose-Headers', 'x-total-count');

    res.status(200).json(sales);
  } catch (err) {
    res.status(500).json({ message: 'Fetching sales failed, please try again later' });
  }
};

const getSaleDetail = async (req, res) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }
    const sale = await Sale.findById(id).populate('creator', 'name email');

    if (!sale) {
      return res.status(404).json({ message: 'Sale does not exist' });
    }
    res.status(200).json(sale);
  } catch (err) {
    res.status(500).json({ message: 'Failed to get the sale details, please try again later' });
  }
};

const createSale = async (req, res) => {
  const {
    seq,
    date,
    clientName,
    tin,
    amount,
    netOfVAT,
    outputVAT,
    email,
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const user = await User.findOne({ email }).session(session);
    if (!user) {
      throw new Error('User not found');
    }

    // Calculate VAT values
    const calculatedNetOfVAT = amount * (100/112);
    const calculatedOutputVAT = amount * (12/112);

    const newSale = new Sale({
      seq,
      date,
      clientName: clientName?.trim() === '' ? 'N/A' : clientName,
      tin: tin?.trim() === '' ? 'N/A' : tin,
      amount,
      netOfVAT: Number(calculatedNetOfVAT.toFixed(2)),
      outputVAT: Number(calculatedOutputVAT.toFixed(2)),
      creator: user._id,
    });

    await newSale.save({ session });

    // Update user's sales list if it exists
    if (user.allSales) {
      user.allSales.push(newSale._id);
      await user.save({ session });
    }

    await session.commitTransaction();

    const populatedSale = await Sale.findById(newSale._id).populate('creator', 'name email');
    
    res.status(201).json({ message: 'Sale created successfully', sale: populatedSale });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to create sale', error: error.message });
  } finally {
    session.endSession();
  }
};

const updateSale = async (req, res) => {
  const { id } = req.params;
  const {
    seq,
    date,
    clientName,
    tin,
    amount,
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const sale = await Sale.findById(id).session(session);
    if (!sale) {
      return res.status(404).json({ message: 'Sale not found' });
    }

    // Calculate VAT values
    const calculatedNetOfVAT = amount * (100/112);
    const calculatedOutputVAT = amount * (12/112);

    // Update fields
    sale.seq = seq;
    sale.date = date;
    sale.clientName = clientName;
    sale.tin = tin;
    sale.amount = amount;
    sale.netOfVAT = Number(calculatedNetOfVAT.toFixed(2));
    sale.outputVAT = Number(calculatedOutputVAT.toFixed(2));

    await sale.save({ session });
    await session.commitTransaction();

    const populatedSale = await Sale.findById(sale._id).populate('creator', 'name email');

    res.status(200).json({ message: 'Sale updated successfully', sale: populatedSale });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to update sale', error: error.message });
  } finally {
    session.endSession();
  }
};

const deleteSale = async (req, res) => {
  const { id } = req.params;
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    // Handle multiple IDs
    const ids = id.split(',').map(id => id.trim());

    // Validate all IDs first
    const validIds = ids.every(id => mongoose.Types.ObjectId.isValid(id));
    if (!validIds) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }

    // Find sales to determine action
    const sales = await Sale.find({ _id: { $in: ids } }).session(session);

    let actionCount = 0;
    for (const sale of sales) {
      if (sale.deleted === false) {
        // Soft delete for non-deleted sales
        sale.deleted = true;
        sale.deletedAt = new Date();
        await sale.save({ session });
        actionCount++;

        // Remove sale from creator's allSales array
        if (sale.creator) {
          await User.findByIdAndUpdate(sale.creator, {
            $pull: { allSales: sale._id }
          }, { session });
        }
      } else if (sale.deleted === true) {
        // Hard delete for already soft-deleted sales
        await Sale.findByIdAndDelete(sale._id, { session });
        actionCount++;

        // Completely remove sale from creator's allSales array
        if (sale.creator) {
          await User.findByIdAndUpdate(sale.creator, {
            $pull: { allSales: sale._id }
          }, { session });
        }
      }
    }

    if (actionCount === 0) {
      return res.status(404).json({ message: 'No sales found to deleted' });
    }

    await session.commitTransaction();
    res.status(200).json({ 
      message: `Successfully deleted ${actionCount} sale(s)`,
      processedCount: actionCount
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to delete sales, please try again later', error: error.message });
  } finally {
    session.endSession();
  }
};

const restoreSale = async (req, res) => {
  const { id } = req.params;
  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    // Handle multiple IDs
    const ids = id.split(',');

    // Validate all IDs first
    const validIds = ids.every(id => mongoose.Types.ObjectId.isValid(id));
    if (!validIds) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }

    // Perform restore for all selected sales
    const updateResult = await Sale.updateMany(
      { 
        _id: { $in: ids },
        deleted: true // Only restore previously deleted sales
      },
      {
        $set: {
          deleted: false,
        },
        $unset: {
          deletedAt: 1 // Remove the deletedAt timestamp
        }
      },
      { session }
    );

    // If no sales were modified, return a 404 error
    if (updateResult.modifiedCount === 0) {
      return res.status(404).json({ message: 'No sales found to restore' });
    }

    // Re-add sales to user's allSales array if needed
    const sales = await Sale.find({ _id: { $in: ids } }).populate('creator').session(session);
    for (const sale of sales) {
      if (sale.creator) {
        if (!sale.creator.allSales) {
          sale.creator.allSales = [];
        }
        if (!sale.creator.allSales.includes(sale._id)) {
          sale.creator.allSales.push(sale._id);
          await sale.creator.save({ session });
        }
      }
    }

    await session.commitTransaction();
    res.status(200).json({ 
      message: `Successfully restored ${updateResult.modifiedCount} sale(s)`,
      restoredCount: updateResult.modifiedCount
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to restore sales, please try again later', error: error.message });
  } finally {
    session.endSession();
  }
};

export {
  getAllSales,
  getSaleDetail,
  createSale,
  updateSale,
  deleteSale,
  restoreSale,
};