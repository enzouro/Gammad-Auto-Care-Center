import * as dotenv from 'dotenv';
import mongoose from 'mongoose';
import Expense from '../mongodb/models/expense.js';
import User from '../mongodb/models/user.js';

dotenv.config();

const getAllExpenses = async (req, res) => {
  const {
    _end, _order, _start, _sort, supplierName_like = '',
  } = req.query;

  const query = {};

  if (supplierName_like) {
    query.supplierName = { $regex: supplierName_like, $options: 'i' };
  }

  try {
    const count = await Expense.countDocuments(query);

    const expenses = await Expense
      .find(query)
      .limit(parseInt(_end) - parseInt(_start))
      .skip(parseInt(_start))
      .sort({ [_sort]: _order })
      .populate('creator', 'name email');

    res.header('x-total-count', count);
    res.header('Access-Control-Expose-Headers', 'x-total-count');

    res.status(200).json(expenses);
  } catch (err) {
    res.status(500).json({ message: 'Fetching expenses failed, please try again later' });
  }
};

const getExpenseDetail = async (req, res) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ message: 'Invalid ID format' });
    }
    const expense = await Expense.findById(id).populate('creator', 'name email');

    if (!expense) {
      return res.status(404).json({ message: 'Expense does not exist' });
    }
    res.status(200).json(expense);
  } catch (err) {
    res.status(500).json({ message: 'Failed to get the expense details, please try again later' });
  }
};

const createExpense = async (req, res) => {
  const {
    seq,
    date,
    supplierName,
    ref,
    tin,
    address,
    description,
    amount,
    netOfVAT,
    inputVAT,
    isNonVat,
    noValidReceipt,
    email,
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const user = await User.findOne({ email }).session(session);
    if (!user) {
      throw new Error('User not found');
    }

    // Calculate VAT values based on conditions
    let calculatedNetOfVAT;
    let calculatedInputVAT = 0;
    let finalIsNonVat = isNonVat;

    if (noValidReceipt) {
      // When no valid receipt, force non-VAT to true and set net to 0
      calculatedNetOfVAT = 0;
      calculatedInputVAT = 0;
      finalIsNonVat = true;
    } else if (isNonVat) {
      // When non-VAT but has valid receipt, set net to amount
      calculatedNetOfVAT = amount;
      calculatedInputVAT = 0;
    } else {
      // Normal VAT calculation
      calculatedNetOfVAT = netOfVAT;
      calculatedInputVAT = inputVAT;
    }

    const newExpense = new Expense({
      seq,
      date,
      supplierName: noValidReceipt ? "N/A" : supplierName,
      ref: noValidReceipt ? "N/A" : ref,
      tin: noValidReceipt ? "N/A" : tin,
      address: noValidReceipt ? "N/A" : address,
      description,
      amount,
      netOfVAT: calculatedNetOfVAT,
      inputVAT: calculatedInputVAT,
      isNonVat: finalIsNonVat,
      noValidReceipt,
      creator: user._id,
    });

    await newExpense.save({ session });

    if (user.allExpenses) {
      user.allExpenses.push(newExpense._id);
      await user.save({ session });
    }

    await session.commitTransaction();

    const populatedExpense = await Expense.findById(newExpense._id).populate('creator', 'name email');
    
    res.status(201).json({ message: 'Expense created successfully', expense: populatedExpense });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to create expense', error: error.message });
  } finally {
    session.endSession();
  }
};

const updateExpense = async (req, res) => {
  const { id } = req.params;
  const {
    seq,
    date,
    supplierName,
    ref,
    tin,
    address,
    description,
    amount,
    netOfVAT,
    inputVAT,
    isNonVat,
    noValidReceipt,
  } = req.body;

  const session = await mongoose.startSession();
  session.startTransaction();

  try {
    const expense = await Expense.findById(id).session(session);
    if (!expense) {
      return res.status(404).json({ message: 'Expense not found' });
    }

    // Calculate VAT values based on conditions
    let calculatedNetOfVAT;
    let calculatedInputVAT = 0;
    let finalIsNonVat = isNonVat;

    if (noValidReceipt) {
      // When no valid receipt, force non-VAT to true and set net to 0
      calculatedNetOfVAT = 0;
      calculatedInputVAT = 0;
      finalIsNonVat = true;
    } else if (isNonVat) {
      // When non-VAT but has valid receipt, set net to amount
      calculatedNetOfVAT = amount;
      calculatedInputVAT = 0;
    } else {
      // Normal VAT calculation
      calculatedNetOfVAT = netOfVAT;
      calculatedInputVAT = inputVAT;
    }

    // Update basic fields
    expense.seq = seq;
    expense.date = date;
    expense.description = description;
    expense.amount = amount;
    expense.isNonVat = finalIsNonVat;
    expense.noValidReceipt = noValidReceipt;
    expense.netOfVAT = calculatedNetOfVAT;
    expense.inputVAT = calculatedInputVAT;

    // Update supplier information based on noValidReceipt
    if (noValidReceipt) {
      expense.supplierName = "N/A";
      expense.ref = "N/A";
      expense.tin = "N/A";
      expense.address = "N/A";
    } else {
      expense.supplierName = supplierName;
      expense.ref = ref;
      expense.tin = tin;
      expense.address = address;
    }

    await expense.save({ session });
    await session.commitTransaction();

    const populatedExpense = await Expense.findById(expense._id).populate('creator', 'name email');

    res.status(200).json({ message: 'Expense updated successfully', expense: populatedExpense });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to update expense', error: error.message });
  } finally {
    session.endSession();
  }
};

const deleteExpense = async (req, res) => {
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

    // Find expenses to determine action
    const expenses = await Expense.find({ _id: { $in: ids } }).session(session);

    let actionCount = 0;
    for (const expense of expenses) {
      if (expense.deleted === false) {
        // Soft delete for non-deleted expenses
        expense.deleted = true;
        expense.deletedAt = new Date();
        await expense.save({ session });
        actionCount++;

        // Remove expense from creator's allExpenses array
        if (expense.creator) {
          await User.findByIdAndUpdate(expense.creator, {
            $pull: { allExpenses: expense._id }
          }, { session });
        }
      } else if (expense.deleted === true) {
        // Hard delete for already soft-deleted expenses
        await Expense.findByIdAndDelete(expense._id, { session });
        actionCount++;

        // Completely remove expense from creator's allExpenses array
        if (expense.creator) {
          await User.findByIdAndUpdate(expense.creator, {
            $pull: { allExpenses: expense._id }
          }, { session });
        }
      }
    }

    if (actionCount === 0) {
      return res.status(404).json({ message: 'No expenses found to deleted' });
    }

    await session.commitTransaction();
    res.status(200).json({ 
      message: `Successfully deleted ${actionCount} expense(s)`,
      processedCount: actionCount
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to delete expenses, please try again later', error: error.message });
  } finally {
    session.endSession();
  }
};

const restoreExpense = async (req, res) => {
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

    // Perform restore for all selected expenses
    const updateResult = await Expense.updateMany(
      { 
        _id: { $in: ids },
        deleted: true // Only restore previously deleted expenses
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

    // If no expenses were modified, return a 404 error
    if (updateResult.modifiedCount === 0) {
      return res.status(404).json({ message: 'No expenses found to restore' });
    }

    // Re-add expenses to user's allExpenses array if needed
    const expenses = await Expense.find({ _id: { $in: ids } }).populate('creator').session(session);
    for (const expense of expenses) {
      if (expense.creator) {
        if (!expense.creator.allExpenses) {
          expense.creator.allExpenses = [];
        }
        if (!expense.creator.allExpenses.includes(expense._id)) {
          expense.creator.allExpenses.push(expense._id);
          await expense.creator.save({ session });
        }
      }
    }

    await session.commitTransaction();
    res.status(200).json({ 
      message: `Successfully restored ${updateResult.modifiedCount} expense(s)`,
      restoredCount: updateResult.modifiedCount
    });
  } catch (error) {
    await session.abortTransaction();
    res.status(500).json({ message: 'Failed to restore expenses, please try again later', error: error.message });
  } finally {
    session.endSession();
  }
};

export {
  getAllExpenses,
  getExpenseDetail,
  createExpense,
  updateExpense,
  deleteExpense,
  restoreExpense,
};
