// server\mongodb\models\procurement.js
import mongoose from 'mongoose';

const ProcurementSchema = new mongoose.Schema({
  seq: {
    type: Number,
    required: true,
  },
  date: {
    type: String,
    required: true,
  },
  supplierName: {
    type: String,
    required: true,
  },
  reference: {
    type: String,
    required: true,
  },
  tin: {
    type: String,
    required: true,
  },
  address: {
    type: String,
    required: true,
  },
  part: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Part',
    required: true,  // Each procurement must reference a part
  },
  description: {
    type: String,
    required: true,
  },
  quantityBought: {
    type: Number,
    required: true,
  },
  amount: {
    type: Number,
    required: true,
  },
  netOfVAT: {
    type: Number,
    required: true,
  },
  inputVAT: {
    type: Number,
    required: true,
  },
  isNonVat: {
    type: Boolean,
    required: true,
  },
  noValidReceipt: {
    type: Boolean,
    required: true,
  },
  creator: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
  },
  deleted: { type: Boolean, default: false },
  deletedAt: { type: Date, default: null },
});

const procurementModel = mongoose.model('Procurement', ProcurementSchema);

export default procurementModel;
