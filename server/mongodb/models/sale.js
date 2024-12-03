// server\mongodb\models\sale.js
import mongoose from 'mongoose';

const SaleSchema = new mongoose.Schema({
  seq: {
    type: Number,
    required: true,
  },
  date: {
    type: String,
    required: true,
  },
  clientName: {
    type: String,
    required: true,
  },
  tin: {
    type: String,
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
  outputVAT: {
    type: Number,
    required: true,
  },
  creator: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
  },
  deleted: { type: Boolean, default: false },
  deletedAt: { type: Date, default: null },
});

const saleModel = mongoose.model('Sale', SaleSchema);

export default saleModel;