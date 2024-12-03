//server\mongodb\models\part.js
import mongoose from 'mongoose';

const PartsSchema = new mongoose.Schema({
  partName: {
    type: String,
    required: true,
  },
  brandName: {
    type: String,
    required: true,
  },
  qtyLeft: {
    type: Number,
    required: true,
    default: 0,  // Start with 0 quantity until parts are procured
  },
  deleted: { type: Boolean, default: false },
  deletedAt: { type: Date, default: null },
  procurements: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Procurement' }] // Make sure this is defined
});

const PartsModel = mongoose.model('Part', PartsSchema);

export default PartsModel;
