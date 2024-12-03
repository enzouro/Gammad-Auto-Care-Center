//server\mongodb\models\deployment.js
import mongoose from 'mongoose';

const DeploymentSchema = new mongoose.Schema({
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
  vehicleModel: {
    type: String,
    required: true,
  },
  arrivalDate: {
    type: String,
    required: true,
  },
  parts: [{
    part: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Part',
    },
    quantityUsed: {
      type: Number,
      required: function() {
        return this.part != null;
      }
    }
  }],
  deploymentStatus: {
    type: Boolean,
    default: false,
  },
  deploymentDate: {
    type: String,
  },
  releaseStatus: {
    type: Boolean,
    default: false,
  },
  releaseDate: {
    type: String,
  },
  creator: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
  },
  repairStatus: {
    type: String,
  },
  repairedDate: {
    type: String,
  },
  trackCode: {
    type: String,
    required: true,
  },
  deleted: { type: Boolean, default: false },
  deletedAt: { type: Date, default: null },
});

const deploymentModel = mongoose.model('Deployment', DeploymentSchema);

export default deploymentModel;