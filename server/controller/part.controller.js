/* eslint-disable consistent-return */
/* eslint-disable no-underscore-dangle */
import Part from '../mongodb/models/part.js';

// Create a new part
const createPart = async (req, res) => {
  const { partName, brandName, quantity } = req.body;

  try {
    const existingPart = await Part.findOne({ partName, brandName });

    if (existingPart) {
      return res.status(400).json({ message: 'Part already exists' });
    }

    const newPart = new Part({
      partName,
      brandName,
      quantity,
      procurements: []  // Initialize with an empty procurements list
    });

    await newPart.save();
    res.status(201).json({ message: 'Part created successfully', part: newPart });
  } catch (error) {
    res.status(500).json({ message: 'Failed to create part', error: error.message });
  }
};

// Get all parts
const getAllParts = async (req, res) => {
    try {
      const parts = await Part.find().populate('procurements', null, null, { strictPopulate: false });
      res.status(200).json(parts);
    } catch (error) {
      res.status(500).json({ message: 'Failed to retrieve parts', error: error.message });
    }
  };

// Get a single part by ID
const getPartById = async (req, res) => {
    const { id } = req.params;
  
    try {
      const part = await Part.findById(id).populate('procurements', null, null, { strictPopulate: false });
  
      if (!part) {
        return res.status(404).json({ message: 'Part not found' });
      }
  
      res.status(200).json(part);
    } catch (error) {
      res.status(500).json({ message: 'Failed to retrieve part', error: error.message });
    }
  };
  

// Update part details
const updatePart = async (req, res) => {
    const { id } = req.params;
    const { partName, brandName, quantity } = req.body;
  
    try {
      const part = await Part.findById(id);
  
      if (!part) {
        return res.status(404).json({ message: 'Part not found' });
      }
  
      part.partName = partName || part.partName;
      part.brandName = brandName || part.brandName;
      part.quantity = quantity || part.quantity;
  
      await part.save();
      res.status(200).json({ message: 'Part updated successfully', part });
    } catch (error) {
      res.status(500).json({ message: 'Failed to update part', error: error.message });
    }
  };
  

// Delete a part
const deletePart = async (req, res) => {
    const { id } = req.params;
  
    try {
      const part = await Part.findById(id).populate('procurements', null, null, { strictPopulate: false });
  
      if (!part) {
        return res.status(404).json({ message: 'Part not found' });
      }
  
      if (part.procurements.length > 0) {
        return res.status(400).json({ message: 'Cannot delete part with procurements associated' });
      }
  
      await part.remove();
      res.status(200).json({ message: 'Part deleted successfully' });
    } catch (error) {
      res.status(500).json({ message: 'Failed to delete part', error: error.message });
    }
  };

export {
  createPart,
  deletePart,
  getAllParts,
  updatePart
}