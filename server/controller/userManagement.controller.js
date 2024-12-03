import User from '../mongodb/models/user.js';

const getAllUsers = async (req, res) => {
  try {
    const users = await User.find({});
    res.status(200).json(users);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

const updateUserAllowedStatus = async (req, res) => {
    try {
      const { id } = req.params;
      const { isAllowed } = req.body;
      const updatedUser = await User.findByIdAndUpdate(
        id,
        { isAllowed },
        { new: true, runValidators: true }
      );
  
      if (!updatedUser) {
        return res.status(404).json({ message: 'User not found' });
      }
  
       // Debug log
      res.status(200).json({ message: 'User Allowed status changed' });
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
  };

const deleteUser = async (req, res) => {
  try {
    const { id } = req.params;

    const deletedUser = await User.findByIdAndDelete(id);

    if (!deletedUser) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.status(200).json({ message: 'User deleted successfully' });
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

export { getAllUsers, updateUserAllowedStatus, deleteUser };