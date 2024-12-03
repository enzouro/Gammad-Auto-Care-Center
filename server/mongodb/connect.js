import mongoose from 'mongoose';

const connectDB = (url) => {
  mongoose.set('strictQuery', true);
  mongoose.connect(url, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
    retryWrites: true
  })
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => {
    console.error('Failed to connect with MongoDB');
    console.error(err);
  });
};

export default connectDB;