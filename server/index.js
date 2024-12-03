// server\index.js
import express from 'express';
import * as dotenv from 'dotenv';
import cors from 'cors';

import connectDB from './mongodb/connect.js';
import userRouter from './routes/user.routes.js';
import procurementRouter from './routes/procurement.routes.js';
import partRouter from './routes/part.routes.js';
import userManagementRoutes from './routes/userManagement.routes.js';
import saleRouter from './routes/sale.routes.js';
import deploymentRouter from './routes/deployment.routes.js'
import clientPortalRouter from './routes/clientPortal.routes.js'
import expenseRouter from './routes/expense.routes.js';
import forecastingRoutes from './routes/forecasting.routes.js';
import reportsRoutes from './routes/reports.routes.js';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));

app.get('/', (req, res) => {
  res.status(200).json({
    message: 'Welcome to Gammad!',
  });
});

app.use('/api/v1/users', userRouter);
app.use('/api/v1/procurements', procurementRouter);
app.use('/api/v1/parts', partRouter);
app.use('/api/v1/user-management', userManagementRoutes);
app.use('/api/v1/sales', saleRouter);
app.use('/api/v1/deployments', deploymentRouter);
app.use('/api/v1/clientPortal', clientPortalRouter);
app.use('/api/v1/expenses', expenseRouter);
app.use('/api/forecasting', forecastingRoutes);
app.use('/api/v1/reports', reportsRoutes);

const startServer = async () => {
  try {
    connectDB(process.env.MONGODB_URL);
    app.listen(8080, () => console.log('Server started on port 8080'));
  } catch (error) {
    console.log(error);
  }
};

startServer();
