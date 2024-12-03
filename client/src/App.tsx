import React from 'react';
import {
  AttachMoneyOutlined,
  FileOpenSharp,
  ManageAccounts,
  NoCrashOutlined,
  Timeline,
  WalletOutlined,
  WarehouseOutlined,
} from '@mui/icons-material';
import { AuthProvider, Refine } from '@pankod/refine-core';
import {
  CssBaseline,
  ErrorComponent,
  GlobalStyles,
  notificationProvider,
  ReadyPage,
  RefineSnackbarProvider,
} from '@pankod/refine-mui';
import routerProvider from '@pankod/refine-react-router-v6';
import dataProvider from '@pankod/refine-simple-rest';
import axios, { AxiosRequestConfig } from 'axios';

import { Header, Layout, Sider, Title } from 'components/layout';
import { ColorModeContextProvider } from 'contexts';
import { CredentialResponse } from 'interfaces/google';
import { parseJwt } from 'utils/parse-jwt';

import {
  CreateProcurement,
  EditProcurement,
  Home,
  Login,
  ProcurementDetails,
} from 'pages';
import AllProcurements from 'pages/all-procurements';
import UserManagement from 'pages/user-management';
import { UnauthorizedPage } from 'pages/unauthorized';
import SaleDetails from 'pages/sale-details';
import AllSales from 'pages/all-sales';
import CreateSale from 'pages/create-sale';
import EditSale from 'pages/edit-sale';
import AllDeployments from 'pages/all-deployment';
import DeploymentDetails from 'pages/deployment-details';
import CreateDeployment from 'pages/create-deployment';
import EditDeployment from 'pages/edit-deployment';
import ClientPortal from 'pages/client-portal';
import AllExpenses from 'pages/all-expenses';
import CreateExpense from 'pages/create-expense';
import EditExpense from 'pages/edit-expense';
import ExpenseDetails from 'pages/expense-details';
import Forecast from 'pages/forecast';
import ReportsPage from 'pages/reports';

const axiosInstance = axios.create();
axiosInstance.interceptors.request.use((request: AxiosRequestConfig) => {
  const token = localStorage.getItem('token');
  if (request.headers) {
    request.headers.Authorization = `Bearer ${token}`;
  } else {
    request.headers = {
      Authorization: `Bearer ${token}`,
    };
  }
  return request;
});

const suppressConsoleErrors = () => {
  const originalConsoleError = console.error;
  console.error = (...args) => {
    if (
      args[0]?.includes('POST https://accounts.google.com/gsi/revoke') ||
      args[0]?.includes('The specified user is not signed in.')
    ) {
      return;
    }
  };
};

const App = () => {
  const [isAdmin, setIsAdmin] = React.useState(false);

  // Add a function to check user authorization
  const checkUserAuthorization = async () => {
    try {
      const user = localStorage.getItem('user');
      if (!user) return;
      
      const parsedUser = JSON.parse(user);
      const response = await fetch(`http://localhost:8080/api/v1/users/${parsedUser.userid}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        // If user is not found or unauthorized, trigger logout
        authProvider.logout({} as any);  // Provide empty object as parameter
        window.location.href = '/unauthorized';
      }

      const userData = await response.json();
      if (!userData.isAllowed) {
        // If user is explicitly not allowed, trigger logout
        authProvider.logout({} as any);  // Provide empty object as parameter
        window.location.href = '/unauthorized';
      }
    } catch (error) {
      console.error('Error checking user authorization:', error);
    }
  };

  React.useEffect(() => {
    const user = localStorage.getItem('user');
    if (user) {
      const parsedUser = JSON.parse(user); // Safe to parse as user is not null
      if (parsedUser.isAdmin) {
        setIsAdmin(parsedUser.isAdmin);
      }
    }
    
    // Set up periodic authorization check
    const authCheckInterval = setInterval(checkUserAuthorization, 30000); // Check every 30 seconds
    
    // Initial check
    checkUserAuthorization();
    
    suppressConsoleErrors();

    // Cleanup interval on component unmount
    return () => clearInterval(authCheckInterval);
  }, []);

  const authProvider: AuthProvider = {
    login: async ({ credential }: CredentialResponse) => {
      const profileObj = credential ? parseJwt(credential) : null;

      // Save user to MongoDB
      if (profileObj) {
        const response = await fetch('http://localhost:8080/api/v1/users', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: profileObj.name,
            email: profileObj.email,
            avatar: profileObj.picture,
          }),
        });

        const data = await response.json();

        if (response.status === 200) {
          if (!data.isAllowed) {
            // Prevent login if user is not allowed
            return Promise.reject(new Error('User is not allowed to access the system'));
          }
          
          localStorage.setItem(
            'user',
            JSON.stringify({
              ...profileObj,
              avatar: profileObj.picture,
              userid: data._id,
              isAdmin: data.isAdmin,
            }),
          );
          setIsAdmin(data.isAdmin); // Set isAdmin after login
        } else {
          return Promise.reject();
        }
      }

      localStorage.setItem('token', `${credential}`);
      return Promise.resolve();
    },
    logout: () => {
      const token = localStorage.getItem('token');

      if (token && typeof window !== 'undefined') {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
        axios.defaults.headers.common = {};
        window.google?.accounts.id.revoke(token, () => Promise.resolve());
      }

      return Promise.resolve();
    },
    checkError: () => Promise.resolve(),
    checkAuth: async () => {
      const token = localStorage.getItem('token');

      if (token) {
        return Promise.resolve();
      }
      return Promise.reject();
    },

    getPermissions: () => Promise.resolve(),
    getUserIdentity: async () => {
      const user = localStorage.getItem('user');
      if (user) {
        return Promise.resolve(JSON.parse(user));
      }
    },
  };

  const generateResources = () => {
    const baseResources = [
      {
        name: 'forecasting',
        list: Forecast,
        icon: <Timeline />,
        options: { label: 'Forecasting' }
      },
      {
        name: 'procurements',
        list: AllProcurements,
        show: ProcurementDetails,
        create: CreateProcurement,
        edit: EditProcurement,
        icon: <WarehouseOutlined />,
      },
      {
        name: 'deployments',
        list: AllDeployments,
        show: DeploymentDetails,
        create: CreateDeployment,
        edit: EditDeployment,
        icon: <NoCrashOutlined />,
      },
      {
        name: 'expenses',
        list: AllExpenses,
        show: ExpenseDetails,
        create: CreateExpense,
        edit: EditExpense,
        icon: <WalletOutlined/>,
      },
      {
        name: 'sales',
        list: AllSales,
        show: SaleDetails,
        create: CreateSale,
        edit: EditSale,
        icon: <AttachMoneyOutlined/>,
      },
      {
        name: 'reports',
        list: ReportsPage,
        icon: <FileOpenSharp />
      },
      
    ];

    // Add user management resource only for admin users
    if (isAdmin) {
      baseResources.push({
        name: 'user-management',
        list: UserManagement,
        options: { label: 'User Management' },
        icon: <ManageAccounts />,
      });
    }
    return baseResources;
  };

  return (
    <ColorModeContextProvider>
      <CssBaseline />
      <GlobalStyles styles={{ html: { WebkitFontSmoothing: 'auto' } }} />
      <RefineSnackbarProvider>
        <Refine
          dataProvider={dataProvider('http://localhost:8080/api/v1')}
          notificationProvider={notificationProvider}
          ReadyPage={ReadyPage}
          catchAll={<ErrorComponent />}
          resources={generateResources()}
          Title={Title}
          Sider={Sider}
          Layout={Layout}
          Header={Header}
            routerProvider={{
              ...routerProvider,
              routes: [
                {
                  path: '/unauthorized',
                  element: <UnauthorizedPage />
                },
                {
                  path: '/client-portal',
                  element: <ClientPortal />
                },
              ],
            }}
          authProvider={authProvider}
          LoginPage={Login}
          DashboardPage={Home}
        />
      </RefineSnackbarProvider>
    </ColorModeContextProvider>
  );
};

export default App;