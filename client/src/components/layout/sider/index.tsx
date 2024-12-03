/* eslint-disable no-shadow */
import React, { useContext, useState, useEffect } from 'react';
import {
  Box,
  Drawer,
  Sider as DefaultSider,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Collapse,
  Tooltip,
  Button,
  IconButton,
  MuiList,
} from '@pankod/refine-mui';
import {
  ListOutlined,
  Logout,
  ExpandLess,
  ExpandMore,
  MenuRounded,
  Dashboard,
  ChevronRightOutlined,
  ChevronLeftOutlined,
} from '@mui/icons-material';
import {
  CanAccess,
  ITreeMenu,
  useIsExistAuthentication,
  useLogout,
  useTitle,
  useTranslate,
  useRouterContext,
  useMenu,
  useRefineContext,
} from '@pankod/refine-core';
import { useLocation } from '@pankod/refine-react-router-v6';
import { ColorModeContext } from 'contexts';
import { Title as DefaultTitle } from '../title';
import LogoutConfirmationDialog from 'components/common/LogutConfirmationDialog';

export const Sider: typeof DefaultSider = ({ render }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [opened, setOpened] = useState(false);
  const [isAdmin, setIsAdmin] = useState(false);
  const [viewHeight, setViewHeight] = useState('100vh');
  const [logoutDialogOpen, setLogoutDialogOpen] = useState(false);
  
  const { pathname } = useLocation();

  console.log('pathname', pathname);
  
  const { mode } = useContext(ColorModeContext);
  // Background gradients for dark and light modes - keeping yellow theme
  const backgroundGradient = mode === 'dark'
    ? 'linear-gradient(0deg, rgba(18,18,18,1)  10%, rgba(51,48,0,1) 40%, rgba(102,97,0,1) 80%, rgba(255,240,0,1) 100%)'
    : 'linear-gradient(0deg, rgba(255,255,255,1) 10%, rgba(255,251,214,1) 40%, rgba(255,246,153,1) 80%, rgba(255,240,0,1) 100%)';

    useEffect(() => {
      // Check isAdmin status from localStorage
      const user = localStorage.getItem('user');
      if (user) {
        const userData = JSON.parse(user);
        setIsAdmin(!!userData.isAdmin); // Convert to boolean with !!
      }
    }, []);
  
  // Dynamic height calculation
  useEffect(() => {
    const updateHeight = () => {
      const windowHeight = window.innerHeight;
      setViewHeight(`${windowHeight}px`);
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  const drawerWidth = () => {
    if (collapsed) return 64;
    return 200;
  };

const handleLogout = () => {
  setLogoutDialogOpen(true); // Open the dialog
};

const confirmLogout = () => {
  mutateLogout(); // Perform the logout action
  setLogoutDialogOpen(false); // Close the dialog
};

const cancelLogout = () => {
  setLogoutDialogOpen(false); // Close the dialog without any action
};

  const t = useTranslate();
  const { Link } = useRouterContext();
  const { hasDashboard } = useRefineContext();
  const translate = useTranslate();

  const { menuItems, selectedKey, defaultOpenKeys } = useMenu();
  const isExistAuthentication = useIsExistAuthentication();
  const { mutate: mutateLogout } = useLogout();
  const Title = useTitle();

  const [open, setOpen] = useState<{ [key: string]: boolean }>({});

  React.useEffect(() => {
    setOpen((previousOpen) => {
      const previousOpenKeys: string[] = Object.keys(previousOpen);
      const uniqueKeys = new Set([...previousOpenKeys, ...defaultOpenKeys]);
      const uniqueKeysRecord = Object.fromEntries(
        Array.from(uniqueKeys.values()).map((key) => [key, true]),
      );
      return uniqueKeysRecord;
    });
  }, [defaultOpenKeys]);

  const RenderToTitle = Title ?? DefaultTitle;

  const handleClick = (key: string) => {
    setOpen({ ...open, [key]: !open[key] });
  };

  const renderTreeView = (tree: ITreeMenu[], selectedKey: string) => tree.map((item: ITreeMenu) => {
    const { icon, label, route, name, children, parentName } = item;

    // Skip rendering user-management for non-admin users
    if (name === 'user-management' && !isAdmin) {
      return null;
    }

    const isOpen = open[route || ''] || false;

    const isSelected = route === selectedKey;
    const isNested = !(parentName === undefined);

    if (children.length > 0) {
      return (
        <CanAccess
          key={route}
          resource={name.toLowerCase()}
          action="list"
          params={{
            resource: item,
          }}
        >
          <div key={route}>
            <Tooltip
              title={label ?? name}
              placement="right"
              disableHoverListener={!collapsed}
              arrow
            >
              <ListItemButton
                onClick={() => {
                  if (collapsed) {
                    setCollapsed(false);
                    if (!isOpen) {
                      handleClick(route || '');
                    }
                  } else {
                    handleClick(route || '');
                  }
                }}
                sx={{
                  pl: isNested ? 4 : 2,
                  justifyContent: 'center',
                  '&.Mui-selected': {
                    '&:hover': {
                      backgroundColor: 'transparent',
                    },
                    backgroundColor: 'transparent',
                  },
                }}
              >
                <ListItemIcon
                  sx={{
                    justifyContent: 'center',
                    minWidth: 36,
                    color: mode === 'dark' ? '#fff' : '#141414',
                  }}
                >
                  {icon ?? <ListOutlined />}
                </ListItemIcon>
                <ListItemText
                  primary={label}
                  primaryTypographyProps={{
                    noWrap: true,
                    fontSize: '16px',
                    fontWeight: isSelected ? 'bold' : 'normal',
                    color: mode === 'dark' ? '#fff' : '#141414',
                  }}
                />
                {!collapsed && (isOpen ? <ExpandLess /> : <ExpandMore />)}
              </ListItemButton>
            </Tooltip>
            {!collapsed && (
            <Collapse in={open[route || '']} timeout="auto" unmountOnExit>
              <MuiList component="div" disablePadding>
                {renderTreeView(children, selectedKey)}
              </MuiList>
            </Collapse>
            )}
          </div>
        </CanAccess>
      );
    }

    return (
      <CanAccess
        key={route}
        resource={name.toLowerCase()}
        action="list"
        params={{ resource: item }}
      >
        <Tooltip
          title={label ?? name}
          placement="right"
          disableHoverListener={!collapsed}
          arrow
        >
          <ListItemButton
            component={Link}
            to={route}
            selected={isSelected}
            onClick={() => {
              setOpened(false);
            }}
            sx={{
              pl: isNested ? 4 : 2,
              py: isNested ? 1.25 : 1,
              '&.Mui-selected': {
                '&:hover': {
                  backgroundColor: isSelected ? '#ffd300' : 'transparent',
                },
                backgroundColor: isSelected ? '#fff000' : 'transparent',
              },
              justifyContent: 'center',
              margin: '10px auto',
              borderRadius: '12px',
              minHeight: '56px',
              width: '90%',
            }}
          >
            <ListItemIcon
              sx={{
                justifyContent: 'center',
                minWidth: 36,
                color: isSelected ? '#141414' : (mode === 'dark' ? '#fff' : '#141414'),
              }}
            >
              {icon ?? <ListOutlined />}
            </ListItemIcon>
            <ListItemText
              primary={label}
              primaryTypographyProps={{
                noWrap: true,
                fontSize: '16px',
                fontWeight: isSelected ? 'bold' : 'normal',
                color: isSelected ? '#141414' : (mode === 'dark' ? '#fff' : '#141414'),
                marginLeft: '10px',
              }}
            />
          </ListItemButton>
        </Tooltip>
      </CanAccess>
    );
  });

  const dashboard = hasDashboard ? (
    <CanAccess resource="dashboard" action="list">
      <Tooltip
        title={translate('dashboard.title', 'Summary')}
        placement="right"
        disableHoverListener={!collapsed}
        arrow
      >
        <ListItemButton
          component={Link}
          to="/"
          selected={selectedKey === '/'}
          onClick={() => {
            setOpened(false);
          }}
          sx={{
            pl: 2,
            py: 1,
            '&.Mui-selected': {
              '&:hover': {
                backgroundColor: pathname === '/' ? '#ffd300' : 'transparent',
              },
              backgroundColor: pathname === '/' ? '#fff000' : 'transparent',
            },
            justifyContent: 'center',
            margin: '10px auto',
            borderRadius: '12px',
            minHeight: '56px',
            width: '90%',
          }}
        >
          <ListItemIcon
            sx={{
              justifyContent: 'center',
              minWidth: 36,
              color: pathname === '/' ? '#141414' : (mode === 'dark' ? '#fff' : '#141414'),
            }}
          >
            <Dashboard />
          </ListItemIcon>
          <ListItemText
            primary={translate('dashboard.title', 'Summary')}
            primaryTypographyProps={{
              noWrap: true,
              fontSize: '16px',
              fontWeight: pathname === '/' ? 'bold' : 'normal',
              color: pathname === '/' ? '#141414' : (mode === 'dark' ? '#fff' : '#141414'),
              marginLeft: '10px',
            }}
          />
        </ListItemButton>
      </Tooltip>
    </CanAccess>
  ) : null;

  const logout = isExistAuthentication && (
    <>
      <Tooltip
        title={t('buttons.logout', 'Logout')}
        placement="right"
        disableHoverListener={!collapsed}
        arrow
      >
        <ListItemButton
          key="logout"
          onClick={handleLogout}
          sx={{
            justifyContent: 'center',
            margin: '10px auto',
            borderRadius: '12px',
            minHeight: '56px',
            width: '90%',
          }}
        >
          <ListItemIcon
            sx={{
              justifyContent: 'center',
              minWidth: 36,
              color: mode === 'dark' ? '#fff' : '#141414',
            }}
          >
            <Logout />
          </ListItemIcon>
          <ListItemText
            primary={t('buttons.logout', 'Logout')}
            primaryTypographyProps={{
              noWrap: true,
              fontSize: '16px',
              color: mode === 'dark' ? '#fff' : '#141414',
            }}
          />
        </ListItemButton>
      </Tooltip>
      <LogoutConfirmationDialog
        open={logoutDialogOpen}
        onConfirm={confirmLogout}
        onCancel={cancelLogout}
      />
    </>
  );
  

  const items = renderTreeView(menuItems, selectedKey).filter((item): item is React.ReactElement => item !== null);
    
  const renderSider = () => {
    if (render) {
      return render({
        dashboard,
        logout,
        items,
        collapsed,
      });
    }
    return (
      <>
        {dashboard}
        {items}
        {logout}
      </>
    );
  };

  const drawer = (
    <MuiList 
      disablePadding 
      sx={{ 
        mt: 1, 
        color: '#919080',
        overflowY: 'auto',  // Allow vertical scroll only in this inner container
        overflowX: 'hidden'
      }}
    >
      {renderSider()}
    </MuiList>
  );
  
  return (
    <>
      <Box
        sx={{
          width: { xs: drawerWidth() },
          display: {
            xs: 'none',
            md: 'block',
          },
          transition: 'width 0.3s ease',
        }}
      />
      <Box
        component="nav"
        sx={{
          position: 'fixed',
          zIndex: 1101,
          width: { sm: drawerWidth() },
          display: 'flex',
          height: viewHeight,
          overflow: 'hidden',
        }}
      >
        <Drawer
          variant="temporary"
          open={opened}
          onClose={() => setOpened(false)}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { sm: 'block', md: 'none' },
            '& .MuiDrawer-paper': {
              width: 256,
              height: viewHeight,
              background: backgroundGradient,
              overflow: 'hidden',
            },
          }}
        >
          <Box
            sx={{
              height: 64,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <RenderToTitle collapsed={false} />
          </Box>
          <Box
            sx={{
              flex: 1,
              overflow: 'hidden', // Prevent overflow on the main Box
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {drawer}
          </Box>
        </Drawer>
        <Drawer
          variant="permanent"
          PaperProps={{ elevation: 0 }}
          sx={{
            display: { xs: 'none', md: 'block' },
            '& .MuiDrawer-paper': {
              width: drawerWidth(),
              height: viewHeight,
              background: backgroundGradient,
              transition: 'width 200ms cubic-bezier(0.4, 0, 0.6, 1) 0ms',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            },
          }}
          open
        >
          <Box
            sx={{
              height: 64,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <RenderToTitle collapsed={collapsed} />
          </Box>
          <Box
            sx={{
              flex: 1,
              overflow: 'hidden',  // Prevent overflow here
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            <Box
              sx={{
                flexGrow: 1,
                overflowX: 'hidden',
                overflowY: 'auto',  // Set scroll here only
                '&::-webkit-scrollbar': {
                  width: '6px',
                },
                '&::-webkit-scrollbar-track': {
                  background: 'transparent',
                },
                '&::-webkit-scrollbar-thumb': {
                  background: mode === 'dark' ? '#ffffff3d' : '#1414143d',
                  borderRadius: '3px',
                },
                '&::-webkit-scrollbar-thumb:hover': {
                  background: mode === 'dark' ? '#ffffff5d' : '#1414145d',
                },
              }}
            >
              {drawer}
            </Box>
          </Box>
          <Button
            sx={{
              background: mode === 'dark' ? '#0D0D0D' : '#fff000',
              color: mode === 'dark' ? '#fff' : '#141414',
              textAlign: 'center',
              borderRadius: 0,
              borderTop: '1px solid #ffffff1a',
              height: '40px',
              minHeight: '40px',
              '&:hover': {
                background: mode === 'dark' ? '#fff000' : '#c3b800',
                color: mode === 'dark' ? '#141414' : '#fff',
              },
            }}
            fullWidth
            size="large"
            onClick={() => setCollapsed((prev) => !prev)}
          >
            {collapsed ? <ChevronRightOutlined /> : <ChevronLeftOutlined />}
          </Button>
        </Drawer>
        <Box
          sx={{
            display: { xs: 'block', md: 'none' },
            position: 'fixed',
            top: '64px',
            left: '0px',
            borderRadius: '0 6px 6px 0',
            bgcolor: mode === 'dark' ? '#333' : '#fff000',
            zIndex: 1199,
            width: '36px',
          }}
        >
          <IconButton
            sx={{ 
              color: mode === 'dark' ? '#fff' : '#141414', 
              width: '36px' 
            }}
            onClick={() => setOpened((prev) => !prev)}
          >
            <MenuRounded />
          </IconButton>
        </Box>
      </Box>
    </>
  );
};