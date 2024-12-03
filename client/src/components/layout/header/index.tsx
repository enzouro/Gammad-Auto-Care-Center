import React, { useContext } from 'react';
import { useGetIdentity } from '@pankod/refine-core';
import {
  AppBar,
  IconButton,
  Avatar,
  Stack,
  Toolbar,
  Typography,
  Tooltip,
} from '@pankod/refine-mui';
import { DarkModeOutlined, LightModeOutlined } from '@mui/icons-material';
import { ColorModeContext } from 'contexts';

export const Header: React.FC = () => {
  const { mode, setMode } = useContext(ColorModeContext);
  const { data: user } = useGetIdentity();
  const showUserInfo = user && (user.name || user.avatar);

  return (
    <AppBar color="default" position="sticky" elevation={0} sx={{ background: mode === 'dark' ? '#333' : '#FCFCFC' }}>
      <Toolbar>
        <Stack direction="row" width="100%" justifyContent="flex-end" alignItems="center">
          <Tooltip title={`Switch to ${mode === 'dark' ? 'Light' : 'Dark'} Mode`}>
            <IconButton
              onClick={() => {
                setMode();
              }}
              sx={{ color: mode === 'dark' ? '#FFF' : '#000' }} // White for dark mode, black for light mode
            >
              {mode === 'dark' ? (
                <LightModeOutlined sx={{ color: '#FFF' }} /> // White icon for dark mode
              ) : (
                <DarkModeOutlined sx={{ color: '#000' }} /> // Black icon for light mode
              )}
            </IconButton>
          </Tooltip>
          {showUserInfo && (
            <Stack direction="row" gap="12px" alignItems="center" justifyContent="center">
              {user.avatar && <Avatar src={user?.avatar} alt={user?.name} />}
              {user.name && (
                <Stack direction="column">
                  <Typography sx={{ fontSize: 14, fontWeight: 600, color: mode === 'dark' ? '#FFF' : '#11142D' }}>{user?.name}</Typography>
                  <Typography sx={{ fontSize: 12, color: mode === 'dark' ? '#B0B0B0' : '#808191' }}>{user?.email}</Typography>
                </Stack>
              )}
            </Stack>
          )}
        </Stack>
      </Toolbar>
    </AppBar>
  );
};
