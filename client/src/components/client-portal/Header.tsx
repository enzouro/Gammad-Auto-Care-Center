import { DarkModeOutlined, LightModeOutlined } from '@mui/icons-material';
import { AppBar, Box, IconButton, Stack, Toolbar, Tooltip } from '@pankod/refine-mui';
import { gammad, glogo } from 'assets';
import { ColorModeContext } from 'contexts';
import { useContext } from 'react'

const Header = () => {
  const { mode, setMode } = useContext(ColorModeContext);
  return (
    <AppBar color="default" position="sticky" elevation={0} sx={{ background: mode === 'dark' ? '#333' : '#FCFCFC' }}>
      <Toolbar>
        <Stack direction="row" width="100%" justifyContent="space-between" alignItems="center">
          {/* Logo on the left */}
     
          <img 
            src={glogo}// Adjust the path based on your assets folder structure
            alt="Logo"
            style={{
              height: '70px', // Adjust height as needed
              width: 'auto',
              objectFit: 'contain',
              background: 'yellow',
              borderRadius: '50%',
            }}
          />
   
          {/* Dark mode toggle on the right */}
          <Tooltip title={`Switch to ${mode === 'dark' ? 'Light' : 'Dark'} Mode`}>
            <IconButton
              onClick={() => {
                setMode();
              }}
              sx={{ color: mode === 'dark' ? '#FFF' : '#000' }}
            >
              {mode === 'dark' ? (
                <LightModeOutlined sx={{ color: '#FFF' }} />
              ) : (
                <DarkModeOutlined sx={{ color: '#000' }} />
              )}
            </IconButton>
          </Tooltip>
        </Stack>
      </Toolbar>
    </AppBar>
  )
}

export default Header