import React, { useContext, useEffect, useState } from 'react';
import { LayoutProps } from '@pankod/refine-core';
import { Box } from '@pankod/refine-mui';

import { ColorModeContext } from 'contexts'; // Import the context for dark/light mode
import { Sider as DefaultSider } from '../sider';
import { Header as DefaultHeader } from '../header';

export const Layout: React.FC<LayoutProps> = ({
  Sider,
  Header,
  Footer,
  OffLayoutArea,
  children,
}) => {
  const SiderToRender = Sider ?? DefaultSider;
  const HeaderToRender = Header ?? DefaultHeader;
  const { mode } = useContext(ColorModeContext); // Access the current theme mode

  const [viewHeight, setViewHeight] = useState('100vh');

  // Adjust container height based on window size and scaling
  useEffect(() => {
    const updateHeight = () => {
      const windowHeight = window.innerHeight;
      setViewHeight(`${windowHeight}px`);
    };

    // Initial setup
    updateHeight();

    // Update on resize
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, []);

  return (
    <Box display="flex" flexDirection="row" height={viewHeight} overflow="hidden">
      <SiderToRender />
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          height: '100%',
          bgcolor: mode === 'dark' ? '#121212' : '#FFFFFF',
          transition: 'background 0.2s ease, color 0.2s ease',
        }}
      >
        <HeaderToRender />
        <Box
          component="main"
          sx={{
            p: { xs: 1, md: 2, lg: 3 },
            flexGrow: 1,
            bgcolor: mode === 'dark' ? '#1E1E1E' : '#F4F4F4',
            color: mode === 'dark' ? '#FFFFFF' : '#000000',
            transition: 'background 0.2s ease, color 0.2s ease',
            overflow: 'auto',
            height: '100%',
          }}
        >
          {children}
        </Box>
        {Footer && <Footer />}
      </Box>
      {OffLayoutArea && <OffLayoutArea />}
    </Box>
  );
};