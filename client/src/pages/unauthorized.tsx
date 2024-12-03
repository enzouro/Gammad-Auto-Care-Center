// eslint-disable
import React from 'react';
import { Box, Typography, Button } from '@pankod/refine-mui';
import { useNavigate } from '@pankod/refine-react-router-v6';

export const UnauthorizedPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      height="100vh"
    >
      <Typography variant="h4" gutterBottom>
        Authorized Personnel Only
      </Typography>
      <Typography variant="body1" gutterBottom>
        If you're a user, please contact the admin for access.
      </Typography>
      <Button
        variant="contained"
        color="primary"
        onClick={() => navigate('/login')}
        sx={{ mt: 2 }}
      >
        Back to Login
      </Button>
    </Box>
  );
};
