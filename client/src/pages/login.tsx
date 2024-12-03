/* eslint-disable */
import { useLogin, useNavigation } from '@pankod/refine-core';
import { Box, Container } from '@pankod/refine-mui';
import axios from 'axios';
import { CredentialResponse } from 'interfaces/google';
import { useEffect, useRef } from 'react';
import { gammad } from '../assets';

const GoogleButton: React.FC<{ onLogin: (res: CredentialResponse) => void }> = ({ onLogin }) => {
  const divRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.google || !divRef.current) {
      return;
    }

    try {
      window.google.accounts.id.initialize({
        ux_mode: 'popup',
        client_id: process.env.REACT_APP_GOOGLE_CLIENT_ID,
        callback: async (res: CredentialResponse) => {
          if (res.credential) {
            const profileObj = JSON.parse(atob(res.credential.split('.')[1]));
            const response = await axios.post('http://localhost:8080/api/v1/users', {
              name: profileObj.name,
              email: profileObj.email,
              avatar: profileObj.picture,
            });

            if (response.data.isAllowed) {
              onLogin(res);
            } else {
              // Redirect to Unauthorized page
              window.location.href = '/unauthorized';
            }
          }
        },
      });
      window.google.accounts.id.renderButton(divRef.current, {
        theme: 'filled_blue',
        size: 'medium',
        type: 'standard',
      });
    } catch (error) {
      console.log(error);
    }
  }, [onLogin]);

  return <div ref={divRef} />;
};

export const Login: React.FC = () => {
  const { mutate: login } = useLogin();

  return (
    <Box component="div" sx={{ background: 'linear-gradient(37deg, rgba(0,0,0,1) 10%, rgba(255,241,0,1) 80%)' }}>
      <Container component="main" maxWidth="xs" sx={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        height: '100vh',
      }}
      >
        <Box sx={{
          display: 'flex',
          justifyContent: 'center',
          flexDirection: 'column',
          alignItems: 'center',
        }}
        >
          <img src={gammad} alt="Gammad Logo" />
          <Box mt={4}>
            <GoogleButton onLogin={login} />
          </Box>
        </Box>
      </Container>
    </Box>
  );
};
