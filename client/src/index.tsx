/* eslint-disable */
import React from 'react';
import { createRoot } from 'react-dom/client';
import { CssBaseline } from '@pankod/refine-mui';

import reportWebVitals from './reportWebVitals';
import App from './App';
import { ColorModeContextProvider } from './contexts';
import './index.css';

const container = document.getElementById('root') as HTMLElement;
const root = createRoot(container);

root.render(
  <React.StrictMode>
    <ColorModeContextProvider>
      <CssBaseline>
        <App />
      </CssBaseline>
    </ColorModeContextProvider>
  </React.StrictMode>,
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
