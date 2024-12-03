// src/hooks/useDynamicHeight.js
import { useEffect, useState } from 'react';

const useDynamicHeight = (headerHeight = 64, marginAndPadding = 80) => {
  const [containerHeight, setContainerHeight] = useState('auto');

  useEffect(() => {
    const calculateHeight = () => {
      const windowHeight = window.innerHeight;
      const availableHeight = windowHeight - headerHeight - marginAndPadding;
      setContainerHeight(`${availableHeight}px`);
    };

    calculateHeight();
    window.addEventListener('resize', calculateHeight);
    return () => window.removeEventListener('resize', calculateHeight);
  }, [headerHeight, marginAndPadding]);

  return containerHeight;
};

export default useDynamicHeight;
