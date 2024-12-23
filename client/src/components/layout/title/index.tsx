/* eslint-disable */
import React from 'react';
import { useRouterContext, TitleProps } from '@pankod/refine-core';
import { Button } from '@pankod/refine-mui';

import { glogo, gammad  } from 'assets';

export const Title: React.FC<TitleProps> = ({ collapsed }) => {
  const { Link } = useRouterContext();

  return (
    <Button fullWidth variant="text" disableRipple>
      <Link to="/">
        {collapsed ? (
          <img src={glogo} alt="Yariga" width="48px" />
        ) : (
          <img src={gammad} alt="Yariga" width="140px" />
        )}
      </Link>
    </Button>
  );
};
