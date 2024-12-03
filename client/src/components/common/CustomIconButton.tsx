// client\src\components\common\CustomIconButton.tsx
import { IconButton, Tooltip } from '@pankod/refine-mui';
import { CustomIconButtonProps } from 'interfaces/common';

const CustomIconButton = ({
  icon,
  title,
  backgroundColor,
  color,
  handleClick,
  size = "small"
}: CustomIconButtonProps) => {
  return (
    <Tooltip title={title}>
      <IconButton
        onClick={handleClick}
        size={size}
        sx={{
          bgcolor: backgroundColor,
          color,
          p: 0.5,
          width: 32,
          height: 32,
          '&:hover': {
            bgcolor: backgroundColor,
            opacity: 0.9,
          },
          transition: 'all 0.2s ease-in-out'
        }}
      >
        {icon}
      </IconButton>
    </Tooltip>
  );
};

export default CustomIconButton;
