// client\src\components\common\CustomButton.tsx
import { Button } from '@pankod/refine-mui';
import { CustomButtonProps } from 'interfaces/common';

const CustomButton = ({
  type = "button",
  title,
  backgroundColor,
  color,
  fullWidth = false,
  icon,
  handleClick,
  disabled = false
}: CustomButtonProps) => {
  return (
    <Button
      type={type}
      disabled={disabled}
      onClick={handleClick}
      sx={{
        flex: fullWidth ? 1 : 'unset',
        padding: '10px 15px',
        width: fullWidth ? '100%' : 'fit-content',
        minWidth: 130,
        backgroundColor,
        color,
        fontSize: 16,
        fontWeight: 600,
        gap: '10px',
        display: 'flex',
        alignItems: 'center',
        textTransform: 'capitalize',
        '&:hover': {
          opacity: 0.9,
          backgroundColor,
        },
        transition: 'all 0.2s ease-in-out',
        borderRadius: 5
      }}
    >
      {icon}
      {title}
    </Button>
  );
};

export default CustomButton;
