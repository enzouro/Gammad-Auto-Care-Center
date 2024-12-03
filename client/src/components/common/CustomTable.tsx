import React, { useContext, useMemo, useState } from 'react';
import { 
  DataGrid, 
  GridColDef,
  GridSelectionModel, 
  GridSortModel
} from '@pankod/refine-mui';
import { 
  Toolbar,
  Typography,
  Box,
  Stack
} from '@pankod/refine-mui';
import { Delete, Edit, Visibility, Restore } from '@mui/icons-material';
import { ColorModeContext } from 'contexts';
import CustomIconButton from 'components/common/CustomIconButton';

interface CustomTableProps {
  rows: any[];
  columns: GridColDef[];
  containerHeight?: string | number;
  onView?: (id: string) => void;
  onEdit?: (id: string) => void;
  onDelete?: (ids: string[]) => void;
  onRestore?: (ids: string[]) => void;
  initialSortModel?: GridSortModel; // Use GridSortModel type
}

const CustomTableToolbar = ({
  numSelected,
  onView,
  onEdit,
  onDelete,
  onRestore,
  selectedId,
  rows
}: {
  numSelected: number;
  onView?: (id: string) => void;
  onEdit?: (id: string) => void;
  onDelete?: (ids: string[]) => void;
  onRestore?: (ids: string[]) => void;
  selectedId?: string;
  rows: any[];
}) => {
  // Find the selected row(s)
  const selectedRows = rows.filter(row => selectedId?.split(',').includes(row.id));
  
  // Check if all selected rows have the same deleted status
  const allDeleted = selectedRows.every(row => row.deleted);
  const noneDeleted = selectedRows.every(row => !row.deleted);

  return (
    <Toolbar
      sx={{
        pl: { sm: 2 },
        pr: { xs: 1, sm: 1 },
        ...(numSelected > 0 && {
          bgcolor: 'rgba(25, 118, 210, 0.08)',
        }),
        display: 'flex',
        justifyContent: 'space-between',
      }}
    >
      <Typography
        sx={{ flex: '1 1 100%' }}
        color="inherit"
        variant="subtitle1"
        component="div"
      >
        {numSelected > 0 ? `${numSelected} selected` : 'All Records'}
      </Typography>

      {numSelected > 0 && (
        <Stack direction="row" spacing={1}>
          {numSelected === 1 && (
            <>
              {/* Always show View button */}
              <CustomIconButton
                title="View"
                icon={<Visibility />}
                backgroundColor="primary.light"
                color="primary.dark"
                handleClick={() => selectedId && onView?.(selectedId)}
              />

              {/* Edit button only for non-deleted items */}
              {noneDeleted && (
                <CustomIconButton
                  title="Edit"
                  icon={<Edit />}
                  backgroundColor="warning.light"
                  color="warning.dark"
                  handleClick={() => selectedId && onEdit?.(selectedId)}
                />
              )}

              {/* Restore button only for deleted items */}
              {allDeleted && onRestore && (
                <CustomIconButton
                  title="Restore"
                  icon={<Restore />}
                  backgroundColor="success.light"
                  color="success.dark"
                  handleClick={() => selectedId && onRestore(selectedId.split(','))}
                />
              )}
            </>
          )}
  
          {numSelected > 1 && allDeleted && (
            <>
              {/* Restore button shown when all selected are deleted */}
              {onRestore && (
                <CustomIconButton
                  title={`Restore ${numSelected}`}
                  icon={<Restore />}
                  backgroundColor="success.light"
                  color="success.dark"
                  handleClick={() => selectedId && onRestore(selectedId.split(','))}
                />
              )}
            </>
          )}
  
          {/* Delete button always shown */}
          <CustomIconButton
            title={`Delete ${numSelected > 1 ? `(${numSelected})` : ''}`}
            icon={<Delete />}
            backgroundColor="error.light"
            color="error.dark"
            handleClick={() => selectedId && onDelete?.(selectedId.split(','))}
          />
        </Stack>
      )}
    </Toolbar>
  );
};

const CustomTable: React.FC<CustomTableProps> = ({
  rows,
  columns,
  containerHeight = '100%',
  onView,
  onEdit,
  onDelete,
  onRestore,
  initialSortModel = [],
}) => {
  const { mode } = useContext(ColorModeContext);
  const [selectionModel, setSelectionModel] = useState<GridSelectionModel>([]);
  const [sortModel, setSortModel] = useState<GridSortModel>(initialSortModel);

  // Sorting logic for rows
  const sortedRows = useMemo(() => {
    if (sortModel.length === 0) return rows;

    return [...rows].sort((a, b) => {
      for (const sort of sortModel) {
        const field = sort.field;
        const value1 = a[field];
        const value2 = b[field];

        if (value1 !== value2) {
          return sort.sort === 'asc' 
            ? (value1 > value2 ? 1 : -1)
            : (value1 < value2 ? 1 : -1);
        }
      }
      return 0;
    });
  }, [rows, sortModel]);

  return (
    <Box sx={{ width: '100%', height: '100%' }}>
      <DataGrid
        rows={sortedRows}
        columns={columns}
        checkboxSelection
        disableSelectionOnClick
        autoHeight={false}
        selectionModel={selectionModel}
        onSelectionModelChange={(newSelectionModel) => {
          setSelectionModel(newSelectionModel);
        }}
        components={{
          Toolbar: () => (
            <CustomTableToolbar
              numSelected={selectionModel.length}
              onView={onView}
              onEdit={onEdit}
              onDelete={onDelete}
              onRestore={onRestore}
              selectedId={selectionModel.join(',')}
              rows={rows}
            />
          ),
        }}
        sx={{
          height: containerHeight,
          // Beautiful scrollbar styling
          '& .MuiDataGrid-main': {
            overflow: 'hidden',
            '& ::-webkit-scrollbar': {
              width: '10px',
              height: '10px',
            },
            '& ::-webkit-scrollbar-track': {
              background: mode === 'light' ? '#f1f1f1' : '#2c2c2c',
              borderRadius: '10px',
            },
            '& ::-webkit-scrollbar-thumb': {
              background: mode === 'light' 
                ? 'linear-gradient(45deg, #e0e0e0, #a0a0a0)' 
                : 'linear-gradient(45deg, #4a4a4a, #2c2c2c)',
              borderRadius: '10px',
              transition: 'background 0.3s ease',
            },
            '& ::-webkit-scrollbar-thumb:hover': {
              background: mode === 'light'
                ? 'linear-gradient(45deg, #c0c0c0, #808080)'
                : 'linear-gradient(45deg, #5a5a5a, #3c3c3c)',
            },
          },
          '& .MuiDataGrid-row:hover': {
            backgroundColor: 'rgba(0, 0, 0, 0.04)'
          },
          '& .MuiDataGrid-cell': {
            padding: '8px',
            whiteSpace: 'normal',
            wordWrap: 'break-word'
          },
          '& .MuiDataGrid-columnHeaders': {
            backgroundColor: mode === 'light' ? '#f5f5f5' : '#333333',
            borderBottom: mode === 'light' ? '2px solid #e0e0e0' : '2px solid #444444',
            color: mode === 'light' ? 'inherit' : '#f5f5f5'
          },
          '& .MuiDataGrid-columnHeader': {
            padding: '8px',
            fontWeight: 'bold'
          }
        }}
      />
    </Box>
  );
};

export default CustomTable;