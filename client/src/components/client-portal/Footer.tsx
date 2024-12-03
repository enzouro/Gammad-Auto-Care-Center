import { Box, Card, CardContent, Grid, Typography } from '@mui/material'
import PhoneIcon from '@mui/icons-material/Phone';
import EmailIcon from '@mui/icons-material/Email';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

const Footer = () => {
  return (
    <Box 
      component="footer"
      sx={{
        width: "100%",
        backgroundColor: 'background.paper',
        mt: 'auto', // This pushes the footer to the bottom
      }}
    >
      <Card sx={{ py: 1 }}>
        <CardContent sx={{ py: '8px !important' }}>
          <Grid container spacing={2}>
            {/* Title Section */}
            <Grid item xs={12} md={3}>
              <Box sx={{ textAlign: { xs: 'center', md: 'left' }, mb: { xs: 1, md: 0 } }}>
                <Typography variant="h6" gutterBottom sx={{ mb: 0 }}>
                  Need Help?
                </Typography>
                <Typography variant="subtitle2" color="text.secondary">
                  Contact our support team
                </Typography>
              </Box>
            </Grid>

            {/* Contact Information */}
            <Grid item xs={12} md={9}>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                    <PhoneIcon color="primary" sx={{ fontSize: '1.2rem', color: "#fee140"  }} />
                    <Typography variant="body2">
                      (555) 123-4567
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                    <EmailIcon  sx={{ fontSize: '1.2rem', color: "#fee140" }} />
                    <Typography variant="body2">
                      support@autorepair.com
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                    <AccessTimeIcon sx={{ fontSize: '1.2rem', color: "#fee140"  }} />
                    <Typography variant="body2">
                      Mon-Fri 8AM-6PM
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  )
}

export default Footer