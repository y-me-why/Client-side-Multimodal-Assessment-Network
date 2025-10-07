import React from 'react';
import { Container, Typography, Card, CardContent, Box } from '@mui/material';

const Settings: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Settings
        </Typography>
        <Card>
          <CardContent>
            <Typography variant="h6">
              Coming Soon
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Application settings and preferences will be available here.
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default Settings;