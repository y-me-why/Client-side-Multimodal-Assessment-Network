import React from 'react';
import { Container, Typography, Card, CardContent, Box, Stack, Chip } from '@mui/material';
import { TrendingUp, Psychology, Assessment } from '@mui/icons-material';

const Dashboard: React.FC = () => {
  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 6 }}>
        <Typography variant="h3" gutterBottom fontWeight={700} color="text.primary">
          Your Learning Journey
        </Typography>
        <Typography variant="h6" color="text.secondary" paragraph sx={{ mb: 4 }}>
          Track your progress and celebrate your growth
        </Typography>
        
        <Card sx={{ 
          background: 'linear-gradient(135deg, rgba(240, 255, 244, 0.3) 0%, rgba(255, 255, 255, 1) 100%)',
          border: '1px solid rgba(4, 120, 87, 0.1)',
          borderRadius: 4,
          p: 2
        }}>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <Psychology sx={{ fontSize: 80, color: 'primary.main', mb: 3 }} />
            <Typography variant="h4" gutterBottom fontWeight={600} color="text.primary">
              Coming Soon
            </Typography>
            <Typography variant="h6" color="text.secondary" paragraph sx={{ maxWidth: 600, mx: 'auto', lineHeight: 1.6 }}>
              Your personalized dashboard with interview history, progress analytics, and growth insights will be available here.
            </Typography>
            
            <Stack direction="row" spacing={2} justifyContent="center" sx={{ mt: 4, flexWrap: 'wrap', gap: 1 }}>
              <Chip 
                icon={<TrendingUp />}
                label="Progress Tracking" 
                sx={{ 
                  backgroundColor: 'primary.main',
                  color: 'white',
                  fontWeight: 600,
                }}
              />
              <Chip 
                icon={<Assessment />}
                label="Performance Analytics" 
                sx={{ 
                  backgroundColor: 'secondary.main',
                  color: 'text.primary',
                  fontWeight: 600,
                }}
              />
              <Chip 
                icon={<Psychology />}
                label="Growth Insights" 
                sx={{ 
                  backgroundColor: 'success.main',
                  color: 'white',
                  fontWeight: 600,
                }}
              />
            </Stack>
          </CardContent>
        </Card>
      </Box>
    </Container>
  );
};

export default Dashboard;