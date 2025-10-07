import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Chip,
  Divider,
} from '@mui/material';
import {
  Home,
  Dashboard,
  Settings,
  AccountCircle,
  Psychology,
  Assessment,
  TrendingUp,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { NavigationHeaderProps } from '../types';

const NavigationHeader: React.FC<NavigationHeaderProps> = ({
  currentSession,
  sessionScore,
  showSessionInfo = false,
}) => {
  const navigate = useNavigate();
  const location = useLocation();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <AppBar 
      position="sticky" 
      elevation={0}
      sx={{
        background: 'linear-gradient(135deg, #047857 0%, #10b981 100%)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.15)',
        boxShadow: '0 4px 20px rgba(4, 120, 87, 0.15)',
      }}
    >
      <Toolbar sx={{ px: { xs: 2, sm: 4 } }}>
        {/* Logo and Brand */}
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            cursor: 'pointer',
            transition: 'transform 0.2s ease',
            '&:hover': {
              transform: 'scale(1.05)',
            },
          }}
          onClick={() => navigate('/')}
        >
          <Psychology sx={{ fontSize: 32, mr: 1, color: 'white' }} />
          <Typography
            variant="h5"
            component="div"
            sx={{
              fontWeight: 700,
              color: 'white',
              display: { xs: 'none', sm: 'block' },
              letterSpacing: '-0.02em',
            }}
          >
            AI Interview Prep
          </Typography>
          <Typography
            variant="h6"
            component="div"
            sx={{
              fontWeight: 700,
              color: 'white',
              display: { xs: 'block', sm: 'none' },
              letterSpacing: '-0.02em',
            }}
          >
            AI Prep
          </Typography>
        </Box>

        {/* Session Info */}
        {showSessionInfo && currentSession && (
          <Box sx={{ ml: 4, display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1 }}>
            <Chip
              label={`Session: ${currentSession.slice(-8)}`}
              size="small"
              sx={{
                backgroundColor: 'rgba(255,255,255,0.2)',
                color: 'white',
                fontWeight: 'bold',
              }}
            />
            {sessionScore !== undefined && (
              <Chip
                icon={<TrendingUp sx={{ color: 'white !important' }} />}
                label={`Score: ${Math.round(sessionScore)}`}
                size="small"
                sx={{
                  backgroundColor: sessionScore >= 70 ? 'rgba(76, 175, 80, 0.8)' : 'rgba(255, 152, 0, 0.8)',
                  color: 'white',
                  fontWeight: 'bold',
                }}
              />
            )}
          </Box>
        )}

        {/* Spacer */}
        <Box sx={{ flexGrow: 1 }} />

        {/* Navigation Buttons */}
        <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1 }}>
          <Button
            startIcon={<Home />}
            onClick={() => navigate('/')}
            sx={{
              color: 'white',
              fontWeight: isActive('/') ? 'bold' : 'normal',
              backgroundColor: isActive('/') ? 'rgba(255,255,255,0.2)' : 'transparent',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,0.1)',
              },
            }}
          >
            Home
          </Button>

          <Button
            startIcon={<Dashboard />}
            onClick={() => navigate('/dashboard')}
            sx={{
              color: 'white',
              fontWeight: isActive('/dashboard') ? 'bold' : 'normal',
              backgroundColor: isActive('/dashboard') ? 'rgba(255,255,255,0.2)' : 'transparent',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,0.1)',
              },
            }}
          >
            Dashboard
          </Button>

          <Button
            startIcon={<Assessment />}
            onClick={() => navigate('/analytics')}
            sx={{
              color: 'white',
              fontWeight: isActive('/analytics') ? 'bold' : 'normal',
              backgroundColor: isActive('/analytics') ? 'rgba(255,255,255,0.2)' : 'transparent',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,0.1)',
              },
            }}
          >
            Analytics
          </Button>
        </Box>

        {/* User Menu */}
        <Box sx={{ ml: 2 }}>
          <IconButton
            size="large"
            onClick={handleMenuOpen}
            sx={{
              color: 'white',
              backgroundColor: 'rgba(255,255,255,0.1)',
              '&:hover': {
                backgroundColor: 'rgba(255,255,255,0.2)',
              },
            }}
          >
            <AccountCircle />
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            PaperProps={{
              sx: {
                mt: 1,
                minWidth: 200,
                borderRadius: 2,
                boxShadow: '0 8px 25px rgba(0,0,0,0.15)',
              },
            }}
          >
            <Box sx={{ px: 2, py: 1, borderBottom: '1px solid rgba(0,0,0,0.1)' }}>
              <Typography variant="subtitle1" fontWeight="bold">
                Guest User
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Interview Practice Session
              </Typography>
            </Box>
            
            <MenuItem onClick={() => { navigate('/dashboard'); handleMenuClose(); }}>
              <Dashboard sx={{ mr: 2 }} />
              Dashboard
            </MenuItem>
            <MenuItem onClick={() => { navigate('/settings'); handleMenuClose(); }}>
              <Settings sx={{ mr: 2 }} />
              Settings
            </MenuItem>
            <Divider />
            <MenuItem
              onClick={() => {
                // Clear session data
                sessionStorage.clear();
                localStorage.clear();
                navigate('/');
                handleMenuClose();
              }}
              sx={{ color: 'error.main' }}
            >
              <Home sx={{ mr: 2 }} />
              New Session
            </MenuItem>
          </Menu>
        </Box>
      </Toolbar>

      {/* Mobile Session Info */}
      {showSessionInfo && currentSession && (
        <Box
          sx={{
            display: { xs: 'flex', md: 'none' },
            justifyContent: 'center',
            pb: 1,
            gap: 1,
          }}
        >
          <Chip
            label={`Session: ${currentSession.slice(-8)}`}
            size="small"
            sx={{
              backgroundColor: 'rgba(255,255,255,0.2)',
              color: 'white',
              fontSize: '0.7rem',
            }}
          />
          {sessionScore !== undefined && (
            <Chip
              label={`Score: ${Math.round(sessionScore)}`}
              size="small"
              sx={{
                backgroundColor: sessionScore >= 70 ? 'rgba(76, 175, 80, 0.8)' : 'rgba(255, 152, 0, 0.8)',
                color: 'white',
                fontSize: '0.7rem',
              }}
            />
          )}
        </Box>
      )}
    </AppBar>
  );
};

export default NavigationHeader;