import { createTheme } from '@mui/material/styles';

// Modern, supportive, and trustworthy theme
export const modernTheme = createTheme({
  palette: {
    primary: {
      main: '#047857', // Deep Teal - trustworthy and professional
      light: '#10b981',
      dark: '#065f46',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#F59E0B', // Warm Amber - encouraging and energetic
      light: '#FFC700',
      dark: '#D97706',
      contrastText: '#ffffff',
    },
    background: {
      default: '#F0FFF4', // Mint Cream - calm and supportive
      paper: '#ffffff',
    },
    text: {
      primary: '#2F4858', // Dark Slate - readable and professional
      secondary: '#6B7280',
    },
    success: {
      main: '#10b981',
      light: '#6ee7b7',
      dark: '#059669',
    },
    warning: {
      main: '#F59E0B',
      light: '#FCD34D',
      dark: '#D97706',
    },
    error: {
      main: '#EF4444',
      light: '#F87171',
      dark: '#DC2626',
    },
    info: {
      main: '#3B82F6',
      light: '#93C5FD',
      dark: '#1D4ED8',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      lineHeight: 1.2,
      color: '#2F4858',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
      color: '#2F4858',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.4,
      color: '#2F4858',
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
      lineHeight: 1.4,
      color: '#2F4858',
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
      lineHeight: 1.5,
      color: '#2F4858',
    },
    h6: {
      fontSize: '1.1rem',
      fontWeight: 600,
      lineHeight: 1.5,
      color: '#2F4858',
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6,
      color: '#2F4858',
    },
    body2: {
      fontSize: '0.875rem',
      lineHeight: 1.6,
      color: '#6B7280',
    },
    button: {
      fontWeight: 600,
      textTransform: 'none',
    },
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '12px 24px',
          fontSize: '1rem',
          fontWeight: 600,
          textTransform: 'none',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: '0 4px 12px rgba(4, 120, 87, 0.15)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #047857 0%, #10b981 100%)',
          '&:hover': {
            background: 'linear-gradient(135deg, #065f46 0%, #047857 100%)',
          },
        },
        containedSecondary: {
          background: 'linear-gradient(135deg, #F59E0B 0%, #FFC700 100%)',
          color: '#2F4858',
          '&:hover': {
            background: 'linear-gradient(135deg, #D97706 0%, #F59E0B 100%)',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 16,
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
          border: '1px solid rgba(4, 120, 87, 0.08)',
          '&:hover': {
            boxShadow: '0 8px 30px rgba(0, 0, 0, 0.12)',
            transform: 'translateY(-2px)',
            transition: 'all 0.3s ease-in-out',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500,
        },
        colorPrimary: {
          backgroundColor: '#047857',
          color: '#ffffff',
        },
        colorSecondary: {
          backgroundColor: '#F59E0B',
          color: '#2F4858',
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 12,
            '& fieldset': {
              borderColor: 'rgba(4, 120, 87, 0.2)',
            },
            '&:hover fieldset': {
              borderColor: 'rgba(4, 120, 87, 0.4)',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#047857',
            },
          },
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          height: 8,
          backgroundColor: 'rgba(4, 120, 87, 0.1)',
        },
        bar: {
          borderRadius: 8,
          background: 'linear-gradient(90deg, #047857 0%, #10b981 100%)',
        },
      },
    },
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          border: 'none',
        },
        standardSuccess: {
          backgroundColor: 'rgba(16, 185, 129, 0.1)',
          color: '#047857',
        },
        standardInfo: {
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          color: '#1D4ED8',
        },
        standardWarning: {
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          color: '#D97706',
        },
      },
    },
  },
});

export default modernTheme;