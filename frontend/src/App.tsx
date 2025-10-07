import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';
import './App.css';

// Components
import Home from './components/Home';
import EnhancedInterviewSession from './components/EnhancedInterviewSession';
import InterviewReport from './components/InterviewReport';
import Dashboard from './components/Dashboard';
import Settings from './components/Settings';
import NavigationHeader from './components/NavigationHeader';

// Modern theme with supportive and trustworthy design
import modernTheme from './theme';

const AppContent: React.FC = () => {
  const location = useLocation();
  const isHomePage = location.pathname === '/';
  const isInterviewPage = location.pathname.includes('/interview/');
  
  return (
    <Box sx={{ minHeight: '100vh', backgroundColor: 'background.default' }}>
      {!isHomePage && (
        <NavigationHeader
          showSessionInfo={isInterviewPage}
          currentSession={isInterviewPage ? location.pathname.split('/')[2] : undefined}
        />
      )}
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/interview/:sessionId" element={<EnhancedInterviewSession />} />
        <Route path="/report/:sessionId" element={<InterviewReport />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Box>
  );
};

function App() {
  return (
    <ThemeProvider theme={modernTheme}>
      <CssBaseline />
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}

export default App;
