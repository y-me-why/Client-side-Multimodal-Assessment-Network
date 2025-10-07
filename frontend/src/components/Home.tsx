import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Card,
  Grid,
  TextField,
  Stack,
  Chip,
  Avatar,
  Fade,
  Zoom,
} from '@mui/material';
import {
  Psychology,
  VideoCall,
  Mic,
  Assessment,
  WorkOutline,
  CheckCircle,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import ResumeUpload from './ResumeUpload';
import NavigationHeader from './NavigationHeader';

const Home: React.FC = () => {
  const navigate = useNavigate();
  const [jobRole, setJobRole] = useState('Software Developer');
  const [resumeData, setResumeData] = useState<any>(null);

  const startInterview = () => {
    const sessionId = uuidv4();
    const params = new URLSearchParams({
      role: jobRole,
      hasResume: resumeData ? 'true' : 'false'
    });
    if (resumeData) {
      // Store resume data in sessionStorage for the interview session
      sessionStorage.setItem(`resume_${sessionId}`, JSON.stringify(resumeData));
      // Also store with generic key as backup
      sessionStorage.setItem('resumeData', JSON.stringify(resumeData));
      console.log('🎯 Stored resume data for session:', sessionId, {
        skills: resumeData.skills?.length || 0,
        experience: resumeData.experience?.length || 0,
        hasText: !!resumeData.text
      });
    }
    navigate(`/interview/${sessionId}?${params.toString()}`);
  };

  const handleResumeAnalyzed = (data: any) => {
    setResumeData(data);
  };

  return (
    <>
      <NavigationHeader />
      <Box sx={{ minHeight: 'calc(100vh - 64px)' }}>
        {/* Hero Section */}
        <Box
          sx={{
            background: 'linear-gradient(135deg, #047857 0%, #10b981 100%)',
            position: 'relative',
            overflow: 'hidden',
            py: { xs: 8, md: 12 },
          }}
        >
          <Container maxWidth="lg">
            <Grid container spacing={4} alignItems="center">
              <Grid item xs={12} md={6}>
                <Fade in timeout={1000}>
                  <Box>
                    <Typography
                      variant="h2"
                      component="h1"
                      sx={{
                        color: 'white',
                        fontWeight: 800,
                        mb: 2,
                        fontSize: { xs: '2.5rem', md: '3.5rem' },
                        lineHeight: 1.2,
                      }}
                    >
                      Build Confidence.
                      <Box component="span" sx={{ color: '#FFC700' }}>
                        {' '}Excel in Interviews.
                      </Box>
                    </Typography>
                    <Typography
                      variant="h5"
                      sx={{
                        color: 'rgba(255,255,255,0.95)',
                        mb: 4,
                        fontWeight: 400,
                        lineHeight: 1.5,
                      }}
                    >
                      Practice in a safe, supportive environment with our AI interview coach. Get personalized feedback and build the confidence you need to succeed.
                    </Typography>
                    
                    <Stack direction="row" spacing={2} sx={{ mb: 4, flexWrap: 'wrap', gap: 1 }}>
                      <Chip
                        icon={<Psychology />}
                        label="AI-Powered Coach"
                        sx={{
                          backgroundColor: 'rgba(255,199,0,0.2)',
                          color: 'white',
                          fontWeight: 600,
                          border: '1px solid rgba(255,199,0,0.3)',
                        }}
                      />
                      <Chip
                        icon={<VideoCall />}
                        label="Safe Practice Space"
                        sx={{
                          backgroundColor: 'rgba(255,255,255,0.15)',
                          color: 'white',
                          fontWeight: 600,
                          border: '1px solid rgba(255,255,255,0.2)',
                        }}
                      />
                      <Chip
                        icon={<Assessment />}
                        label="Personalized Growth"
                        sx={{
                          backgroundColor: 'rgba(255,255,255,0.15)',
                          color: 'white',
                          fontWeight: 600,
                          border: '1px solid rgba(255,255,255,0.2)',
                        }}
                      />
                    </Stack>
                  </Box>
                </Fade>
              </Grid>
              
              <Grid item xs={12} md={6}>
                <Zoom in timeout={1200}>
                  <Card
                    sx={{
                      background: 'rgba(255,255,255,0.98)',
                      backdropFilter: 'blur(20px)',
                      borderRadius: 4,
                      p: 4,
                      boxShadow: '0 20px 60px rgba(4,120,87,0.15)',
                      border: '1px solid rgba(4,120,87,0.1)',
                    }}
                  >
                    <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', fontWeight: 700, color: 'text.primary' }}>
                      Ready to Practice?
                    </Typography>
                    
                    <ResumeUpload onResumeAnalyzed={handleResumeAnalyzed} />
                    
                    <Box sx={{ mt: 3 }}>
                      <TextField
                        fullWidth
                        label="Job Role"
                        value={jobRole}
                        onChange={(e) => setJobRole(e.target.value)}
                        variant="outlined"
                        helperText="Enter the position you're interviewing for"
                        sx={{ mb: 3 }}
                      />
                      
                      <Button
                        fullWidth
                        variant="contained"
                        size="large"
                        onClick={startInterview}
                        startIcon={<WorkOutline />}
                        sx={{
                          py: 2.5,
                          fontSize: '1.1rem',
                          fontWeight: 700,
                          background: 'linear-gradient(135deg, #F59E0B 0%, #FFC700 100%)',
                          color: '#2F4858',
                          '&:hover': {
                            background: 'linear-gradient(135deg, #D97706 0%, #F59E0B 100%)',
                            transform: 'translateY(-1px)',
                          },
                        }}
                      >
                        Start Your Journey
                      </Button>
                    </Box>
                  </Card>
                </Zoom>
              </Grid>
            </Grid>
          </Container>
        </Box>

        {/* Features Section */}
        <Container maxWidth="lg" sx={{ py: 8 }}>
          <Typography
            variant="h3"
            component="h2"
            sx={{
              textAlign: 'center',
              fontWeight: 700,
              mb: 2,
              color: 'text.primary',
            }}
          >
            Why Choose AI Interview Prep?
          </Typography>
          <Typography
            variant="h6"
            sx={{
              textAlign: 'center',
              color: 'text.secondary',
              mb: 6,
              maxWidth: '650px',
              mx: 'auto',
              lineHeight: 1.6,
            }}
          >
            Your supportive AI interview coach combines advanced technology with personalized guidance to help you build confidence and succeed in interviews.
          </Typography>

          <Grid container spacing={4}>
            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
                <Avatar
                  sx={{
                    width: 80,
                    height: 80,
                    mx: 'auto',
                    mb: 3,
                    background: 'linear-gradient(135deg, #047857 0%, #10b981 100%)',
                  }}
                >
                  <VideoCall sx={{ fontSize: 40 }} />
                </Avatar>
                <Typography variant="h5" gutterBottom fontWeight={600} color="text.primary">
                  Supportive Video Analysis
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph sx={{ lineHeight: 1.6 }}>
                  Gentle computer vision technology analyzes your presence and body language, offering supportive feedback to help you build natural confidence and authentic engagement.
                </Typography>
                <Stack direction="row" spacing={1} justifyContent="center">
                  <Chip icon={<CheckCircle />} label="Confidence Tracking" size="small" />
                  <Chip icon={<CheckCircle />} label="Eye Contact" size="small" />
                </Stack>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
                <Avatar
                  sx={{
                    width: 80,
                    height: 80,
                    mx: 'auto',
                    mb: 3,
                    background: 'linear-gradient(135deg, #047857 0%, #10b981 100%)',
                  }}
                >
                  <Mic sx={{ fontSize: 40 }} />
                </Avatar>
                <Typography variant="h5" gutterBottom fontWeight={600} color="text.primary">
                  Voice Coaching & Clarity
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph sx={{ lineHeight: 1.6 }}>
                  Smart speech analysis helps you find your authentic voice, improving clarity, pace, and tone to communicate with confidence and natural authority.
                </Typography>
                <Stack direction="row" spacing={1} justifyContent="center">
                  <Chip icon={<CheckCircle />} label="Speech Clarity" size="small" />
                  <Chip icon={<CheckCircle />} label="Tone Analysis" size="small" />
                </Stack>
              </Card>
            </Grid>

            <Grid item xs={12} md={4}>
              <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
                <Avatar
                  sx={{
                    width: 80,
                    height: 80,
                    mx: 'auto',
                    mb: 3,
                    background: 'linear-gradient(135deg, #047857 0%, #10b981 100%)',
                  }}
                >
                  <Psychology sx={{ fontSize: 40 }} />
                </Avatar>
                <Typography variant="h5" gutterBottom fontWeight={600} color="text.primary">
                  Personalized AI Questions
                </Typography>
                <Typography variant="body1" color="text.secondary" paragraph sx={{ lineHeight: 1.6 }}>
                  Thoughtful AI creates personalized interview scenarios tailored to your background and goals, helping you practice for real-world situations.
                </Typography>
                <Stack direction="row" spacing={1} justifyContent="center">
                  <Chip icon={<CheckCircle />} label="Personalized" size="small" />
                  <Chip icon={<CheckCircle />} label="Role-specific" size="small" />
                </Stack>
              </Card>
            </Grid>
          </Grid>
        </Container>

        {/* Values Section */}
        <Box
          sx={{
            background: 'linear-gradient(135deg, #F0FFF4 0%, #ECFDF5 100%)',
            py: 8,
          }}
        >
          <Container maxWidth="lg">
            <Grid container spacing={6} textAlign="center">
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="h3" fontWeight={700} sx={{ color: '#047857', mb: 1 }}>
                  Safe
                </Typography>
                <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
                  Practice Environment
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="h3" fontWeight={700} sx={{ color: '#047857', mb: 1 }}>
                  Real-time
                </Typography>
                <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
                  Supportive Feedback
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="h3" fontWeight={700} sx={{ color: '#047857', mb: 1 }}>
                  Personalized
                </Typography>
                <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
                  Learning Journey
                </Typography>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Typography variant="h3" fontWeight={700} sx={{ color: '#047857', mb: 1 }}>
                  24/7
                </Typography>
                <Typography variant="h6" color="text.secondary" sx={{ fontWeight: 500 }}>
                  Always Available
                </Typography>
              </Grid>
            </Grid>
          </Container>
        </Box>
      </Box>
    </>
  );
};

export default Home;