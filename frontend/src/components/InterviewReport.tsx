import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Card,
  Grid,
  LinearProgress,
  Chip,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Button,
  CircularProgress,
  Stack
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Warning,
  Star,
  School,
  Psychology,
  EmojiEvents,
  Home,
  RestartAlt,
  Download
} from '@mui/icons-material';

interface InterviewReportData {
  session_id: string;
  overall_score: number;
  placement_likelihood: string;
  performance_summary: string;
  strengths: string[];
  development_areas: string[];
  detailed_feedback: {
    communication: { score: number; feedback: string };
    technical_knowledge: { score: number; feedback: string };
    problem_solving: { score: number; feedback: string };
    confidence: { score: number; feedback: string };
  };
  skill_breakdown: {
    verbal_communication: number;
    confidence_level: number;
    technical_competency: number;
    problem_solving: number;
    professionalism: number;
  };
  recommendations: string[];
  generated_at: string;
}

const InterviewReport: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const [report, setReport] = useState<InterviewReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (sessionId) {
      generateReport();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const generateReport = async () => {
    try {
      setLoading(true);
      
      // Get interview data from sessionStorage
      const storedData = sessionStorage.getItem(`interview_data_${sessionId}`);
      if (!storedData) {
        throw new Error('No interview data found. Please complete an interview first.');
      }
      
      const interviewData = JSON.parse(storedData);
      console.log('📄 Sending interview data to Gemini AI for comprehensive evaluation:', {
        sessionId: interviewData.sessionId,
        jobRole: interviewData.jobRole,
        totalQuestions: interviewData.totalQuestions,
        responses: interviewData.responses?.length || 0
      });
      
      // Use the new comprehensive AI evaluation endpoint
      const reportData = await generateComprehensiveEvaluation(interviewData);
      
      setReport(reportData);
      console.log('✅ Comprehensive AI evaluation completed:', {
        score: reportData.overall_score,
        likelihood: reportData.placement_likelihood,
        strengths: reportData.strengths?.length || 0,
        recommendations: reportData.recommendations?.length || 0
      });
      
    } catch (err) {
      console.error('Comprehensive evaluation error:', err);
      setError(`Failed to generate interview evaluation: ${err instanceof Error ? err.message : 'AI evaluation service is temporarily unavailable. Please try again later.'}`);
    } finally {
      setLoading(false);
    }
  };
  
  const generateComprehensiveEvaluation = async (interviewData: any): Promise<InterviewReportData> => {
    const aiServiceUrl = 'http://localhost:8001';
    
    // Prepare comprehensive interview data for AI evaluation
    const evaluationData = {
      sessionId: interviewData.sessionId,
      job_role: interviewData.jobRole || 'Software Developer',
      totalQuestions: interviewData.totalQuestions || 5,
      questionsAnswered: interviewData.questionsAnswered || 0,
      questionsSkipped: interviewData.questionsSkipped || 0,
      averageResponseTime: interviewData.averageResponseTime || 60,
      confidence: interviewData.confidence || 0.5,
      engagement: interviewData.engagement || 0.5,
      eyeContact: interviewData.eyeContact || 0.5,
      responses: interviewData.responses || [],  // Full conversation history
      hasResume: interviewData.hasResume || false,
      totalScore: interviewData.totalScore || 0
    };
    
    console.log('🤖 Sending comprehensive evaluation request to Gemini AI:', {
      endpoint: '/generate/comprehensive-evaluation',
      sessionId: evaluationData.sessionId,
      questionsCount: evaluationData.responses.length,
      jobRole: evaluationData.job_role
    });
    
    try {
      const response = await fetch(`${aiServiceUrl}/generate/comprehensive-evaluation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(evaluationData),
        // Increase timeout for AI processing
        signal: AbortSignal.timeout(60000) // 60 seconds timeout
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`AI evaluation service responded with ${response.status}: ${errorText}`);
      }
      
      const reportData = await response.json();
      console.log('✅ Gemini AI evaluation completed successfully:', {
        score: reportData.overall_score,
        likelihood: reportData.placement_likelihood,
        strengths: reportData.strengths?.length || 0,
        recommendations: reportData.recommendations?.length || 0
      });
      
      return reportData;
      
    } catch (error) {
      console.error('\u274c Comprehensive AI evaluation failed:', error);
      
      // Provide more specific error messages based on error type
      if (error instanceof TypeError && error.message.includes('fetch')) {
        throw new Error('Unable to connect to AI evaluation service. Please check your internet connection and try again.');
      } else if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error('AI evaluation is taking longer than expected. Please try again or contact support if the issue persists.');
      } else if (error instanceof Error) {
        // If it's a known error with a message, use it
        throw new Error(`AI evaluation failed: ${error.message}`);
      } else {
        // Generic fallback error
        throw new Error('An unexpected error occurred during AI evaluation. Our team has been notified. Please try again in a few minutes.');
      }
    }
  };
  

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getLikelihoodColor = (likelihood: string) => {
    switch (likelihood.toLowerCase()) {
      case 'high': return 'success';
      case 'medium': return 'warning';
      case 'low': return 'error';
      default: return 'info';
    }
  };

  const handleStartNewInterview = () => {
    navigate('/');
  };

  const handleGoHome = () => {
    navigate('/dashboard');
  };

  if (loading) {
    return (
      <Container maxWidth="md" sx={{ py: 8, textAlign: 'center' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Gemini AI is evaluating your interview...
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Analyzing conversation history, behavioral patterns, and response quality
        </Typography>
        <Typography variant="body2" color="text.secondary">
          This comprehensive evaluation typically takes 10-30 seconds
        </Typography>
      </Container>
    );
  }

  if (error || !report) {
    return (
      <Container maxWidth="md" sx={{ py: 8 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          {error || 'No report data available'}
        </Alert>
        <Button variant="contained" onClick={handleGoHome}>
          Return to Dashboard
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Paper sx={{ 
        p: 4, 
        mb: 4, 
        background: 'linear-gradient(135deg, #047857 0%, #10b981 100%)', 
        color: 'white',
        borderRadius: 3,
        boxShadow: '0 8px 32px rgba(4, 120, 87, 0.15)'
      }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <EmojiEvents sx={{ fontSize: 40, mr: 2 }} />
          <Typography variant="h3" fontWeight={700}>
            Your Growth Report
          </Typography>
        </Box>
        <Typography variant="h6" sx={{ opacity: 0.95, lineHeight: 1.5 }}>
          Personalized insights and supportive guidance for your interview journey
        </Typography>
        <Typography variant="body2" sx={{ opacity: 0.85, mt: 1 }}>
          Generated on {new Date(report.generated_at).toLocaleDateString()}
        </Typography>
      </Paper>

      <Grid container spacing={4}>
        {/* Overall Score Card */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', textAlign: 'center', p: 3 }}>
            <Box sx={{ position: 'relative', display: 'inline-flex', mb: 2 }}>
              <CircularProgress
                variant="determinate"
                value={report.overall_score}
                size={120}
                thickness={4}
                color={getScoreColor(report.overall_score) as any}
              />
              <Box sx={{
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                position: 'absolute',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column'
              }}>
                <Typography variant="h3" fontWeight={700} color={getScoreColor(report.overall_score) + '.main'}>
                  {report.overall_score}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  out of 100
                </Typography>
              </Box>
            </Box>
            <Typography variant="h5" fontWeight={600} gutterBottom>
              Overall Score
            </Typography>
            <Chip 
              label={`${report.placement_likelihood} Placement Likelihood`}
              color={getLikelihoodColor(report.placement_likelihood) as any}
              sx={{ mt: 1, fontWeight: 600 }}
            />
          </Card>
        </Grid>

        {/* Performance Summary */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%', p: 3 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
              <Psychology sx={{ mr: 1 }} />
              Performance Summary
            </Typography>
            <Typography variant="body1" paragraph sx={{ lineHeight: 1.7 }}>
              {report.performance_summary}
            </Typography>
            
            <Grid container spacing={2} sx={{ mt: 2 }}>
              <Grid item xs={6}>
                <Paper sx={{ p: 2, bgcolor: 'success.50', border: '1px solid', borderColor: 'success.200' }}>
                  <Typography variant="h6" color="success.dark" fontWeight={600}>
                    {report.strengths.length} Key Strengths
                  </Typography>
                  <Typography variant="body2" color="success.dark">
                    Areas where you excelled
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={6}>
                <Paper sx={{ p: 2, bgcolor: 'warning.50', border: '1px solid', borderColor: 'warning.200' }}>
                  <Typography variant="h6" color="warning.dark" fontWeight={600}>
                    {report.development_areas.length} Development Areas
                  </Typography>
                  <Typography variant="body2" color="warning.dark">
                    Opportunities for improvement
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </Card>
        </Grid>

        {/* Skill Breakdown */}
        <Grid item xs={12}>
          <Card sx={{ p: 3 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Star sx={{ mr: 1 }} />
              Skill Breakdown
            </Typography>
            <Grid container spacing={3}>
              {Object.entries(report.skill_breakdown).map(([skill, score]) => (
                <Grid item xs={12} sm={6} md={4} key={skill}>
                  <Box>
                    <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 1, textTransform: 'capitalize' }}>
                      {skill.replace(/_/g, ' ')}
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={score}
                      sx={{ height: 10, borderRadius: 5, mb: 1 }}
                      color={getScoreColor(score) as any}
                    />
                    <Typography variant="body2" color="text.secondary" textAlign="center">
                      {score}/100
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Card>
        </Grid>

        {/* Detailed Feedback */}
        <Grid item xs={12}>
          <Card sx={{ p: 3 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <School sx={{ mr: 1 }} />
              Detailed Feedback
            </Typography>
            <Grid container spacing={3}>
              {Object.entries(report.detailed_feedback).map(([category, feedback]) => (
                <Grid item xs={12} md={6} key={category}>
                  <Paper sx={{ p: 3, height: '100%', border: '1px solid', borderColor: 'grey.200' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6" fontWeight={600} sx={{ textTransform: 'capitalize' }}>
                        {category.replace(/_/g, ' ')}
                      </Typography>
                      <Chip 
                        label={`${feedback.score}/100`}
                        color={getScoreColor(feedback.score) as any}
                        size="small"
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {feedback.feedback}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Card>
        </Grid>

        {/* Strengths and Development Areas */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <CheckCircle sx={{ mr: 1, color: 'success.main' }} />
              Key Strengths
            </Typography>
            <List>
              {report.strengths.map((strength, index) => (
                <ListItem key={index} sx={{ px: 0 }}>
                  <ListItemIcon>
                    <TrendingUp color="success" />
                  </ListItemIcon>
                  <ListItemText primary={strength} />
                </ListItem>
              ))}
            </List>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Warning sx={{ mr: 1, color: 'warning.main' }} />
              Development Areas
            </Typography>
            <List>
              {report.development_areas.map((area, index) => (
                <ListItem key={index} sx={{ px: 0 }}>
                  <ListItemIcon>
                    <TrendingDown color="warning" />
                  </ListItemIcon>
                  <ListItemText primary={area} />
                </ListItem>
              ))}
            </List>
          </Card>
        </Grid>

        {/* Recommendations */}
        <Grid item xs={12}>
          <Card sx={{ p: 3 }}>
            <Typography variant="h5" fontWeight={600} gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <EmojiEvents sx={{ mr: 1 }} />
              Recommendations for Improvement
            </Typography>
            <Grid container spacing={2}>
              {report.recommendations.map((recommendation, index) => (
                <Grid item xs={12} md={6} key={index}>
                  <Paper sx={{ 
                    p: 2, 
                    bgcolor: 'primary.50', 
                    border: '1px solid', 
                    borderColor: 'primary.200',
                    display: 'flex',
                    alignItems: 'center'
                  }}>
                    <Typography variant="h6" color="primary.main" sx={{ mr: 2, minWidth: '24px' }}>
                      {index + 1}.
                    </Typography>
                    <Typography variant="body2">
                      {recommendation}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Card>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, textAlign: 'center', bgcolor: 'grey.50' }}>
            <Typography variant="h6" gutterBottom>
              What's Next?
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Use this report to improve your interview skills and increase your chances of success
            </Typography>
            <Stack direction="row" spacing={2} justifyContent="center" sx={{ flexWrap: 'wrap', gap: 2 }}>
              <Button 
                variant="contained" 
                startIcon={<RestartAlt />}
                onClick={handleStartNewInterview}
                size="large"
              >
                Practice Again
              </Button>
              <Button 
                variant="outlined" 
                startIcon={<Home />}
                onClick={handleGoHome}
                size="large"
              >
                Dashboard
              </Button>
              <Button 
                variant="outlined" 
                startIcon={<Download />}
                size="large"
                onClick={() => window.print()}
              >
                Download Report
              </Button>
            </Stack>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default InterviewReport;