import React, { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
  Alert,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  ExpandMore,
  Mic,
  MicOff,
  SkipNext,
  Stop,
  Assessment,
  History,
} from '@mui/icons-material';
import Webcam from 'react-webcam';
import Avatar3D from './Avatar3D';
import {
  ConversationEntry,
  InterviewScore,
} from '../types';

const InterviewSession: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  
  const jobRole = searchParams.get('role') || 'Software Developer';
  const hasResume = searchParams.get('hasResume') === 'true';
  
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [analysisData, setAnalysisData] = useState({
    confidence: 0.75,
    engagement: 0.82,
    eyeContact: 0.68,
  });
  const [resumeData, setResumeData] = useState<any>(null);
  const [avatarSpeaking, setAvatarSpeaking] = useState(false);
  const [avatarText, setAvatarText] = useState('');
  
  // New state for enhanced features
  const [conversationLog, setConversationLog] = useState<ConversationEntry[]>([]);
  const [currentQuestionNumber, setCurrentQuestionNumber] = useState(0);
  const [interviewScore, setInterviewScore] = useState<InterviewScore>({
    totalScore: 100,
    questionsAnswered: 0,
    questionsSkipped: 0,
    averageResponseTime: 0,
    confidence: 0,
    engagement: 0,
    eyeContact: 0,
    communication: 0,
  });
  const [questionStartTime, setQuestionStartTime] = useState<Date | null>(null);
  const [showConversationLog, setShowConversationLog] = useState(false);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [interviewEnded, setInterviewEnded] = useState(false);
  const [showFeedbackReport, setShowFeedbackReport] = useState(false);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  
  const MAX_QUESTIONS = 5;

  useEffect(() => {
    // Load resume data if available
    if (hasResume && sessionId) {
      const storedResume = sessionStorage.getItem(`resume_${sessionId}`);
      if (storedResume) {
        setResumeData(JSON.parse(storedResume));
      }
    }
  }, [hasResume, sessionId]);


  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognitionConstructor = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognitionConstructor) {
        recognitionRef.current = new SpeechRecognitionConstructor();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setCurrentAnswer(prev => prev + ' ' + finalTranscript);
        }
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
      };
      }
    }
  }, []);

  useEffect(() => {
    if (!interviewStarted) return;
    
    // Generate initial question based on resume or general category
    generateNewQuestion();
  }, [jobRole, resumeData, interviewStarted]);
  
  // Save conversation log to localStorage
  useEffect(() => {
    if (sessionId && conversationLog.length > 0) {
      localStorage.setItem(`interview_log_${sessionId}`, JSON.stringify(conversationLog));
    }
  }, [conversationLog, sessionId]);
  
  // Load existing conversation log
  useEffect(() => {
    if (sessionId) {
      const savedLog = localStorage.getItem(`interview_log_${sessionId}`);
      if (savedLog) {
        setConversationLog(JSON.parse(savedLog));
      }
    }
  }, [sessionId]);

  const startInterview = () => {
    setInterviewStarted(true);
    setQuestionStartTime(new Date());
  };
  
  const endInterview = () => {
    setInterviewEnded(true);
    setIsRecording(false);
    setIsListening(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    
    // Calculate final scores
    const finalScore = {
      ...interviewScore,
      confidence: analysisData.confidence,
      engagement: analysisData.engagement,
      eyeContact: analysisData.eyeContact,
    };
    setInterviewScore(finalScore);
    
    // Store interview data for potential report generation
    const interviewData = {
      sessionId: sessionId!,
      jobRole,
      totalQuestions: currentQuestionNumber,
      questionsAnswered: finalScore.questionsAnswered,
      questionsSkipped: finalScore.questionsSkipped,
      averageResponseTime: finalScore.averageResponseTime,
      totalScore: finalScore.totalScore,
      confidence: finalScore.confidence,
      engagement: finalScore.engagement,
      eyeContact: finalScore.eyeContact,
      hasResume: !!resumeData,
      responses: conversationLog.filter(entry => entry.type === 'answer').map(entry => ({
        question: conversationLog[conversationLog.indexOf(entry) - 1]?.content || 'Question not found',
        response: entry.content,
        score: entry.score || 0,
        evaluation: entry.evaluation || 'No evaluation'
      }))
    };
    
    // Store the interview data in sessionStorage
    sessionStorage.setItem(`interview_data_${sessionId}`, JSON.stringify(interviewData));
    
    setShowFeedbackReport(true);
  };
  
  const addToConversationLog = (entry: Omit<ConversationEntry, 'id' | 'timestamp'>) => {
    const newEntry: ConversationEntry = {
      ...entry,
      id: Date.now().toString(),
      timestamp: new Date(),
    };
    setConversationLog(prev => [...prev, newEntry]);
  };
  
  const evaluateAnswer = (answer: string): { score: number; evaluation: string } => {
    // Improved evaluation logic focusing on key elements
    const wordCount = answer.trim().split(' ').length;
    const hasStar = answer.includes('STAR') || 
                   (answer.includes('situation') && 
                    answer.includes('task') && 
                    answer.includes('action') && 
                    answer.includes('result'));
                   
    const hasSpecifics = /\d+|specific|example|instance|achieved|improved|increased|decreased/i.test(answer);
    const hasKeywords = /skill|experience|project|challenge|learn|solve|team|collaborate/i.test(answer);
    
    // Start with a lower base score
    let score = 60;
    
    // More meaningful scoring criteria
    if (wordCount > 30) score += 10; // Reasonable length
    if (wordCount > 60) score += 10; // Good elaboration
    if (hasStar) score += 20; // Using STAR method
    if (hasSpecifics) score += 15; // Including specifics
    if (hasKeywords) score += 10; // Using relevant keywords
    
    // Cap at 100
    score = Math.min(score, 100);
    
    // More constructive feedback
    let evaluation = '';
    if (score >= 90) {
      evaluation = 'Excellent response using STAR method with specific examples.';
    } else if (score >= 80) {
      evaluation = 'Good response, but should better highlight specific achievements.';
    } else if (score >= 70) {
      evaluation = 'Adequate response, try using the STAR method (Situation, Task, Action, Result).';
    } else {
      evaluation = 'Your answer should be more specific, use the STAR method and include measurable results.';
    }
    
    return { score, evaluation };
  };

  const startRecording = () => {
    setIsRecording(true);
    setIsListening(true);
    if (recognitionRef.current) {
      recognitionRef.current.start();
    }
    setCurrentAnswer('');
  };

  const stopRecording = () => {
    setIsRecording(false);
    setIsListening(false);
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    
    if (currentAnswer.trim()) {
      const responseTime = questionStartTime ? (new Date().getTime() - questionStartTime.getTime()) / 1000 : 0;
      
      // For now, just use local evaluation
      // TODO: Add AI evaluation service
      
      // Use local evaluation as fallback or immediate feedback
      const { score, evaluation } = evaluateAnswer(currentAnswer);
      
      // Add answer to conversation log
      addToConversationLog({
        type: 'answer',
        content: currentAnswer,
        score,
        evaluation,
        isAnswered: true,
      });
      
      // Update interview score
      setInterviewScore(prev => ({
        ...prev,
        totalScore: prev.totalScore + score - 70, // Add/subtract from base score
        questionsAnswered: prev.questionsAnswered + 1,
        averageResponseTime: (prev.averageResponseTime * prev.questionsAnswered + responseTime) / (prev.questionsAnswered + 1),
      }));
      
      setCurrentAnswer('');
      
      // Check if interview should end after 5 questions
      const totalQuestions = interviewScore.questionsAnswered + 1 + interviewScore.questionsSkipped;
      if (totalQuestions >= MAX_QUESTIONS) {
        setTimeout(() => {
          endInterview();
        }, 2000);
      } else {
        // Generate next question after a brief delay
        setTimeout(() => {
          generateNewQuestion();
        }, 2000);
      }
    }
  };
  
  const skipQuestion = () => {
    // Deduct points for skipping
    const skipPenalty = 15;
    setInterviewScore(prev => ({
      ...prev,
      totalScore: Math.max(0, prev.totalScore - skipPenalty),
      questionsSkipped: prev.questionsSkipped + 1,
    }));
    
    // Add skip entry to conversation log
    addToConversationLog({
      type: 'skipped',
      content: currentQuestion,
      isAnswered: false,
    });
    
    // Check if interview should end after 5 questions
    const totalQuestions = interviewScore.questionsAnswered + interviewScore.questionsSkipped + 1;
    if (totalQuestions >= MAX_QUESTIONS) {
      setTimeout(() => {
        endInterview();
      }, 1000);
    } else {
      // Generate new question
      generateNewQuestion();
    }
  };

  const generateNewQuestion = async () => {
    // Don't generate if we've reached max questions
    if (currentQuestionNumber >= MAX_QUESTIONS) {
      endInterview();
      return;
    }
    
    setCurrentQuestionNumber(prev => prev + 1);
    
    console.log('Generating AI question...', {
      currentQuestionNumber: currentQuestionNumber + 1,
      jobRole,
      resumeData,
      sessionId
    });
    
    try {
      // Get AI-generated question with retry
      const aiQuestion = await getAIQuestionWithRetry();
      
      console.log('✅ AI Question successfully generated:', aiQuestion.substring(0, 100) + '...');
      
      setCurrentQuestion(aiQuestion);
      setAvatarText(aiQuestion);
      setAvatarSpeaking(true);
      setQuestionStartTime(new Date());
      
      // Add to conversation log
      addToConversationLog({
        type: 'question',
        content: aiQuestion,
        isAnswered: false,
      });
      
      // Stop avatar speaking
      setTimeout(() => setAvatarSpeaking(false), aiQuestion.length * 80);
      
    } catch (error) {
      console.error('❌ Failed to generate AI question:', error);
      
      // Show error to user instead of fallback
      const errorQuestion = `I apologize, but I'm having trouble connecting to the AI service to generate personalized questions. Please refresh the page and try again. Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
      
      setCurrentQuestion(errorQuestion);
      setAvatarText('Technical difficulties occurred');
      setAvatarSpeaking(false);
      setQuestionStartTime(new Date());
      
      addToConversationLog({
        type: 'question',
        content: errorQuestion,
        isAnswered: false,
      });
    }
  };
  
  const getAIQuestionWithRetry = async (): Promise<string> => {
    const maxRetries = 3;
    let lastError: any = null;
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const categories = ['behavioral', 'technical', 'situational'];
        const category = categories[(currentQuestionNumber - 1) % categories.length];
        const difficulty = currentQuestionNumber <= 2 ? 'easy' : currentQuestionNumber <= 4 ? 'medium' : 'hard';
        
        console.log(`AI Question Attempt ${attempt}:`, {
          category,
          difficulty,
          jobRole,
          hasResumeData: !!resumeData,
          resumeSkills: resumeData?.skills?.slice(0, 3)
        });
        
        const requestBody = {
          job_role: jobRole,
          difficulty: difficulty,
          session_id: sessionId || `session-${Date.now()}`,
          category: category,
          resume_data: resumeData || null
        };
        
        console.log('Sending request to AI service:', requestBody);
        
        const response = await fetch('http://localhost:8001/generate/question', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(requestBody)
        });
        
        console.log('AI Service Response Status:', response.status);
        
        if (response.ok) {
          const data = await response.json();
          console.log('AI Question received:', data.question?.substring(0, 100) + '...');
          
          if (data.question && data.question.length > 10) {
            return data.question;
          } else {
            throw new Error('Invalid question received from AI service');
          }
        } else {
          const errorText = await response.text();
          throw new Error(`AI service responded with ${response.status}: ${errorText}`);
        }
        
      } catch (error) {
        console.error(`AI API attempt ${attempt} failed:`, error);
        lastError = error;
        
        if (attempt < maxRetries) {
          console.log(`Retrying in ${attempt} seconds...`);
          await new Promise(resolve => setTimeout(resolve, attempt * 1000));
        }
      }
    }
    
    console.error('All AI API attempts failed, throwing error to force reload');
    throw new Error(`AI service unavailable after ${maxRetries} attempts. Last error: ${lastError?.message}`);
  };
  
  
  
  const FeedbackReport = () => {
    const overallGrade = 
      interviewScore.totalScore >= 90 ? 'A' :
      interviewScore.totalScore >= 80 ? 'B' :
      interviewScore.totalScore >= 70 ? 'C' :
      interviewScore.totalScore >= 60 ? 'D' : 'F';
      
    // Calculate improved score based on answered questions
    const answeredScore = interviewScore.questionsAnswered > 0 
      ? Math.round((interviewScore.totalScore / interviewScore.questionsAnswered) * 20)
      : 0;
      
    const resumeGaps = resumeData ? [
      resumeData.skills.length < 3 && 'Consider adding more relevant technical skills to your resume.',
      resumeData.projects.length < 2 && 'Include more project examples that showcase your abilities.',
      resumeData.experience.length < 2 && 'Highlight more work experiences with specific achievements.'
    ].filter(Boolean) : [];
    
    const answerImprovements = [
      'Use the STAR method: Situation, Task, Action, Result for behavioral questions.',
      'Include specific metrics and numbers to quantify your achievements.',
      'Practice explaining technical concepts in simple terms.',
      'Prepare examples that demonstrate leadership and problem-solving skills.'
    ];
    
    const recommendations = [...resumeGaps, ...answerImprovements];
      
    return (
      <Dialog open={showFeedbackReport} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box display="flex" alignItems="center" gap={2}>
            <Assessment color="primary" />
            Interview Performance Report
          </Box>
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            {/* Overall Score */}
            <Grid item xs={12} md={4}>
              <Card sx={{ textAlign: 'center', p: 2 }}>
                <Typography variant="h2" color="primary" gutterBottom>
                  {overallGrade}
                </Typography>
                <Typography variant="h6" gutterBottom>
                  Overall Grade
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Score: {Math.round(interviewScore.totalScore)}/100
                </Typography>
              </Card>
            </Grid>
            
            {/* Statistics */}
            <Grid item xs={12} md={8}>
              <Card sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Interview Statistics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Questions Answered
                    </Typography>
                    <Typography variant="h6">
                      {interviewScore.questionsAnswered}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Questions Skipped
                    </Typography>
                    <Typography variant="h6" color={interviewScore.questionsSkipped > 0 ? 'error' : 'primary'}>
                      {interviewScore.questionsSkipped}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Avg. Response Time
                    </Typography>
                    <Typography variant="h6">
                      {Math.round(interviewScore.averageResponseTime)}s
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Total Questions
                    </Typography>
                    <Typography variant="h6">
                      {interviewScore.questionsAnswered + interviewScore.questionsSkipped}
                    </Typography>
                  </Grid>
                </Grid>
              </Card>
            </Grid>
            
            {/* Analysis Scores */}
            <Grid item xs={12}>
              <Card sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Performance Analysis
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" gutterBottom>
                      Confidence Level
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={interviewScore.confidence * 100}
                      sx={{ mb: 1, height: 8 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      {Math.round(interviewScore.confidence * 100)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" gutterBottom>
                      Engagement Score
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={interviewScore.engagement * 100}
                      color="secondary"
                      sx={{ mb: 1, height: 8 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      {Math.round(interviewScore.engagement * 100)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" gutterBottom>
                      Eye Contact
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={interviewScore.eyeContact * 100}
                      color="success"
                      sx={{ mb: 1, height: 8 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      {Math.round(interviewScore.eyeContact * 100)}%
                    </Typography>
                  </Grid>
                </Grid>
              </Card>
            </Grid>
            
            {/* Recommendations */}
            {recommendations.length > 0 && (
              <Grid item xs={12}>
                <Card sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Recommendations for Improvement
                  </Typography>
                  <List dense>
                    {recommendations.map((rec, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={`• ${rec}`} />
                      </ListItem>
                    ))}
                  </List>
                </Card>
              </Grid>
            )}
            
            {/* Top Performing Answers */}
            <Grid item xs={12}>
              <Card sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Answer Analysis
                </Typography>
                <List>
                  {conversationLog
                    .filter(entry => entry.type === 'answer' && entry.score)
                    .sort((a, b) => (b.score || 0) - (a.score || 0))
                    .slice(0, 3)
                    .map((entry, index) => (
                      <div key={entry.id}>
                        <ListItem>
                          <ListItemText
                            primary={`Answer ${index + 1} - Score: ${entry.score}/100`}
                            secondary={
                              <>
                                <Typography variant="body2" sx={{ mb: 1 }}>
                                  {entry.evaluation}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  "{entry.content.substring(0, 100)}..."
                                </Typography>
                              </>
                            }
                          />
                        </ListItem>
                        {index < 2 && <Divider />}
                      </div>
                    ))
                  }
                </List>
              </Card>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowConversationLog(true)} startIcon={<History />}>
            View Full Conversation
          </Button>
          <Button onClick={() => navigate('/')} variant="contained">
            Start New Interview
          </Button>
        </DialogActions>
      </Dialog>
    );
  };

  if (!interviewStarted) {
    return (
      <Container maxWidth="md">
        <div style={{ paddingTop: '32px', paddingBottom: '32px', textAlign: 'center' }}>
          <Typography variant="h3" gutterBottom color="primary">
            AI Interview Preparation
          </Typography>
          <Typography variant="h6" color="text.secondary" paragraph>
            Ready to practice your interview skills with our AI interviewer?
          </Typography>
          
          <Card sx={{ mt: 4, p: 4 }}>
            <Typography variant="h5" gutterBottom>
              Interview Details
            </Typography>
            <Box sx={{ mb: 3 }}>
              <Chip label={`Role: ${jobRole}`} sx={{ mr: 1 }} size="medium" />
              {hasResume && <Chip label="Resume-based" color="secondary" size="medium" />}
              <Chip label="AI-Powered" color="primary" size="medium" />
            </Box>
            
            <Typography variant="body1" paragraph>
              This interview will be conducted by our AI interviewer using a ReadyPlayer.me avatar. 
              Your responses will be evaluated in real-time, and you'll receive a detailed feedback report at the end.
            </Typography>
            
            <Alert severity="info" sx={{ mb: 3 }}>
              <Typography variant="body2">
                • Speak clearly and answer completely for best evaluation results<br/>
                • Skipping questions will result in score penalties<br/>
                • All conversations are logged for your review
              </Typography>
            </Alert>
            
            <Button
              variant="contained"
              size="large"
              onClick={startInterview}
              sx={{ px: 4, py: 2 }}
            >
              Start Interview
            </Button>
          </Card>
        </div>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl">
      <div style={{ paddingTop: '16px', paddingBottom: '16px' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" gutterBottom>
              AI Interview Session
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Chip label={`Role: ${jobRole}`} size="small" />
              {hasResume && <Chip label="Resume-based" color="secondary" size="small" />}
              <Chip 
                label={`Score: ${Math.round(interviewScore.totalScore)}`} 
                color={interviewScore.totalScore >= 70 ? 'success' : 'warning'} 
                size="small" 
              />
              <Chip label={`Questions: ${conversationLog.filter(e => e.type === 'question').length}`} size="small" />
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            <IconButton 
              onClick={() => setShowConversationLog(true)}
              color="primary"
              disabled={conversationLog.length === 0}
            >
              <History />
            </IconButton>
            <Button
              variant="outlined"
              color="error"
              onClick={endInterview}
              startIcon={<Stop />}
            >
              End Interview
            </Button>
          </Box>
        </Box>

        {resumeData && (
          <Card sx={{ mb: 3, backgroundColor: '#f8f9fa' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom color="primary">
                📄 Resume Analysis
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
                {resumeData.skills.slice(0, 8).map((skill: string, index: number) => (
                  <Chip key={index} label={skill} size="small" color="primary" variant="outlined" />
                ))}
              </Box>
              <Typography variant="body2" color="text.secondary">
                Questions will be personalized based on your resume
              </Typography>
            </CardContent>
          </Card>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} md={7}>
            {/* AI Interviewer Avatar */}
            <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="white">
                  AI Interviewer - ReadyPlayer.me Avatar
                </Typography>
                <Avatar3D
                  isListening={isRecording}
                  isSpeaking={avatarSpeaking}
                  currentText={avatarText}
                  userPosition={{ x: 0, y: 0, z: 5 }}
                  onAvatarReady={() => console.log('Avatar ready!')}
                />
              </CardContent>
            </Card>
            
            {/* User Video Feed */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Your Video Feed
                </Typography>
                <Box sx={{ position: 'relative', width: '100%', maxWidth: 640 }}>
                  <Webcam
                    width="100%"
                    height="auto"
                    style={{ borderRadius: 8 }}
                    screenshotFormat="image/jpeg"
                  />
                  {isRecording && (
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 10,
                        right: 10,
                        backgroundColor: 'red',
                        color: 'white',
                        px: 2,
                        py: 1,
                        borderRadius: 1,
                      }}
                    >
                      🔴 Recording
                    </Box>
                  )}
                </Box>
              </CardContent>
            </Card>

            {/* Question and Recording Section */}
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Question {currentQuestionNumber} of {MAX_QUESTIONS}
                </Typography>
                <Typography variant="body1" sx={{ mb: 3, fontSize: '1.1rem', fontWeight: 500 }}>
                  {currentQuestion || 'Ready to start your interview? Press "Start Interview" to begin.'}
                </Typography>
                
                {/* Real-time answer display */}
                {currentAnswer && (
                  <Paper sx={{ p: 2, mb: 2, bgcolor: '#f5f5f5' }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Your Response:
                    </Typography>
                    <Typography variant="body2">
                      {currentAnswer}
                    </Typography>
                  </Paper>
                )}
                
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <Button
                    variant={isRecording ? 'contained' : 'outlined'}
                    color={isRecording ? 'error' : 'primary'}
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={!currentQuestion}
                    startIcon={isRecording ? <MicOff /> : <Mic />}
                    size="large"
                  >
                    {isRecording ? 'Stop & Submit Answer' : 'Start Answering'}
                  </Button>
                  
                  <Button
                    variant="outlined"
                    color="warning"
                    onClick={skipQuestion}
                    disabled={isRecording || !currentQuestion}
                    startIcon={<SkipNext />}
                    size="large"
                  >
                    Skip Question (-15 pts)
                  </Button>
                </Box>
                
                {isListening && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    Listening... Speak clearly into your microphone.
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={5}>
            {/* Live Score Panel */}
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Interview Score: {Math.round(interviewScore.totalScore)}/100
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={interviewScore.totalScore} 
                  sx={{ height: 10, borderRadius: 5, mb: 2 }}
                  color={interviewScore.totalScore >= 70 ? 'success' : 'warning'}
                />
                
                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Questions Answered
                    </Typography>
                    <Typography variant="h6" color="primary">
                      {interviewScore.questionsAnswered}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="caption" color="text.secondary">
                      Questions Skipped
                    </Typography>
                    <Typography variant="h6" color={interviewScore.questionsSkipped > 0 ? 'error' : 'success'}>
                      {interviewScore.questionsSkipped}
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
            
            {/* Real-time Analysis Panel */}
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Live Performance Analysis
                </Typography>
                
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Confidence Level
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={analysisData.confidence * 100}
                    sx={{ mb: 1, height: 6 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {Math.round(analysisData.confidence * 100)}%
                  </Typography>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Engagement Score
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={analysisData.engagement * 100}
                    color="secondary"
                    sx={{ mb: 1, height: 6 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {Math.round(analysisData.engagement * 100)}%
                  </Typography>
                </Box>

                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Eye Contact
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={analysisData.eyeContact * 100}
                    color="success"
                    sx={{ mb: 1, height: 6 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {Math.round(analysisData.eyeContact * 100)}%
                  </Typography>
                </Box>
              </CardContent>
            </Card>
            
            {/* Recent Conversation */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Conversation
                </Typography>
                <Box sx={{ maxHeight: 200, overflow: 'auto' }}>
                  {conversationLog.slice(-3).map((entry) => (
                    <Paper key={entry.id} sx={{ p: 1, mb: 1, bgcolor: entry.type === 'question' ? '#e3f2fd' : '#f3e5f5' }}>
                      <Typography variant="caption" color="text.secondary">
                        {entry.type === 'question' ? 'Q:' : entry.type === 'answer' ? 'A:' : 'SKIPPED:'}
                      </Typography>
                      <Typography variant="body2">
                        {entry.content.substring(0, 100)}{entry.content.length > 100 ? '...' : ''}
                      </Typography>
                      {entry.score && (
                        <Typography variant="caption" color="primary">
                          Score: {entry.score}/100
                        </Typography>
                      )}
                    </Paper>
                  ))}
                </Box>
                <Button 
                  size="small" 
                  onClick={() => setShowConversationLog(true)}
                  disabled={conversationLog.length === 0}
                  sx={{ mt: 1 }}
                >
                  View Full History
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
        
        {/* Conversation Log Dialog */}
        <Dialog open={showConversationLog} onClose={() => setShowConversationLog(false)} maxWidth="md" fullWidth>
          <DialogTitle>
            <Box display="flex" alignItems="center" gap={2}>
              <History color="primary" />
              Interview Conversation Log
            </Box>
          </DialogTitle>
          <DialogContent>
            <List>
              {conversationLog.map((entry, index) => (
                <div key={entry.id}>
                  <ListItem>
                    <ListItemText
                      primary={
                        <Box display="flex" alignItems="center" gap={1}>
                          <Chip 
                            label={entry.type.toUpperCase()} 
                            size="small" 
                            color={
                              entry.type === 'question' ? 'primary' : 
                              entry.type === 'answer' ? 'success' : 'warning'
                            }
                          />
                          {entry.score && (
                            <Chip label={`${entry.score}/100`} size="small" variant="outlined" />
                          )}
                        </Box>
                      }
                      secondary={
                        <>
                          <Typography variant="body2" sx={{ mt: 1 }}>
                            {entry.content}
                          </Typography>
                          {entry.evaluation && (
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                              {entry.evaluation}
                            </Typography>
                          )}
                          <Typography variant="caption" color="text.secondary">
                            {entry.timestamp.toLocaleTimeString()}
                          </Typography>
                        </>
                      }
                    />
                  </ListItem>
                  {index < conversationLog.length - 1 && <Divider />}
                </div>
              ))}
            </List>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setShowConversationLog(false)}>Close</Button>
          </DialogActions>
        </Dialog>
        
        {/* Feedback Report */}
        <FeedbackReport />
      </div>
    </Container>
  );
};

export default InterviewSession;