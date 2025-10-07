import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  Alert,
  IconButton,
  Paper,
  Stack,
} from '@mui/material';
import {
  Mic,
  MicOff,
  SkipNext,
  Stop,
  Videocam,
  VideocamOff,
  Psychology,
} from '@mui/icons-material';
import Webcam from 'react-webcam';
import LiveAnalysisDisplay from './LiveAnalysisDisplay';
import webSocketService from '../services/WebSocketService';
import {
  InterviewScore,
} from '../types';

const EnhancedInterviewSession: React.FC = () => {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  
  const jobRole = searchParams.get('role') || 'Software Developer';
  const hasResume = searchParams.get('hasResume') === 'true';
  
  // State
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [questionStartTime, setQuestionStartTime] = useState<Date | null>(null);
  const [currentQuestionNumber, setCurrentQuestionNumber] = useState(0);

  // Live analysis data
  const [facialAnalysisData, setFacialAnalysisData] = useState<any>(null);
  const [voiceAnalysisData, setVoiceAnalysisData] = useState<any>(null);
  const [wsConnected, setWsConnected] = useState(false);

  // Performance analysis aggregation
  const [analysisHistory, setAnalysisHistory] = useState<{
    facial: Array<{timestamp: number, data: any}>,
    voice: Array<{timestamp: number, data: any}>,
    responses: Array<{question: string, answer: string, score: number, duration: number, timestamp: number}>
  }>({ facial: [], voice: [], responses: [] });

  // Interview scoring - Start with realistic base
  const [interviewScore, setInterviewScore] = useState<InterviewScore>({
    totalScore: 0, // Start at 0, earn points through performance
    questionsAnswered: 0,
    questionsSkipped: 0,
    averageResponseTime: 0,
    confidence: 0.50, // Start at neutral
    engagement: 0.50,
    eyeContact: 0.50,
    communication: 0.50, // Start at neutral
  });

  // Refs
  const webcamRef = useRef<Webcam>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const analysisIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isRecordingRef = useRef(false); // Track recording state for auto-restart

  const MAX_QUESTIONS = 5;

  // Realistic scoring algorithm based on multiple factors
  const calculateRealisticScore = (answer: string, responseTime: number, question: string) => {
    const trimmedAnswer = answer.trim();
    const words = trimmedAnswer.split(/\s+/).filter(word => word.length > 1); // Filter out single chars
    const wordCount = words.length;
    const sentenceCount = trimmedAnswer.split(/[.!?]+/).filter(s => s.trim().length > 2).length;
    const avgWordsPerSentence = sentenceCount > 0 ? wordCount / sentenceCount : 0;
    
    // IMMEDIATE FAIL CONDITIONS - These should result in very low scores
    if (wordCount === 0 || trimmedAnswer.length < 3) {
      return {
        score: 0,
        feedback: 'No meaningful response provided. Please answer the question with specific details.',
        strengths: [],
        improvements: ['Provide a complete answer to the question', 'Use specific examples from your experience', 'Speak clearly and confidently']
      };
    }
    
    if (wordCount < 5) {
      return {
        score: Math.floor(Math.random() * 10) + 5, // 5-15 points for minimal effort
        feedback: 'Response is too brief and lacks substance. Elaborate with specific examples.',
        strengths: ['Attempted to answer'],
        improvements: ['Provide much more detail', 'Include specific examples', 'Explain your thought process', 'Use the STAR method (Situation, Task, Action, Result)']
      };
    }
    
    if (wordCount < 15) {
      return {
        score: Math.floor(Math.random() * 20) + 10, // 10-30 points for short answers
        feedback: 'Response needs significant development. Add more detail and examples.',
        strengths: wordCount > 8 ? ['Provided a basic answer'] : [],
        improvements: ['Expand your answer with more detail', 'Include specific examples', 'Explain the impact of your actions']
      };
    }
    
    // Content quality factors - much stricter requirements
    const hasSpecificExamples = /\b(example|instance|situation|time when|experience|project|when I|I worked|I handled|I managed)\b/i.test(trimmedAnswer);
    const hasQuantifiableResults = /\b(\d+%|\d+\s*percent|\$\d+|increased|decreased|improved|reduced|achieved|saved|gained|grew|doubled|tripled|\d+\s*(users|customers|sales|revenue|efficiency))\b/i.test(trimmedAnswer);
    const hasSTARStructure = /(situation|task|action|result|challenge|problem|solution|outcome)/gi.test(trimmedAnswer);
    const technicalTerms = (trimmedAnswer.match(/\b(developed|designed|implemented|managed|led|created|optimized|solved|programmed|architected|deployed|maintained|debugged|tested|analyzed)\b/gi) || []).length;
    
    // Communication quality - much stricter
    let clarityScore = 0;
    if (wordCount >= 20) clarityScore += 10;
    if (wordCount >= 40) clarityScore += 10;
    if (wordCount >= 60) clarityScore += 10; // Max 30 points for word count
    
    // Structure scoring - much more demanding
    let structureScore = 0;
    if (sentenceCount >= 2) structureScore += 5;
    if (avgWordsPerSentence >= 8 && avgWordsPerSentence <= 25) structureScore += 10;
    if (trimmedAnswer.includes(',') || trimmedAnswer.includes(';')) structureScore += 5; // Complex sentences
    
    // Timing factors - stricter
    let timingScore = 0;
    if (responseTime < 5) timingScore = 0; // Too fast = no thought
    else if (responseTime < 15) timingScore = 5; // Still too quick
    else if (responseTime >= 20 && responseTime <= 120) timingScore = 15; // Good thinking time
    else if (responseTime <= 180) timingScore = 10; // Acceptable
    else timingScore = 5; // Too long
    
    // Behavioral analysis from facial data - no free points
    let nonverbalScore = 0;
    if (analysisHistory.facial.length > 0) {
      const recentFacial = analysisHistory.facial.slice(-10);
      const avgConfidence = recentFacial.reduce((sum, entry) => sum + (entry.data.confidence || 0), 0) / recentFacial.length;
      const avgEngagement = recentFacial.reduce((sum, entry) => sum + (entry.data.engagement || 0), 0) / recentFacial.length;
      const avgEyeContact = recentFacial.reduce((sum, entry) => sum + (entry.data.eye_contact || 0), 0) / recentFacial.length;
      
      // Only award points if metrics are actually good
      if (avgConfidence > 0.6) nonverbalScore += 5;
      if (avgEngagement > 0.6) nonverbalScore += 5;
      if (avgEyeContact > 0.6) nonverbalScore += 5;
      nonverbalScore = Math.round(nonverbalScore);
    }
    
    // Start with ZERO base score - must earn every point
    let totalScore = 0;
    totalScore += clarityScore; // Communication clarity (max 30)
    totalScore += structureScore; // Structure (max 20)
    totalScore += timingScore; // Timing (max 15)
    totalScore += nonverbalScore; // Non-verbal (max 15)
    
    // Quality bonuses - only for genuinely good content
    if (hasSpecificExamples && wordCount >= 30) totalScore += 10;
    if (hasQuantifiableResults && wordCount >= 25) totalScore += 10;
    if (hasSTARStructure && wordCount >= 40) totalScore += 10;
    if (technicalTerms >= 2 && wordCount >= 35) totalScore += 5;
    
    // Additional penalties for poor quality
    if (trimmedAnswer.toLowerCase().includes('i dont know') || trimmedAnswer.toLowerCase().includes('not sure')) {
      totalScore = Math.floor(totalScore * 0.7); // 30% penalty for uncertainty
    }
    
    if (trimmedAnswer.split(' ').filter(word => word === word.toLowerCase()).length > wordCount * 0.8) {
      totalScore -= 5; // Penalty for no capitalization (shows lack of care)
    }
    
    // Realistic score caps
    if (wordCount < 20) totalScore = Math.min(totalScore, 35);
    if (wordCount < 30 && !hasSpecificExamples) totalScore = Math.min(totalScore, 45);
    if (!hasSpecificExamples && !hasQuantifiableResults && !hasSTARStructure) {
      totalScore = Math.min(totalScore, 50); // Cap at 50 for generic answers
    }
    
    // Final score range: 0-100 (realistic failures possible)
    totalScore = Math.max(0, Math.min(100, totalScore));
    
    // Generate realistic feedback based on performance
    const strengths = [];
    const improvements = [];
    
    // Only give strengths credit for genuinely good performance
    if (hasSpecificExamples && wordCount >= 25) strengths.push('Used specific examples to support your answer');
    if (hasQuantifiableResults && wordCount >= 20) strengths.push('Included measurable results and impact');
    if (hasSTARStructure && wordCount >= 30) strengths.push('Followed a structured approach (STAR method)');
    if (wordCount >= 50 && clarityScore >= 20) strengths.push('Provided comprehensive and detailed response');
    if (nonverbalScore >= 10) strengths.push('Demonstrated confidence through body language');
    if (technicalTerms >= 2) strengths.push('Used relevant technical terminology effectively');
    if (timingScore >= 10) strengths.push('Took appropriate time to think through the response');
    
    // Be more specific about what needs improvement
    if (wordCount < 20) improvements.push('Expand your answers with significantly more detail and depth');
    if (wordCount >= 20 && wordCount < 40) improvements.push('Provide more comprehensive explanations with additional context');
    if (!hasSpecificExamples) improvements.push('Include concrete examples from your actual experience');
    if (!hasQuantifiableResults) improvements.push('Add measurable outcomes and impact of your work');
    if (!hasSTARStructure) improvements.push('Structure responses using STAR method (Situation, Task, Action, Result)');
    if (responseTime < 10) improvements.push('Take more time to think and plan your response');
    if (responseTime > 150) improvements.push('Work on being more concise while maintaining detail');
    if (nonverbalScore < 5) improvements.push('Maintain better eye contact and show more engagement');
    if (technicalTerms === 0 && wordCount > 15) improvements.push('Include more specific technical terminology relevant to the role');
    
    // Ensure there's always constructive feedback
    if (strengths.length === 0) {
      if (wordCount > 0) strengths.push('Made an attempt to answer the question');
    }
    if (improvements.length === 0 && totalScore < 85) {
      improvements.push('Continue practicing to refine your interview responses');
    }
    
    // Much more realistic feedback messages
    let feedback;
    if (totalScore >= 85) {
      feedback = 'Excellent response! Strong content with specific examples and clear impact.';
    } else if (totalScore >= 70) {
      feedback = 'Good response with solid content. Could be enhanced with more specific examples.';
    } else if (totalScore >= 50) {
      feedback = 'Adequate start, but needs more depth, examples, and structure to be compelling.';
    } else if (totalScore >= 25) {
      feedback = 'Basic attempt, but significantly lacks detail and specific examples. Needs major improvement.';
    } else {
      feedback = 'Insufficient response. Focus on providing detailed answers with concrete examples from your experience.';
    }
    
    return {
      score: totalScore,
      feedback,
      strengths,
      improvements
    };
  };

  // Capture and analyze function
  const captureAndAnalyze = useCallback(() => {
    if (webcamRef.current && wsConnected) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        webSocketService.sendFacialData(imageSrc);
      }
    }
  }, [wsConnected]);

  // Initialize WebSocket connection
  useEffect(() => {
    if (sessionId && interviewStarted) {
      const initWebSocket = async () => {
        const connected = await webSocketService.connect(sessionId);
        if (connected) {
          setWsConnected(true);
          
          // Set up event handlers
          webSocketService.setOnFacialAnalysis((data) => {
            setFacialAnalysisData(data);
            
            // Store analysis data with timestamp
            setAnalysisHistory(prev => ({
              ...prev,
              facial: [...prev.facial, { timestamp: Date.now(), data }].slice(-50) // Keep last 50 entries
            }));
            
            // Calculate running averages from recent data
            setAnalysisHistory(current => {
              const recentFacial = current.facial.slice(-20); // Last 20 readings
              if (recentFacial.length > 0) {
                const avgConfidence = recentFacial.reduce((sum, entry) => sum + (entry.data.confidence || 0), 0) / recentFacial.length;
                const avgEngagement = recentFacial.reduce((sum, entry) => sum + (entry.data.engagement || 0), 0) / recentFacial.length;
                const avgEyeContact = recentFacial.reduce((sum, entry) => sum + (entry.data.eye_contact || 0), 0) / recentFacial.length;
                
                setInterviewScore(prev => ({
                  ...prev,
                  confidence: Number(avgConfidence.toFixed(3)),
                  engagement: Number(avgEngagement.toFixed(3)),
                  eyeContact: Number(avgEyeContact.toFixed(3)),
                }));
              }
              return current;
            });
          });

          webSocketService.setOnVoiceAnalysis((data) => {
            setVoiceAnalysisData(data);
          });

          webSocketService.setOnConnectionChange((connected) => {
            setWsConnected(connected);
          });

          webSocketService.setOnError((error) => {
            console.error('WebSocket error:', error);
          });

          // Start heartbeat
          heartbeatIntervalRef.current = setInterval(() => {
            webSocketService.sendHeartbeat();
          }, 30000);
        }
      };

      initWebSocket();
    }

    return () => {
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
      webSocketService.disconnect();
    };
  }, [sessionId, interviewStarted]);

  // Start live analysis when video is enabled
  useEffect(() => {
    if (isVideoEnabled && wsConnected && webcamRef.current) {
      analysisIntervalRef.current = setInterval(() => {
        captureAndAnalyze();
      }, 3000); // Analyze every 3 seconds
    }

    return () => {
      if (analysisIntervalRef.current) {
        clearInterval(analysisIntervalRef.current);
      }
    };
  }, [isVideoEnabled, wsConnected, captureAndAnalyze]);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognitionConstructor = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognitionConstructor) {
        recognitionRef.current = new SpeechRecognitionConstructor();
        recognitionRef.current.continuous = true;
        recognitionRef.current.interimResults = true;
        recognitionRef.current.lang = 'en-US';
        recognitionRef.current.maxAlternatives = 1;
        
        recognitionRef.current.onstart = () => {
          console.log('🎤 Speech recognition started');
          setIsListening(true);
        };
        
        recognitionRef.current.onresult = (event) => {
          let finalTranscript = '';
          
          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            }
          }
          
          if (finalTranscript) {
            console.log('Final transcript:', finalTranscript);
            setCurrentAnswer(prev => {
              const newAnswer = prev + (prev ? ' ' : '') + finalTranscript.trim();
              console.log('Updated answer:', newAnswer);
              return newAnswer;
            });
          }
        };
        
        recognitionRef.current.onerror = (event) => {
          console.error('Speech recognition error:', event.error, event.message);
          setIsListening(false);
          
          // Handle specific errors and try to restart if appropriate
          if (event.error === 'no-speech') {
            console.log('No speech detected, continuing to listen...');
            // Don't stop recording for no-speech errors
            setIsListening(true);
          } else if (event.error === 'network') {
            console.log('Network error in speech recognition');
          } else if (event.error === 'aborted') {
            console.log('Speech recognition aborted');
          }
        };
        
        recognitionRef.current.onend = () => {
          console.log('🎤 Speech recognition ended');
          
          // Auto-restart if we're still recording (use ref to avoid stale closure)
          if (isRecordingRef.current && recognitionRef.current) {
            try {
              console.log('Restarting speech recognition...');
              recognitionRef.current.start();
            } catch (error) {
              console.error('Failed to restart speech recognition:', error);
              setIsListening(false);
            }
          } else {
            setIsListening(false);
          }
        };
        
        recognitionRef.current.onnomatch = () => {
          console.log('No speech match found');
        };
        
        recognitionRef.current.onspeechstart = () => {
          console.log('Speech detected');
        };
        
        recognitionRef.current.onspeechend = () => {
          console.log('Speech ended');
        };
      }
    } else {
      console.warn('Speech recognition not supported in this browser');
    }
  }, []);

  const startInterview = () => {
    setInterviewStarted(true);
    setQuestionStartTime(new Date());
    generateNewQuestion();
  };

  const endInterview = () => {
    setIsRecording(false);
    setIsListening(false);
    
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    
    if (analysisIntervalRef.current) {
      clearInterval(analysisIntervalRef.current);
    }
    
    webSocketService.disconnect();
    
    // Calculate comprehensive performance metrics from collected data
    const calculatePerformanceMetrics = () => {
      const facialMetrics = analysisHistory.facial.length > 0 ? {
        samples: analysisHistory.facial.length,
        avgConfidence: analysisHistory.facial.reduce((sum, entry) => sum + (entry.data.confidence || 0), 0) / analysisHistory.facial.length,
        avgEngagement: analysisHistory.facial.reduce((sum, entry) => sum + (entry.data.engagement || 0), 0) / analysisHistory.facial.length,
        avgEyeContact: analysisHistory.facial.reduce((sum, entry) => sum + (entry.data.eye_contact || 0), 0) / analysisHistory.facial.length,
        confidenceVariance: calculateVariance(analysisHistory.facial.map(entry => entry.data.confidence || 0)),
        engagementTrend: calculateTrend(analysisHistory.facial.map(entry => entry.data.engagement || 0))
      } : null;

      const responseMetrics = analysisHistory.responses.length > 0 ? {
        avgScore: analysisHistory.responses.reduce((sum, resp) => sum + resp.score, 0) / analysisHistory.responses.length,
        avgDuration: analysisHistory.responses.reduce((sum, resp) => sum + resp.duration, 0) / analysisHistory.responses.length,
        scoreImprovement: analysisHistory.responses.length > 1 ? 
          analysisHistory.responses[analysisHistory.responses.length - 1].score - analysisHistory.responses[0].score : 0,
        consistencyScore: calculateConsistency(analysisHistory.responses.map(resp => resp.score))
      } : null;

      return { facialMetrics, responseMetrics };
    };

    const calculateVariance = (values: number[]) => {
      const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
      return values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    };

    const calculateTrend = (values: number[]) => {
      if (values.length < 2) return 0;
      const firstHalf = values.slice(0, Math.floor(values.length / 2));
      const secondHalf = values.slice(Math.floor(values.length / 2));
      const firstAvg = firstHalf.reduce((sum, val) => sum + val, 0) / firstHalf.length;
      const secondAvg = secondHalf.reduce((sum, val) => sum + val, 0) / secondHalf.length;
      return secondAvg - firstAvg;
    };

    const calculateConsistency = (scores: number[]) => {
      if (scores.length < 2) return 100;
      const variance = calculateVariance(scores);
      return Math.max(0, 100 - variance * 2); // Convert variance to consistency score
    };

    const performanceMetrics = calculatePerformanceMetrics();

    // Store comprehensive interview data for report generation
    const interviewData = {
      sessionId: sessionId!,
      jobRole,
      totalQuestions: currentQuestionNumber,
      questionsAnswered: interviewScore.questionsAnswered,
      questionsSkipped: interviewScore.questionsSkipped,
      averageResponseTime: interviewScore.averageResponseTime,
      totalScore: interviewScore.totalScore,
      confidence: interviewScore.confidence,
      engagement: interviewScore.engagement,
      eyeContact: interviewScore.eyeContact,
      hasResume,
      responses: analysisHistory.responses,
      performanceMetrics,
      analysisData: {
        facialAnalysisSamples: analysisHistory.facial.length,
        voiceAnalysisSamples: analysisHistory.voice.length,
        interviewDuration: questionStartTime ? (Date.now() - questionStartTime.getTime()) / 1000 : 0,
        questionsAttempted: currentQuestionNumber
      }
    };
    
    // Store the interview data in sessionStorage for the report component
    sessionStorage.setItem(`interview_data_${sessionId}`, JSON.stringify(interviewData));
    
    // Navigate to interview report
    navigate(`/report/${sessionId}`);
  };

  const generateNewQuestion = async () => {
    if (currentQuestionNumber >= MAX_QUESTIONS) {
      endInterview();
      return;
    }
    
    setCurrentQuestionNumber(prev => prev + 1);
    
    try {
      // Always use direct AI service API call for question generation
      // WebSocket is only used for facial/voice analysis, not for question generation
      const aiServiceUrl = 'http://localhost:8001';
      
      // Prepare request data with debug logging
      const requestData = {
        session_id: sessionId,
        job_role: jobRole,
        category: getQuestionCategory(currentQuestionNumber),
        difficulty: getDifficulty(currentQuestionNumber),
        resume_data: hasResume ? getEnhancedResumeData() : null
      };
      
      console.log('🚀 Sending request to AI service with:', {
        sessionId: requestData.session_id,
        jobRole: requestData.job_role,
        questionNumber: currentQuestionNumber,
        category: requestData.category,
        difficulty: requestData.difficulty,
        hasResumeData: !!requestData.resume_data,
        resumeSkillsCount: requestData.resume_data?.skills?.length || 0,
        resumeExperienceCount: requestData.resume_data?.experience?.length || 0
      });
      
      const response = await fetch(`${aiServiceUrl}/generate/question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
      });
      
      if (response.ok) {
        const questionData = await response.json();
        setCurrentQuestion(questionData.question);
        setQuestionStartTime(new Date());
        console.log('✅ Question generated via API:', questionData.question.substring(0, 100) + '...');
      } else {
        const errorText = await response.text();
        throw new Error(`AI service responded with ${response.status}: ${errorText}`);
      }
      
    } catch (error) {
      console.error('Error generating question:', error);
      
      // Show error instead of fallback questions
      const errorMessage = `❌ AI Question Generation Failed: ${error.message}. Please refresh the page and try again. The system requires AI-generated personalized questions.`;
      setCurrentQuestion(errorMessage);
      setQuestionStartTime(new Date());
      console.error('AI question generation failed:', error);
    }
  };
  
  const getQuestionCategory = (questionNumber: number) => {
    // Role-based question categories to ensure appropriate questions for each job type
    const roleBasedCategories = {
      // Technical Roles
      'Software Developer': ['behavioral', 'technical', 'problem-solving', 'technical', 'behavioral'],
      'Data Scientist': ['behavioral', 'analytical', 'technical', 'project-based', 'behavioral'], 
      'DevOps Engineer': ['behavioral', 'technical', 'problem-solving', 'technical', 'behavioral'],
      'Frontend Developer': ['behavioral', 'technical', 'design-thinking', 'technical', 'behavioral'],
      'Backend Developer': ['behavioral', 'technical', 'system-design', 'technical', 'behavioral'],
      'Full Stack Developer': ['behavioral', 'technical', 'problem-solving', 'technical', 'behavioral'],
      
      // Business & Management Roles  
      'Product Manager': ['behavioral', 'strategic', 'stakeholder-management', 'leadership', 'behavioral'],
      'Project Manager': ['behavioral', 'organizational', 'leadership', 'conflict-resolution', 'behavioral'],
      'Business Analyst': ['behavioral', 'analytical', 'stakeholder-management', 'process-improvement', 'behavioral'],
      'Marketing Manager': ['behavioral', 'strategic', 'creative-thinking', 'leadership', 'behavioral'],
      'Sales Manager': ['behavioral', 'relationship-building', 'negotiation', 'leadership', 'behavioral'],
      
      // Design & Creative Roles
      'UX Designer': ['behavioral', 'design-thinking', 'user-research', 'creative-problem-solving', 'behavioral'],
      'UI Designer': ['behavioral', 'design-thinking', 'visual-design', 'creative-problem-solving', 'behavioral'],
      'Graphic Designer': ['behavioral', 'creative-thinking', 'project-management', 'client-communication', 'behavioral'],
      
      // Operations & Support Roles
      'HR Manager': ['behavioral', 'people-management', 'conflict-resolution', 'policy-implementation', 'behavioral'],
      'Customer Success': ['behavioral', 'relationship-building', 'problem-solving', 'communication', 'behavioral'],
      'Operations Manager': ['behavioral', 'process-improvement', 'leadership', 'problem-solving', 'behavioral'],
      
      // Financial Roles
      'Financial Analyst': ['behavioral', 'analytical', 'attention-to-detail', 'strategic-thinking', 'behavioral'],
      'Accountant': ['behavioral', 'analytical', 'compliance', 'attention-to-detail', 'behavioral'],
      
      // Default for unknown roles
      'default': ['behavioral', 'situational', 'problem-solving', 'leadership', 'behavioral']
    };
    
    // Get categories for the specific role, fallback to default
    const categories = roleBasedCategories[jobRole] || roleBasedCategories['default'];
    return categories[(questionNumber - 1) % categories.length];
  };
  
  const getDifficulty = (questionNumber: number) => {
    if (questionNumber <= 2) return 'easy';
    if (questionNumber <= 4) return 'medium';
    return 'hard';
  };
  
  const getEnhancedResumeData = () => {
    // Try to get resume data from session-specific storage key first
    const sessionSpecificKey = sessionId ? `resume_${sessionId}` : null;
    let storedResumeData = null;
    
    if (sessionSpecificKey) {
      storedResumeData = sessionStorage.getItem(sessionSpecificKey);
      console.log('Looking for resume data with session key:', sessionSpecificKey);
    }
    
    // Fallback to generic keys if session-specific key doesn't exist
    if (!storedResumeData) {
      storedResumeData = localStorage.getItem('resumeData') || sessionStorage.getItem('resumeData');
      console.log('Fallback: Looking for resume data with generic keys');
    }
    
    if (storedResumeData) {
      try {
        const parsedData = JSON.parse(storedResumeData);
        console.log('Found resume data:', {
          skills: parsedData?.skills?.length || 0,
          experience: parsedData?.experience?.length || 0,
          hasText: !!parsedData?.text
        });
        return parsedData;
      } catch (error) {
        console.warn('Failed to parse stored resume data:', error);
      }
    }
    
    console.log('No resume data found');
    // Return null if no resume data is available
    return null;
  };


  // Audio recording state for voice analysis
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioChunks, setAudioChunks] = useState<Blob[]>([]);
  
  const startRecording = async () => {
    console.log('🎤 Starting recording...');
    setIsRecording(true);
    isRecordingRef.current = true; // Update ref
    setCurrentAnswer('');
    setAudioChunks([]);
    
    try {
      // Start audio recording for voice analysis
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100
        } 
      });
      
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          setAudioChunks(prev => [...prev, event.data]);
        }
      };
      
      recorder.start(1000); // Collect data every 1000ms
      setMediaRecorder(recorder);
      
      // Start speech recognition for transcript
      if (recognitionRef.current) {
        try {
          recognitionRef.current.start();
          console.log('Speech recognition started successfully');
        } catch (error) {
          console.error('Error starting speech recognition:', error);
          setIsListening(false);
        }
      }
      
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setIsRecording(false);
      isRecordingRef.current = false;
      alert('Unable to access microphone. Please check your browser permissions.');
    }
  };

  const stopRecording = async () => {
    console.log('🛑 Stopping recording...');
    setIsRecording(false);
    isRecordingRef.current = false; // Update ref
    setIsListening(false);
    
    // Stop speech recognition
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
        console.log('Speech recognition stopped successfully');
      } catch (error) {
        console.error('Error stopping speech recognition:', error);
      }
    }
    
    // Stop audio recording and send for analysis
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      
      // Stop all audio tracks
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
      
      // Wait for final audio chunks and process
      setTimeout(() => {
        processAudioAndTranscript();
      }, 500);
    } else if (currentAnswer.trim()) {
      // If only speech recognition was used
      processTranscriptOnly();
    }
  };
  
  const processAudioAndTranscript = async () => {
    if (audioChunks.length === 0 && !currentAnswer.trim()) {
      console.log('No audio or transcript data to process');
      return;
    }
    
    const responseTime = questionStartTime ? (new Date().getTime() - questionStartTime.getTime()) / 1000 : 0;
    
    try {
      let voiceAnalysisData = null;
      
      // Send audio for voice analysis if available
      if (audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'response.webm');
        formData.append('session_id', sessionId!);
        formData.append('question_id', currentQuestionNumber.toString());
        
        console.log('🎵 Sending audio for voice analysis...');
        const voiceResponse = await fetch('http://localhost:8001/analyze/voice', {
          method: 'POST',
          body: formData
        });
        
        if (voiceResponse.ok) {
          voiceAnalysisData = await voiceResponse.json();
          console.log('✅ Voice analysis completed:', voiceAnalysisData);
        } else {
          console.error('Voice analysis failed:', voiceResponse.status);
        }
      }
      
      // Evaluate the response with both transcript and voice data
      const evaluation = await evaluateResponse(currentAnswer, responseTime, voiceAnalysisData);
      updateScoreWithEvaluation(evaluation, responseTime);
      
      // Proceed to next question or end interview
      proceedToNextStep();
      
    } catch (error) {
      console.error('Error processing audio and transcript:', error);
      // Fallback to transcript-only evaluation
      processTranscriptOnly();
    }
  };
  
  const processTranscriptOnly = async () => {
    const responseTime = questionStartTime ? (new Date().getTime() - questionStartTime.getTime()) / 1000 : 0;
    const evaluation = calculateRealisticScore(currentAnswer, responseTime, currentQuestion);
    updateScoreWithEvaluation(evaluation, responseTime);
    proceedToNextStep();
  };
  
  const proceedToNextStep = () => {
    setCurrentAnswer('');
    setAudioChunks([]);
    
    // Check if interview should end
    const totalQuestions = interviewScore.questionsAnswered + 1 + interviewScore.questionsSkipped;
    if (totalQuestions >= MAX_QUESTIONS) {
      setTimeout(endInterview, 2000);
    } else {
      setTimeout(generateNewQuestion, 2000);
    }
  };
  
  const evaluateResponse = async (transcript: string, responseTime: number, voiceData: any) => {
    try {
      // Enhanced evaluation with AI service including voice data
      const aiServiceUrl = 'http://localhost:8001';
      const response = await fetch(`${aiServiceUrl}/evaluate/comprehensive`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          answer_text: transcript,
          question: currentQuestion,
          session_id: sessionId,
          response_time: responseTime,
          voice_analysis: voiceData, // Include voice analysis data
          job_role: jobRole
        })
      });
      
      if (response.ok) {
        const evaluation = await response.json();
        console.log('✅ Comprehensive evaluation completed:', evaluation);
        return evaluation;
      } else {
        console.error('AI evaluation failed, using fallback');
        throw new Error('AI evaluation service failed');
      }
    } catch (error) {
      console.error('Error in AI evaluation:', error);
      // Fallback to local evaluation
      return calculateRealisticScore(transcript, responseTime, currentQuestion);
    }
  };
  
  const updateScoreWithEvaluation = (evaluation: any, responseTime: number) => {
    // Use the actual evaluation score - no generous defaults!
    const score = evaluation.score || 0; // If no score provided, default to 0
    
    // Store response data for comprehensive analysis
    setAnalysisHistory(prev => ({
      ...prev,
      responses: [...prev.responses, {
        question: currentQuestion,
        answer: currentAnswer.trim(),
        score: score,
        duration: responseTime,
        timestamp: Date.now()
      }]
    }));
    
    // Calculate realistic cumulative score
    setInterviewScore(prev => {
      const newQuestionsAnswered = prev.questionsAnswered + 1;
      
      // Each question is worth up to 20 points (100 total / 5 questions)
      const questionValue = score * 0.20; // Convert 100-point scale to 20-point scale
      const newTotalScore = prev.totalScore + questionValue;
      
      return {
        ...prev,
        totalScore: newTotalScore,
        questionsAnswered: newQuestionsAnswered,
        averageResponseTime: (prev.averageResponseTime * prev.questionsAnswered + responseTime) / newQuestionsAnswered,
      };
    });
    
    // Show evaluation feedback in console
    console.log('✅ Answer evaluated:', {
      score,
      feedback: evaluation.feedback,
      strengths: evaluation.strengths,
      improvements: evaluation.improvements,
      responseTime: `${responseTime.toFixed(1)}s`
    });
  };

  const skipQuestion = () => {
    console.log('❌ Question skipped:', currentQuestion);
    
    // Store skip as a failed response
    setAnalysisHistory(prev => ({
      ...prev,
      responses: [...prev.responses, {
        question: currentQuestion,
        answer: '[SKIPPED]',
        score: 0, // Zero points for skipped questions
        duration: 0,
        timestamp: Date.now()
      }]
    }));
    
    // Apply realistic skip penalties
    setInterviewScore(prev => {
      const newQuestionsSkipped = prev.questionsSkipped + 1;
      const newTotalQuestions = prev.questionsAnswered + newQuestionsSkipped;
      
      // Skipping severely damages interview performance
      // Each skip: 0 points + penalty based on skip frequency
      const skipPenalty = calculateSkipPenalty(newQuestionsSkipped, newTotalQuestions);
      let newTotalScore = Math.max(0, prev.totalScore - skipPenalty);
      
      // Auto-fail for excessive skipping
      const skipRate = newQuestionsSkipped / newTotalQuestions;
      if (skipRate >= 0.6) { // Failing 3+ out of 5 questions
        newTotalScore = Math.min(newTotalScore, 15); // Cap at very low score
      } else if (skipRate >= 0.4) { // Skipping 2+ out of 5 questions  
        newTotalScore = Math.min(newTotalScore, 35); // Cap at poor score
      }
      
      return {
        ...prev,
        totalScore: newTotalScore,
        questionsSkipped: newQuestionsSkipped,
        // Harsher behavioral penalties for skipping
        confidence: Math.max(0.05, prev.confidence - 0.25), // Severe confidence drop
        engagement: Math.max(0.05, prev.engagement - 0.20), // Major engagement penalty
        communication: Math.max(0.05, prev.communication - 0.15), // Communication suffers too
      };
    });
    
    const totalQuestions = interviewScore.questionsAnswered + interviewScore.questionsSkipped + 1;
    if (totalQuestions >= MAX_QUESTIONS) {
      setTimeout(endInterview, 1000);
    } else {
      generateNewQuestion();
    }
  };
  
  const calculateSkipPenalty = (totalSkips: number, totalQuestions: number) => {
    const skipRate = totalSkips / totalQuestions;
    
    // Much harsher progressive penalties for skipping
    if (skipRate >= 0.8) return 50; // Devastating - skipping 4+ out of 5 questions
    if (skipRate >= 0.6) return 40; // Severe - skipping 3+ out of 5 questions  
    if (skipRate >= 0.4) return 30; // High - skipping 2+ out of 5 questions
    if (skipRate >= 0.2) return 20; // Significant - skipping 1+ out of 5 questions
    return 12; // Noticeable base penalty for any skip
  };

  if (!interviewStarted) {
    return (
      <Container maxWidth="md" sx={{ py: 8 }}>
          <Card 
            sx={{ 
              textAlign: 'center', 
              p: 6,
              background: 'linear-gradient(135deg, rgba(240, 255, 244, 0.8) 0%, rgba(255, 255, 255, 1) 100%)',
              border: '1px solid rgba(4, 120, 87, 0.1)',
              boxShadow: '0 8px 32px rgba(4, 120, 87, 0.08)',
            }}
          >
            <Psychology sx={{ fontSize: 80, color: 'primary.main', mb: 3 }} />
            <Typography variant="h3" gutterBottom fontWeight={700} color="text.primary">
              Practice Interview Session
            </Typography>
            <Typography variant="h6" color="text.secondary" paragraph sx={{ lineHeight: 1.6 }}>
              Take your time in this supportive space to practice and build your confidence.
            </Typography>
            
            <Stack direction="row" spacing={2} justifyContent="center" sx={{ mb: 4, flexWrap: 'wrap', gap: 1 }}>
              <Chip 
                label={`Role: ${jobRole}`} 
                sx={{ 
                  backgroundColor: 'primary.main',
                  color: 'white',
                  fontWeight: 600,
                }} 
              />
              {hasResume && (
                <Chip 
                  label="Resume-Enhanced" 
                  sx={{
                    backgroundColor: 'secondary.main',
                    color: 'text.primary',
                    fontWeight: 600,
                  }}
                />
              )}
              <Chip 
                label="AI-Guided" 
                sx={{
                  backgroundColor: 'success.main',
                  color: 'white',
                  fontWeight: 600,
                }}
              />
            </Stack>
            
            <Alert severity="info" sx={{ mb: 4, textAlign: 'left', borderRadius: 3 }}>
              <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                Your Safe Practice Experience:
              </Typography>
              <Typography variant="body2" sx={{ lineHeight: 1.6 }}>
                • Supportive real-time feedback and guidance<br/>
                • 5 personalized questions tailored to your goals<br/>
                • Constructive insights to help you improve<br/>
                • Detailed growth report for your journey
              </Typography>
            </Alert>
            
            <Button
              variant="contained"
              size="large"
              onClick={startInterview}
              startIcon={<Videocam />}
              sx={{ 
                px: 8, 
                py: 3,
                fontSize: '1.2rem',
                fontWeight: 700,
                background: 'linear-gradient(135deg, #F59E0B 0%, #FFC700 100%)',
                color: '#2F4858',
                '&:hover': {
                  background: 'linear-gradient(135deg, #D97706 0%, #F59E0B 100%)',
                  transform: 'translateY(-2px)',
                },
              }}
            >
              Begin Practice Session
            </Button>
          </Card>
        </Container>
      );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 2 }}>
        <Grid container spacing={3}>
          {/* Main Interview Section */}
          <Grid item xs={12} lg={8}>
            <Stack spacing={3}>
              {/* Question Card */}
              <Card>
                <CardContent sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" fontWeight={600}>
                      Question {currentQuestionNumber} of {MAX_QUESTIONS}
                    </Typography>
                    <Chip
                      label={wsConnected ? 'Live Analysis Active' : 'Connecting...'}
                      color={wsConnected ? 'success' : 'warning'}
                      size="small"
                    />
                  </Box>
                  
                  <Typography
                    variant="body1"
                    sx={{
                      fontSize: '1.1rem',
                      lineHeight: 1.6,
                      mb: 3,
                      p: 2,
                      backgroundColor: 'grey.50',
                      borderRadius: 2,
                      borderLeft: '4px solid',
                      borderLeftColor: 'primary.main',
                    }}
                  >
                    {currentQuestion}
                  </Typography>
                  
                  {currentAnswer && (
                    <Paper sx={{ p: 2, mb: 2, bgcolor: 'success.50' }}>
                      <Typography variant="subtitle2" gutterBottom color="success.dark">
                        Your Response:
                      </Typography>
                      <Typography variant="body2">
                        {currentAnswer}
                      </Typography>
                    </Paper>
                  )}
                  
                  {/* Voice Recording Controls */}
                  <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', gap: 1, mb: 3 }}>
                    <Button
                      variant={isRecording ? 'contained' : 'outlined'}
                      color={isRecording ? 'error' : 'primary'}
                      onClick={isRecording ? stopRecording : startRecording}
                      disabled={!currentQuestion}
                      startIcon={isRecording ? <MicOff /> : <Mic />}
                      size="large"
                      sx={{ minWidth: '200px' }}
                    >
                      {isRecording ? 'Stop & Analyze Answer' : 'Start Voice Recording'}
                    </Button>
                  </Stack>
                  
                  {/* Action Buttons */}
                  <Stack direction="row" spacing={2} sx={{ flexWrap: 'wrap', gap: 1 }}>
                    <Button
                      variant="outlined"
                      color="warning"
                      onClick={skipQuestion}
                      disabled={isRecording || !currentQuestion}
                      startIcon={<SkipNext />}
                      size="large"
                      sx={{
                        borderColor: 'warning.main',
                        '&:hover': {
                          backgroundColor: 'warning.light',
                          borderColor: 'warning.dark',
                        }
                      }}
                    >
                      Skip Question (Major Penalty)
                    </Button>
                    
                    <Button
                      variant="outlined"
                      color="error"
                      onClick={endInterview}
                      startIcon={<Stop />}
                      size="large"
                    >
                      End Interview
                    </Button>
                  </Stack>
                  
                  {isListening && (
                    <Alert severity="info" sx={{ mt: 2 }}>
                      🎤 Listening... Speak clearly into your microphone.
                    </Alert>
                  )}
                </CardContent>
              </Card>

              {/* Video Feed */}
              <Card>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" fontWeight={600}>
                      Video Feed
                    </Typography>
                    <IconButton
                      onClick={() => setIsVideoEnabled(!isVideoEnabled)}
                      color={isVideoEnabled ? 'primary' : 'default'}
                    >
                      {isVideoEnabled ? <Videocam /> : <VideocamOff />}
                    </IconButton>
                  </Box>
                  
                  {isVideoEnabled ? (
                    <Box sx={{ position: 'relative', borderRadius: 2, overflow: 'hidden' }}>
                      <Webcam
                        ref={webcamRef}
                        width="100%"
                        height="auto"
                        screenshotFormat="image/jpeg"
                        style={{ maxHeight: '400px', objectFit: 'cover' }}
                      />
                      {isRecording && (
                        <Box
                          sx={{
                            position: 'absolute',
                            top: 16,
                            right: 16,
                            backgroundColor: 'error.main',
                            color: 'white',
                            px: 2,
                            py: 1,
                            borderRadius: 1,
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                          }}
                        >
                          <Box
                            sx={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              backgroundColor: 'white',
                              animation: 'pulse 2s infinite',
                            }}
                          />
                          RECORDING
                        </Box>
                      )}
                    </Box>
                  ) : (
                    <Box
                      sx={{
                        height: 300,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        backgroundColor: 'grey.100',
                        borderRadius: 2,
                      }}
                    >
                      <Typography variant="h6" color="text.secondary">
                        Video Disabled
                      </Typography>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Stack>
          </Grid>

          {/* Live Analysis Panel */}
          <Grid item xs={12} lg={4}>
            <LiveAnalysisDisplay
              facialData={facialAnalysisData}
              voiceData={voiceAnalysisData}
              sessionId={sessionId || ''}
              isConnected={wsConnected}
            />
            
            {/* Interview Progress */}
            <Card sx={{ mt: 3 }}>
              <CardContent>
                <Typography variant="h6" gutterBottom fontWeight={600}>
                  Interview Progress
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Overall Score</Typography>
                    <Typography variant="body2" fontWeight={600}>
                      {Math.round(interviewScore.totalScore)}/100
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={Math.max(0, interviewScore.totalScore)}
                    sx={{ height: 10, borderRadius: 5 }}
                    color={interviewScore.totalScore >= 70 ? 'success' : 'warning'}
                  />
                </Box>
                
                <Grid container spacing={2} sx={{ textAlign: 'center' }}>
                  <Grid item xs={6}>
                    <Typography variant="h5" fontWeight={700} color="primary.main">
                      {interviewScore.questionsAnswered}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Answered
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="h5" fontWeight={700} color="warning.main">
                      {interviewScore.questionsSkipped}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Skipped
                    </Typography>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>
    );
};

export default EnhancedInterviewSession;