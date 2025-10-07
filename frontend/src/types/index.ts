// Import speech recognition types from declaration file
import './speech-recognition';

// Interview Related Types
export interface ConversationEntry {
  id: string;
  type: 'question' | 'answer' | 'skipped';
  content: string;
  timestamp: Date;
  score?: number;
  evaluation?: string;
  isAnswered: boolean;
}

export interface InterviewScore {
  totalScore: number;
  questionsAnswered: number;
  questionsSkipped: number;
  averageResponseTime: number;
  confidence: number;
  engagement: number;
  eyeContact: number;
  communication: number;
}

// Analysis Data Types
export interface AnalysisData {
  confidence: number;
  engagement: number;
  eye_contact: number;
  emotions: { [key: string]: number };
  trends?: { [key: string]: string };
  real_time_feedback?: {
    suggestions: string[];
    alerts: string[];
    encouragements: string[];
  };
  frame_number?: number;
}

export interface VoiceData {
  clarity: number;
  pace: string;
  tone: string;
  volume: string;
  confidence_score: number;
  fundamental_freq?: number;
  speech_rate?: number;
  trends?: { [key: string]: string };
  feedback?: {
    suggestions: string[];
    alerts: string[];
    encouragements: string[];
  };
  speaking_stats?: {
    average_clarity: number;
    average_confidence: number;
    dominant_pace: string;
    words_per_minute: number;
  };
}

// Component Props Types
export interface LiveAnalysisDisplayProps {
  facialData?: AnalysisData;
  voiceData?: VoiceData;
  sessionId: string;
  isConnected: boolean;
}

export interface NavigationHeaderProps {
  currentSession?: string;
  sessionScore?: number;
  showSessionInfo?: boolean;
}