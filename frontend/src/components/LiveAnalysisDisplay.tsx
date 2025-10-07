import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Alert,
  Grid,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Psychology,
  RecordVoiceOver,
  Face,
} from '@mui/icons-material';
import {
  LiveAnalysisDisplayProps,
} from '../types';

const LiveAnalysisDisplay: React.FC<LiveAnalysisDisplayProps> = ({
  facialData,
  voiceData,
  sessionId,
  isConnected,
}) => {
  const [alertCount, setAlertCount] = useState(0);
  const [encouragementCount, setEncouragementCount] = useState(0);
  const [lastAnalysisTime, setLastAnalysisTime] = useState<Date | null>(null);

  useEffect(() => {
    // Update last analysis time
    if (facialData || voiceData) {
      setLastAnalysisTime(new Date());
    }
    
    // Update feedback counts
    if (facialData?.real_time_feedback?.alerts) {
      setAlertCount(prev => prev + facialData.real_time_feedback!.alerts.length);
    }
    if (facialData?.real_time_feedback?.encouragements) {
      setEncouragementCount(prev => prev + facialData.real_time_feedback!.encouragements.length);
    }
    if (voiceData?.feedback?.alerts) {
      setAlertCount(prev => prev + voiceData.feedback!.alerts.length);
    }
    if (voiceData?.feedback?.encouragements) {
      setEncouragementCount(prev => prev + voiceData.feedback!.encouragements.length);
    }
  }, [facialData, voiceData]);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return <TrendingUp color="success" />;
      case 'declining':
        return <TrendingDown color="error" />;
      default:
        return <TrendingFlat color="info" />;
    }
  };

  const getProgressColor = (value: number) => {
    if (value >= 0.8) return 'success';
    if (value >= 0.6) return 'info';
    if (value >= 0.4) return 'warning';
    return 'error';
  };

  const getDominantEmotion = (emotions: { [key: string]: number }) => {
    return Object.entries(emotions).reduce((a, b) => 
      emotions[a[0]] > emotions[b[0]] ? a : b
    )[0];
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Connection Status */}
      <Card elevation={2}>
        <CardContent sx={{ pb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: isConnected ? 'success.main' : 'error.main',
                  animation: isConnected ? 'pulse 2s infinite' : 'none',
                }}
              />
              <Typography variant="body2" fontWeight={600}>
                {isConnected ? 'Live Analysis Active' : 'Disconnected'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Chip
                label={`Session: ${sessionId.slice(-8)}`}
                size="small"
                color="primary"
                variant="outlined"
              />
              {lastAnalysisTime && (
                <Chip
                  label={`Last: ${lastAnalysisTime.toLocaleTimeString()}`}
                  size="small"
                  color="success"
                  variant="outlined"
                />
              )}
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Grid container spacing={2}>
        {/* Facial Analysis */}
        {facialData && (
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Face color="primary" />
                  Facial Analysis
                  {facialData.frame_number && (
                    <Chip label={`Frame ${facialData.frame_number}`} size="small" />
                  )}
                </Typography>

                {/* Core Metrics */}
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Confidence</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {facialData.trends?.confidence && getTrendIcon(facialData.trends.confidence)}
                        <Typography variant="body2">{Math.round(facialData.confidence * 100)}%</Typography>
                      </Box>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={facialData.confidence * 100}
                      color={getProgressColor(facialData.confidence)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Engagement</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {facialData.trends?.engagement && getTrendIcon(facialData.trends.engagement)}
                        <Typography variant="body2">{Math.round(facialData.engagement * 100)}%</Typography>
                      </Box>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={facialData.engagement * 100}
                      color={getProgressColor(facialData.engagement)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Eye Contact</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {facialData.trends?.eye_contact && getTrendIcon(facialData.trends.eye_contact)}
                        <Typography variant="body2">{Math.round(facialData.eye_contact * 100)}%</Typography>
                      </Box>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={facialData.eye_contact * 100}
                      color={getProgressColor(facialData.eye_contact)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>
                </Box>

                {/* Emotions */}
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>Current Emotion</Typography>
                  <Chip
                    label={getDominantEmotion(facialData.emotions)}
                    color="secondary"
                    icon={<Psychology />}
                  />
                </Box>

                {/* Real-time Feedback */}
                {facialData.real_time_feedback && (
                  <Box>
                    {facialData.real_time_feedback.alerts?.length > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="caption" color="warning.main" fontWeight={600}>
                          Areas to Improve:
                        </Typography>
                        {facialData.real_time_feedback.alerts.slice(0, 2).map((alert, idx) => (
                          <Alert key={idx} severity="warning" sx={{ mt: 0.5, fontSize: '0.75rem', py: 0.5 }}>
                            {alert}
                          </Alert>
                        ))}
                      </Box>
                    )}
                    
                    {facialData.real_time_feedback.encouragements?.length > 0 && (
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="caption" color="success.main" fontWeight={600}>
                          You're doing well:
                        </Typography>
                        {facialData.real_time_feedback.encouragements.slice(0, 2).map((encouragement, idx) => (
                          <Alert key={idx} severity="success" sx={{ mt: 0.5, fontSize: '0.75rem', py: 0.5 }}>
                            {encouragement}
                          </Alert>
                        ))}
                      </Box>
                    )}
                    
                    {facialData.real_time_feedback.suggestions?.length > 0 && (
                      <Box>
                        <Typography variant="caption" color="info.main" fontWeight={600}>
                          Quick Tips:
                        </Typography>
                        {facialData.real_time_feedback.suggestions.slice(0, 1).map((suggestion, idx) => (
                          <Alert key={idx} severity="info" sx={{ mt: 0.5, fontSize: '0.75rem', py: 0.5 }}>
                            {suggestion}
                          </Alert>
                        ))}
                      </Box>
                    )}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* Voice Analysis */}
        {voiceData && (
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <RecordVoiceOver color="primary" />
                  Voice Analysis
                </Typography>

                {/* Voice Metrics */}
                <Box sx={{ mb: 3 }}>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Speech Clarity</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {voiceData.trends?.clarity && getTrendIcon(voiceData.trends.clarity)}
                        <Typography variant="body2">{Math.round(voiceData.clarity * 100)}%</Typography>
                      </Box>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={voiceData.clarity * 100}
                      color={getProgressColor(voiceData.clarity)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Voice Confidence</Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {voiceData.trends?.confidence && getTrendIcon(voiceData.trends.confidence)}
                        <Typography variant="body2">{Math.round(voiceData.confidence_score * 100)}%</Typography>
                      </Box>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={voiceData.confidence_score * 100}
                      color={getProgressColor(voiceData.confidence_score)}
                      sx={{ height: 8, borderRadius: 1 }}
                    />
                  </Box>
                </Box>

                {/* Voice Characteristics */}
                <Grid container spacing={1} sx={{ mb: 2 }}>
                  <Grid item xs={6}>
                    <Chip
                      label={`Pace: ${voiceData.pace}`}
                      size="small"
                      color={voiceData.pace === 'moderate' ? 'success' : 'default'}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Chip
                      label={`Volume: ${voiceData.volume}`}
                      size="small"
                      color={voiceData.volume === 'appropriate' ? 'success' : 'default'}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <Chip
                      label={`Tone: ${voiceData.tone}`}
                      size="small"
                      color={voiceData.tone === 'confident' ? 'success' : 'default'}
                    />
                  </Grid>
                  {voiceData.speaking_stats?.words_per_minute && (
                    <Grid item xs={6}>
                      <Chip
                        label={`${voiceData.speaking_stats.words_per_minute} WPM`}
                        size="small"
                        color="info"
                      />
                    </Grid>
                  )}
                </Grid>

                {/* Voice Feedback */}
                {voiceData.feedback && (
                  <Box>
                    {voiceData.feedback.alerts.map((alert, idx) => (
                      <Alert key={idx} severity="warning" sx={{ mb: 1, fontSize: '0.8rem' }}>
                        {alert}
                      </Alert>
                    ))}
                    {voiceData.feedback.encouragements.map((enc, idx) => (
                      <Alert key={idx} severity="success" sx={{ mb: 1, fontSize: '0.8rem' }}>
                        {enc}
                      </Alert>
                    ))}
                    {voiceData.feedback.suggestions.slice(0, 2).map((sug, idx) => (
                      <Alert key={idx} severity="info" sx={{ mb: 1, fontSize: '0.8rem' }}>
                        {sug}
                      </Alert>
                    ))}
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Performance Summary */}
      <Card elevation={2}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Session Performance</Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="success.main">{encouragementCount}</Typography>
                <Typography variant="body2">Positive Moments</Typography>
              </Box>
            </Grid>
            <Grid item xs={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="warning.main">{alertCount}</Typography>
                <Typography variant="body2">Areas to Improve</Typography>
              </Box>
            </Grid>
            <Grid item xs={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="info.main">
                  {facialData?.frame_number || 0}
                </Typography>
                <Typography variant="body2">Frames Analyzed</Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default LiveAnalysisDisplay;