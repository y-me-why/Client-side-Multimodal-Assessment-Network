import librosa
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class VoiceAnalyzer:
    def __init__(self):
        """Initialize voice analyzer with default parameters."""
        self.sample_rate = 22050
        self.hop_length = 512
        self.n_mfcc = 13
        self.scaler = StandardScaler()
        
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = None) -> Dict:
        """
        Analyze audio data for voice characteristics and emotional indicators.
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary containing voice analysis results
        """
        try:
            if sample_rate is None:
                sample_rate = self.sample_rate
                
            # Resample if necessary
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Extract features
            features = self._extract_features(audio_data)
            
            # Analyze voice characteristics
            analysis = {
                "voice_features": features,
                "confidence_score": self._calculate_voice_confidence(features),
                "emotional_state": self._analyze_emotional_state(features),
                "speech_quality": self._analyze_speech_quality(features),
                "engagement_level": self._calculate_engagement_level(features)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            return {"error": str(e)}
    
    def _extract_features(self, audio_data: np.ndarray) -> Dict:
        """Extract comprehensive audio features."""
        features = {}
        
        # Basic audio properties
        features['duration'] = len(audio_data) / self.sample_rate
        features['energy'] = np.sum(audio_data ** 2)
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio_data)[0])
        
        # Pitch and fundamental frequency
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=self.sample_rate)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = max(pitch_values) - min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Tempo and rhythm
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        features['tempo'] = tempo
        features['beat_frames'] = len(beats)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        return features
    
    def _calculate_voice_confidence(self, features: Dict) -> float:
        """Calculate confidence score based on voice characteristics."""
        confidence_indicators = []
        
        # Stable pitch indicates confidence
        if features['pitch_std'] > 0:
            pitch_stability = 1 / (1 + features['pitch_std'] / max(features['pitch_mean'], 1))
            confidence_indicators.append(pitch_stability * 0.3)
        
        # Appropriate energy level
        energy_score = min(1, features['energy'] / 1000)  # Normalize energy
        confidence_indicators.append(energy_score * 0.2)
        
        # Clear articulation (low zero crossing rate indicates steady speech)
        articulation_score = max(0, 1 - features['zero_crossing_rate'])
        confidence_indicators.append(articulation_score * 0.2)
        
        # Spectral characteristics
        spectral_stability = 1 / (1 + features['spectral_centroid_std'] / max(features['spectral_centroid_mean'], 1))
        confidence_indicators.append(spectral_stability * 0.3)
        
        return min(1, sum(confidence_indicators))
    
    def _analyze_emotional_state(self, features: Dict) -> Dict[str, float]:
        """Analyze emotional state from voice features."""
        emotions = {}
        
        # Happiness/Positivity (higher pitch, faster tempo)
        pitch_happiness = min(1, features['pitch_mean'] / 200) if features['pitch_mean'] > 0 else 0
        tempo_happiness = min(1, features['tempo'] / 120) if features['tempo'] > 0 else 0
        emotions['happiness'] = (pitch_happiness + tempo_happiness) / 2
        
        # Nervousness (high pitch variation, fast speech)
        nervousness = min(1, features['pitch_std'] / 50) if features['pitch_std'] > 0 else 0
        emotions['nervousness'] = nervousness
        
        # Calmness (stable pitch, moderate tempo)
        pitch_stability = 1 / (1 + features['pitch_std'] / max(features['pitch_mean'], 1)) if features['pitch_mean'] > 0 else 0
        tempo_stability = 1 - abs(features['tempo'] - 100) / 100 if features['tempo'] > 0 else 0
        emotions['calmness'] = (pitch_stability + max(0, tempo_stability)) / 2
        
        # Stress (high spectral centroid, irregular patterns)
        stress_indicator = min(1, features['spectral_centroid_mean'] / 3000)
        emotions['stress'] = stress_indicator
        
        return emotions
    
    def _analyze_speech_quality(self, features: Dict) -> Dict[str, float]:
        """Analyze speech quality metrics."""
        quality = {}
        
        # Clarity (based on spectral contrast)
        quality['clarity'] = min(1, features['spectral_contrast_mean'] / 20)
        
        # Fluency (consistent tempo and rhythm)
        if features['tempo'] > 0:
            fluency_score = 1 - abs(features['tempo'] - 100) / 100  # Optimal around 100 BPM
            quality['fluency'] = max(0, fluency_score)
        else:
            quality['fluency'] = 0.5
        
        # Articulation (low zero crossing rate with good energy)
        articulation = (1 - features['zero_crossing_rate']) * min(1, features['energy'] / 500)
        quality['articulation'] = articulation
        
        # Overall speech quality
        quality['overall'] = (quality['clarity'] + quality['fluency'] + quality['articulation']) / 3
        
        return quality
    
    def _calculate_engagement_level(self, features: Dict) -> float:
        """Calculate engagement level based on voice dynamics."""
        engagement_factors = []
        
        # Pitch variation (engaged speakers vary their pitch)
        if features['pitch_mean'] > 0:
            pitch_variation = min(1, features['pitch_range'] / 100)
            engagement_factors.append(pitch_variation * 0.4)
        
        # Energy level (engaged speakers have appropriate energy)
        energy_engagement = min(1, features['energy'] / 800)
        engagement_factors.append(energy_engagement * 0.3)
        
        # Spectral richness (engaged speech has rich spectral content)
        spectral_richness = min(1, features['spectral_centroid_mean'] / 2000)
        engagement_factors.append(spectral_richness * 0.3)
        
        return min(1, sum(engagement_factors))