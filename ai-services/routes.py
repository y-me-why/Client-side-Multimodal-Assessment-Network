from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
from PIL import Image
import numpy as np

# Import required libraries
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
import cv2
import numpy as np
import mediapipe as mp
import librosa
import google.generativeai as genai
import logging
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

class RealFacialAnalyzer:
    def __init__(self):
        """Initialize facial expression analyzer with MediaPipe."""
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def analyze_frame(self, image_np):
        """Analyze facial expressions from image frame."""
        try:
            rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {
                    "confidence": 0.5,
                    "engagement": 0.5,
                    "eye_contact": 0.5,
                    "emotions": {"neutral": 1.0}
                }
            
            # Extract landmarks for analysis
            landmarks = results.multi_face_landmarks[0]
            
            # Calculate metrics
            confidence_score = self._calculate_confidence(landmarks, image_np.shape)
            engagement_score = self._calculate_engagement(landmarks, image_np.shape)
            eye_contact_score = self._calculate_eye_contact(landmarks, image_np.shape)
            emotions = self._detect_emotions(landmarks, image_np.shape)
            
            return {
                "confidence": confidence_score,
                "engagement": engagement_score,
                "eye_contact": eye_contact_score,
                "emotions": emotions
            }
            
        except Exception as e:
            logger.error(f"Facial analysis error: {str(e)}")
            return {
                "confidence": 0.5,
                "engagement": 0.5,
                "eye_contact": 0.5,
                "emotions": {"neutral": 1.0},
                "error": str(e)
            }
    
    def _calculate_confidence(self, landmarks, frame_shape):
        """Calculate confidence based on facial posture and expressions."""
        h, w = frame_shape[:2]
        
        # Extract key landmark points
        landmark_points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
        
        # Calculate head pose (confidence indicator)
        nose_tip = landmark_points[1]  # Nose tip
        chin = landmark_points[175]    # Chin
        forehead = landmark_points[10] # Forehead center
        
        # Head stability (less movement = more confidence)
        head_center_y = (nose_tip[1] + chin[1]) / 2
        head_tilt = abs(nose_tip[0] - chin[0]) / w  # Normalized head tilt
        
        # Eye openness ratio (confident people maintain good eye contact)
        left_eye_top = landmark_points[159]
        left_eye_bottom = landmark_points[145]
        right_eye_top = landmark_points[386]
        right_eye_bottom = landmark_points[374]
        
        left_eye_ratio = abs(left_eye_top[1] - left_eye_bottom[1]) / h
        right_eye_ratio = abs(right_eye_top[1] - right_eye_bottom[1]) / h
        eye_openness = (left_eye_ratio + right_eye_ratio) / 2
        
        # Mouth analysis (slight smile indicates confidence)
        mouth_left = landmark_points[61]
        mouth_right = landmark_points[291]
        mouth_top = landmark_points[13]
        mouth_bottom = landmark_points[14]
        
        mouth_width = abs(mouth_right[0] - mouth_left[0]) / w
        mouth_curve = (mouth_left[1] + mouth_right[1]) / 2 - mouth_top[1]
        
        # Combine factors for confidence score
        stability_score = max(0, 1 - head_tilt * 5)  # Penalize head tilting
        eye_score = min(1, eye_openness * 30)        # Reward good eye openness
        posture_score = 0.8 if head_center_y < h * 0.6 else 0.6  # Head position
        expression_score = min(1, max(0.3, mouth_curve * 0.1 + 0.7))  # Slight positive expression
        
        confidence = (stability_score * 0.3 + eye_score * 0.4 + posture_score * 0.2 + expression_score * 0.1)
        return min(1.0, max(0.3, confidence))
    
    def _calculate_engagement(self, landmarks, frame_shape):
        """Calculate engagement level from facial activity and attention cues."""
        h, w = frame_shape[:2]
        landmark_points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
        
        # Eye gaze direction (looking forward indicates engagement)
        left_eye_center = landmark_points[468]
        right_eye_center = landmark_points[473]
        nose_tip = landmark_points[1]
        
        # Calculate if eyes are looking forward (engaged)
        eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2
        gaze_alignment = 1 - abs(eye_center_x - nose_tip[0]) / (w * 0.1)
        gaze_score = max(0, min(1, gaze_alignment))
        
        # Eyebrow position (raised eyebrows can indicate interest/surprise)
        left_eyebrow = landmark_points[70]
        right_eyebrow = landmark_points[300]
        eyebrow_height = h - (left_eyebrow[1] + right_eyebrow[1]) / 2
        eyebrow_score = min(1, eyebrow_height / (h * 0.4))
        
        # Face orientation (facing camera indicates engagement)
        face_center_x = nose_tip[0]
        face_orientation = 1 - abs(face_center_x - w/2) / (w/2)
        orientation_score = max(0.5, face_orientation)
        
        # Eye openness (alert, engaged people have open eyes)
        left_eye_top = landmark_points[159]
        left_eye_bottom = landmark_points[145]
        eye_openness = abs(left_eye_top[1] - left_eye_bottom[1]) / h
        alertness_score = min(1, eye_openness * 25)
        
        # Combine engagement factors
        engagement = (gaze_score * 0.4 + orientation_score * 0.3 + alertness_score * 0.2 + eyebrow_score * 0.1)
        return min(1.0, max(0.4, engagement))
    
    def _calculate_eye_contact(self, landmarks, frame_shape):
        """Calculate eye contact score based on gaze direction and eye position."""
        h, w = frame_shape[:2]
        landmark_points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
        
        # Key eye landmarks
        left_eye_center = landmark_points[468]
        right_eye_center = landmark_points[473]
        left_eye_inner = landmark_points[463]
        left_eye_outer = landmark_points[414]
        right_eye_inner = landmark_points[243]
        right_eye_outer = landmark_points[189]
        
        # Calculate eye gaze direction
        left_iris_x = left_eye_center[0]
        left_eye_width = abs(left_eye_outer[0] - left_eye_inner[0])
        left_gaze_ratio = (left_iris_x - left_eye_inner[0]) / left_eye_width if left_eye_width > 0 else 0.5
        
        right_iris_x = right_eye_center[0]
        right_eye_width = abs(right_eye_outer[0] - right_eye_inner[0])
        right_gaze_ratio = (right_iris_x - right_eye_inner[0]) / right_eye_width if right_eye_width > 0 else 0.5
        
        # Good eye contact means iris is centered (ratio around 0.5)
        left_contact_score = 1 - abs(left_gaze_ratio - 0.5) * 2
        right_contact_score = 1 - abs(right_gaze_ratio - 0.5) * 2
        
        gaze_score = (left_contact_score + right_contact_score) / 2
        
        # Eye openness (must have eyes open for eye contact)
        left_eye_top = landmark_points[159]
        left_eye_bottom = landmark_points[145]
        right_eye_top = landmark_points[386]
        right_eye_bottom = landmark_points[374]
        
        left_openness = abs(left_eye_top[1] - left_eye_bottom[1]) / h
        right_openness = abs(right_eye_top[1] - right_eye_bottom[1]) / h
        openness_score = min(1, (left_openness + right_openness) * 20)
        
        # Head orientation (should be facing forward for good eye contact)
        nose_tip = landmark_points[1]
        face_center_x = nose_tip[0]
        head_forward_score = 1 - abs(face_center_x - w/2) / (w/2)
        
        # Combine factors
        eye_contact = (gaze_score * 0.5 + openness_score * 0.3 + head_forward_score * 0.2)
        return min(1.0, max(0.2, eye_contact))
    
    def _detect_emotions(self, landmarks, frame_shape):
        """Detect emotions from facial landmarks with advanced analysis."""
        h, w = frame_shape[:2]
        landmark_points = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
        
        # Mouth analysis for happiness/smile
        mouth_left = landmark_points[61]
        mouth_right = landmark_points[291]
        mouth_top = landmark_points[13]
        mouth_bottom = landmark_points[14]
        mouth_left_corner = landmark_points[308]
        mouth_right_corner = landmark_points[78]
        
        # Smile detection (mouth corners raised)
        mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
        corner_lift = mouth_center_y - (mouth_left_corner[1] + mouth_right_corner[1]) / 2
        smile_intensity = max(0, min(1, corner_lift / (h * 0.01)))
        
        # Mouth curvature
        mouth_curve = abs(mouth_left[0] - mouth_right[0]) / w
        mouth_openness = abs(mouth_top[1] - mouth_bottom[1]) / h
        
        # Eye analysis for different emotions
        left_eye_top = landmark_points[159]
        left_eye_bottom = landmark_points[145]
        right_eye_top = landmark_points[386]
        right_eye_bottom = landmark_points[374]
        
        eye_openness = (abs(left_eye_top[1] - left_eye_bottom[1]) + abs(right_eye_top[1] - right_eye_bottom[1])) / (2 * h)
        
        # Eyebrow analysis for surprise/concern
        left_eyebrow_inner = landmark_points[70]
        left_eyebrow_outer = landmark_points[63]
        right_eyebrow_inner = landmark_points[300]
        right_eyebrow_outer = landmark_points[293]
        
        eyebrow_height = h - (left_eyebrow_inner[1] + right_eyebrow_inner[1]) / 2
        eyebrow_raise = max(0, min(1, (eyebrow_height / (h * 0.3) - 1)))
        
        # Forehead tension (frowning)
        forehead_center = landmark_points[10]
        brow_furrow = abs(left_eyebrow_inner[0] - right_eyebrow_inner[0]) / w
        
        # Calculate emotion probabilities
        happy_score = smile_intensity * 0.7 + mouth_curve * 0.3
        neutral_score = 1 - abs(smile_intensity - 0.2) - abs(eyebrow_raise - 0.1)
        confident_score = (eye_openness * 0.4 + smile_intensity * 0.3 + (1 - eyebrow_raise) * 0.3)
        nervous_score = eyebrow_raise * 0.5 + (1 - eye_openness) * 0.3 + mouth_openness * 0.2
        surprised_score = eyebrow_raise * 0.6 + eye_openness * 0.4
        
        # Normalize scores
        total = happy_score + neutral_score + confident_score + nervous_score + surprised_score
        if total > 0:
            emotions = {
                "happy": max(0.05, happy_score / total),
                "neutral": max(0.1, neutral_score / total),
                "confident": max(0.05, confident_score / total),
                "nervous": max(0.02, nervous_score / total),
                "surprised": max(0.02, surprised_score / total)
            }
        else:
            emotions = {
                "neutral": 0.6,
                "happy": 0.2,
                "confident": 0.15,
                "nervous": 0.03,
                "surprised": 0.02
            }
        
        # Ensure probabilities sum to approximately 1
        emotion_sum = sum(emotions.values())
        emotions = {k: v/emotion_sum for k, v in emotions.items()}
        
        return emotions

class RealVoiceAnalyzer:
    def __init__(self):
        """Initialize voice analyzer."""
        self.sample_rate = 22050
        self.scaler = StandardScaler()
        
    def analyze_audio(self, audio_bytes):
        """Analyze voice characteristics from audio data."""
        try:
            # Convert bytes to numpy array (simplified)
            # In real implementation, you'd properly decode the audio format
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            
            if len(audio_data) == 0:
                return self._default_analysis()
            
            # Extract features
            features = self._extract_features(audio_data)
            
            return {
                "clarity": features.get("clarity", 0.8),
                "pace": self._classify_pace(features.get("tempo", 120)),
                "tone": self._analyze_tone(features),
                "volume": self._classify_volume(features.get("energy", 100)),
                "confidence_score": features.get("confidence", 0.75),
                "emotional_state": features.get("emotions", {"neutral": 0.8})
            }
            
        except Exception as e:
            logger.error(f"Voice analysis error: {str(e)}")
            return self._default_analysis()
    
    def _extract_features(self, audio_data):
        """Extract comprehensive audio features using librosa."""
        try:
            # Ensure we have enough data
            if len(audio_data) < 100:
                return self._get_default_features()
            
            # Normalize audio data
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Basic energy and volume analysis
            energy = np.sum(audio_data ** 2)
            rms_energy = np.sqrt(np.mean(audio_data ** 2))
            
            # Zero crossing rate (indicates speech clarity)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data))))
            zcr = zero_crossings / len(audio_data)
            
            # Pitch analysis
            try:
                # Use autocorrelation for pitch detection
                autocorr = np.correlate(audio_data, audio_data, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find fundamental frequency
                if len(autocorr) > 100:
                    pitch_idx = np.argmax(autocorr[20:]) + 20  # Avoid DC component
                    if pitch_idx > 0:
                        fundamental_freq = self.sample_rate / pitch_idx
                    else:
                        fundamental_freq = 150  # Default
                else:
                    fundamental_freq = 150
            except:
                fundamental_freq = 150
            
            # Spectral features
            try:
                # FFT analysis
                fft = np.fft.fft(audio_data)
                magnitude = np.abs(fft[:len(fft)//2])
                freqs = np.fft.fftfreq(len(fft), 1/self.sample_rate)[:len(fft)//2]
                
                # Spectral centroid (brightness)
                if np.sum(magnitude) > 0:
                    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                else:
                    spectral_centroid = 1000
                
                # Spectral rolloff
                cumulative_mag = np.cumsum(magnitude)
                total_mag = cumulative_mag[-1]
                if total_mag > 0:
                    rolloff_idx = np.where(cumulative_mag >= 0.85 * total_mag)[0]
                    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 4000
                else:
                    spectral_rolloff = 4000
                    
            except:
                spectral_centroid = 1000
                spectral_rolloff = 4000
            
            # Voice quality metrics
            
            # Jitter (pitch stability)
            pitch_periods = []
            if fundamental_freq > 50 and fundamental_freq < 500:
                period_samples = int(self.sample_rate / fundamental_freq)
                for i in range(0, len(audio_data) - period_samples, period_samples):
                    period = audio_data[i:i + period_samples]
                    if len(period) > 0:
                        pitch_periods.append(len(period))
            
            if len(pitch_periods) > 1:
                jitter = np.std(pitch_periods) / np.mean(pitch_periods) if np.mean(pitch_periods) > 0 else 0
            else:
                jitter = 0.02  # Default moderate jitter
            
            # Shimmer (amplitude stability)
            if len(audio_data) > 100:
                frame_size = 100
                amplitudes = []
                for i in range(0, len(audio_data) - frame_size, frame_size//2):
                    frame = audio_data[i:i + frame_size]
                    amplitudes.append(np.max(np.abs(frame)))
                
                if len(amplitudes) > 1:
                    shimmer = np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0
                else:
                    shimmer = 0.05
            else:
                shimmer = 0.05
            
            # Speech rate estimation
            speech_rate = self._estimate_speech_rate(audio_data, zcr)
            
            # Emotional indicators from voice
            emotions = self._analyze_voice_emotions(fundamental_freq, energy, jitter, shimmer, spectral_centroid)
            
            return {
                "energy": float(energy),
                "rms_energy": float(rms_energy),
                "fundamental_freq": float(fundamental_freq),
                "spectral_centroid": float(spectral_centroid),
                "spectral_rolloff": float(spectral_rolloff),
                "zcr": float(zcr),
                "jitter": float(jitter),
                "shimmer": float(shimmer),
                "speech_rate": float(speech_rate),
                "clarity": self._calculate_clarity(zcr, spectral_centroid, jitter),
                "confidence": self._calculate_voice_confidence(jitter, shimmer, energy, fundamental_freq),
                "emotions": emotions
            }
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return self._get_default_features()
    
    def _estimate_speech_rate(self, audio_data, zcr):
        """Estimate speaking rate from audio features."""
        # Simple speech rate estimation based on energy and zero crossings
        if len(audio_data) == 0:
            return 120
            
        # Detect speech segments based on energy
        frame_size = int(0.025 * self.sample_rate)  # 25ms frames
        energy_threshold = np.max(audio_data ** 2) * 0.1
        
        speech_frames = 0
        for i in range(0, len(audio_data) - frame_size, frame_size):
            frame = audio_data[i:i + frame_size]
            if np.sum(frame ** 2) > energy_threshold:
                speech_frames += 1
        
        # Estimate words per minute based on speech activity
        duration_seconds = len(audio_data) / self.sample_rate
        if duration_seconds > 0:
            speech_ratio = speech_frames / (len(audio_data) // frame_size)
            estimated_wpm = speech_ratio * 150  # Baseline WPM
        else:
            estimated_wpm = 120
            
        return min(200, max(60, estimated_wpm))
    
    def _calculate_clarity(self, zcr, spectral_centroid, jitter):
        """Calculate speech clarity score."""
        # Good clarity: moderate ZCR, appropriate spectral centroid, low jitter
        zcr_score = 1 - abs(zcr - 0.1) / 0.1  # Optimal ZCR around 0.1
        centroid_score = 1 - abs(spectral_centroid - 1500) / 2000  # Optimal around 1500 Hz
        jitter_score = max(0, 1 - jitter * 10)  # Lower jitter is better
        
        clarity = (zcr_score * 0.4 + centroid_score * 0.3 + jitter_score * 0.3)
        return max(0.2, min(1.0, clarity))
    
    def _calculate_voice_confidence(self, jitter, shimmer, energy, pitch):
        """Calculate confidence from voice characteristics."""
        # Confident voice: stable pitch, steady amplitude, appropriate energy
        pitch_stability = max(0, 1 - jitter * 15)
        amplitude_stability = max(0, 1 - shimmer * 10)
        energy_score = min(1, energy / 100)  # Normalize energy
        pitch_score = 1 - abs(pitch - 150) / 150 if 80 < pitch < 300 else 0.5
        
        confidence = (pitch_stability * 0.3 + amplitude_stability * 0.3 + energy_score * 0.2 + pitch_score * 0.2)
        return max(0.3, min(1.0, confidence))
    
    def _analyze_voice_emotions(self, pitch, energy, jitter, shimmer, spectral_centroid):
        """Analyze emotional state from voice features."""
        # Emotion detection based on voice characteristics
        
        # Happy: higher pitch, stable, good energy
        happy_score = 0
        if pitch > 180:
            happy_score += 0.3
        if jitter < 0.03:
            happy_score += 0.2
        if energy > 50:
            happy_score += 0.2
            
        # Confident: steady pitch, stable amplitude, moderate energy
        confident_score = 0
        if 120 < pitch < 200:
            confident_score += 0.3
        if jitter < 0.025:
            confident_score += 0.3
        if shimmer < 0.05:
            confident_score += 0.2
            
        # Nervous: unstable pitch, higher jitter, variable energy
        nervous_score = 0
        if jitter > 0.04:
            nervous_score += 0.4
        if shimmer > 0.08:
            nervous_score += 0.3
        if pitch > 200:
            nervous_score += 0.2
            
        # Neutral: baseline characteristics
        neutral_score = max(0.3, 1 - happy_score - confident_score - nervous_score)
        
        # Normalize
        total = happy_score + confident_score + nervous_score + neutral_score
        if total > 0:
            return {
                "happy": happy_score / total,
                "confident": confident_score / total,
                "nervous": nervous_score / total,
                "neutral": neutral_score / total
            }
        else:
            return {"neutral": 0.7, "confident": 0.2, "happy": 0.05, "nervous": 0.05}
    
    def _get_default_features(self):
        """Return default features when analysis fails."""
        return {
            "energy": 50,
            "rms_energy": 0.1,
            "fundamental_freq": 150,
            "spectral_centroid": 1500,
            "spectral_rolloff": 3000,
            "zcr": 0.1,
            "jitter": 0.02,
            "shimmer": 0.05,
            "speech_rate": 120,
            "clarity": 0.8,
            "confidence": 0.7,
            "emotions": {"neutral": 0.7, "confident": 0.3}
        }
    
    def _classify_pace(self, tempo):
        if tempo < 80: return "slow"
        elif tempo > 140: return "fast" 
        else: return "moderate"
    
    def _analyze_tone(self, features):
        confidence = features.get("confidence", 0.5)
        if confidence > 0.8: return "confident"
        elif confidence > 0.6: return "neutral"
        else: return "uncertain"
    
    def _classify_volume(self, energy):
        if energy < 50: return "low"
        elif energy > 200: return "high"
        else: return "appropriate"
    
    def _default_analysis(self):
        return {
            "clarity": 0.8,
            "pace": "moderate",
            "tone": "neutral",
            "volume": "appropriate",
            "confidence_score": 0.75,
            "emotional_state": {"neutral": 0.8}
        }

class RealGeminiClient:
    def __init__(self):
        """Initialize Gemini client with API key."""
        # You'll need to set your GOOGLE_API_KEY environment variable
        api_key = os.getenv('GOOGLE_API_KEY', 'your-api-key-here')
        if api_key and api_key != 'your-api-key-here':
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.api_available = True
        else:
            logger.warning("GOOGLE_API_KEY not set, using fallback question generation")
            self.api_available = False
    
    async def generate_question(self, job_role, resume_data=None, difficulty="medium", category="behavioral"):
        """Generate interview question using Gemini API."""
        if not self.api_available:
            return self._fallback_question_generation(job_role, difficulty)
        
        try:
            # Build context-aware prompt
            prompt = self._build_question_prompt(job_role, resume_data, difficulty, category)
            
            response = self.model.generate_content(prompt)
            
            # Parse response
            question_data = self._parse_question_response(response.text)
            
            return {
                "question": question_data.get("question", f"Tell me about your experience as a {job_role}."),
                "category": question_data.get("category", "behavioral"),
                "difficulty": difficulty,
                "duration": question_data.get("duration", "2-3 minutes"),
                "criteria": question_data.get("criteria", ["Clarity", "Relevance", "Depth"]),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return self._fallback_question_generation(job_role, difficulty)
    
    async def evaluate_answer(self, answer_text, question):
        """Evaluate answer using Gemini API."""
        if not self.api_available:
            return self._fallback_answer_evaluation(answer_text)
        
        try:
            prompt = f"""
            Please evaluate this interview answer and provide detailed feedback:
            
            Question: {question}
            Answer: {answer_text}
            
            Provide your response in JSON format with:
            - score (1-100)
            - feedback (detailed feedback)
            - strengths (list of strengths)
            - improvements (list of improvement areas)
            
            JSON Response:
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                # Remove code block markers if present
                text = response.text.strip()
                if text.startswith('```json'):
                    text = text[7:]  # Remove ```json
                if text.endswith('```'):
                    text = text[:-3]  # Remove closing ```
                text = text.strip()
                
                evaluation = json.loads(text)
                return evaluation
            except json.JSONDecodeError:
                return self._parse_text_evaluation(response.text)
                
        except Exception as e:
            logger.error(f"Answer evaluation error: {str(e)}")
            return self._fallback_answer_evaluation(answer_text)
    
    def _build_question_prompt(self, job_role, resume_data, difficulty, category="behavioral"):
        """Build a comprehensive prompt for question generation."""
        
        # Define role-appropriate interviewer personas
        interviewer_personas = {
            # Technical roles - can include technical aspects but balanced
            'technical': 'You are an experienced hiring manager who understands both technical and soft skills.',
            # Business roles - focus on strategy, leadership, communication
            'business': 'You are a senior executive interviewing for strategic and leadership capabilities.',
            # Creative roles - focus on creativity, problem-solving, collaboration
            'creative': 'You are a creative director assessing design thinking and collaboration skills.',
            # Operations roles - focus on process, efficiency, people management
            'operations': 'You are an operations leader evaluating process improvement and people skills.'
        }
        
        # Categorize job roles
        role_type = 'technical'
        if job_role.lower() in ['product manager', 'project manager', 'business analyst', 'marketing manager', 'sales manager', 'financial analyst', 'accountant']:
            role_type = 'business'
        elif job_role.lower() in ['ux designer', 'ui designer', 'graphic designer', 'content creator', 'marketing specialist']:
            role_type = 'creative'
        elif job_role.lower() in ['hr manager', 'customer success', 'operations manager', 'office manager', 'support specialist']:
            role_type = 'operations'
        
        interviewer_persona = interviewer_personas.get(role_type, interviewer_personas['business'])
        
        # Category-specific guidance
        category_guidance = {
            'behavioral': 'Focus on past experiences, decision-making, and interpersonal situations. Use STAR method prompts.',
            'technical': 'Ask about hands-on experience with relevant tools, problem-solving approaches, and practical knowledge.',
            'situational': 'Present realistic workplace scenarios they might encounter in this role.',
            'leadership': 'Explore their experience leading teams, projects, or initiatives.',
            'strategic': 'Focus on planning, vision, market understanding, and strategic thinking.',
            'analytical': 'Ask about data-driven decision making, analysis approaches, and problem-solving methods.',
            'creative-thinking': 'Explore innovation, creative problem-solving, and design thinking approaches.',
            'stakeholder-management': 'Focus on communication, relationship building, and managing competing priorities.',
            'problem-solving': 'Present real-world challenges relevant to the role.',
            'communication': 'Assess presentation skills, difficult conversations, and clarity of thought.'
        }
        
        guidance = category_guidance.get(category, category_guidance['behavioral'])
        
        prompt = f"""
        {interviewer_persona} You are conducting a {difficulty}-level interview for a {job_role} position.
        
        INTERVIEW STYLE: Ask questions like a real, experienced interviewer would. Be conversational, professional, and genuinely interested in understanding the candidate.
        
        QUESTION TYPE: {category} - {guidance}
        
        CRITICAL REQUIREMENTS:
        1. Sound natural and conversational like a real person asking
        2. Make it specific to the {job_role} role and its typical responsibilities
        3. For non-technical roles, focus on business skills, leadership, communication, and problem-solving
        4. For technical roles, balance technical skills with soft skills and real-world application
        5. Reference their background specifically when possible
        6. Avoid overly complex or academic language
        7. Ask ONE clear, focused question
        """
        
        if resume_data:
            skills = resume_data.get('skills', [])
            experience = resume_data.get('experience', [])
            projects = resume_data.get('projects', [])
            asked_questions = resume_data.get('asked_questions', [])
            
            prompt += f"""
            
            CANDIDATE PROFILE:
            Skills: {', '.join(skills[:8]) if skills else 'Not specified'}
            
            Experience:"""
            
            for i, exp in enumerate(experience[:2]):
                if isinstance(exp, dict):
                    prompt += f"""
            {i+1}. {exp.get('role', 'Role')} at {exp.get('company', 'Company')} ({exp.get('duration', 'Duration')})
               - {exp.get('description', 'No description')}"""
                else:
                    prompt += f"""
            {i+1}. {exp}"""
            
            if projects:
                prompt += """
            
            Notable Projects:"""
                for i, proj in enumerate(projects[:2]):
                    if isinstance(proj, dict):
                        prompt += f"""
            {i+1}. {proj.get('name', 'Project')}
               - Description: {proj.get('description', 'No description')}
               - Technologies: {', '.join(proj.get('technologies', [])) if proj.get('technologies') else 'Not specified'}"""
                    else:
                        prompt += f"""
            {i+1}. {proj}"""
            
            if asked_questions:
                prompt += f"""
            
            ALREADY ASKED QUESTIONS (DO NOT REPEAT):
            {chr(10).join(['- ' + q for q in asked_questions[-5:]])}
            
            GENERATE A COMPLETELY DIFFERENT QUESTION that explores new aspects of their experience."""
        
        prompt += f"""
        
        REQUIREMENTS:
        1. Make the question specific to their background (reference specific skills, projects, or experiences)
        2. For {difficulty} difficulty, adjust complexity appropriately
        3. Focus on practical scenarios they would have encountered
        4. Ensure the question is engaging and thought-provoking
        5. Avoid repetition of previous questions
        
        Provide response in JSON format:
        {{
            "question": "Your personalized interview question here",
            "category": "behavioral/technical/situational/leadership",
            "duration": "expected time to answer",
            "criteria": ["what makes a good response"]
        }}
        
        JSON Response:
        """
        
        return prompt
    
    def _parse_question_response(self, response_text):
        """Parse Gemini response for question data."""
        try:
            # Remove code block markers if present
            text = response_text.strip()
            if text.startswith('```json'):
                text = text[7:]  # Remove ```json
            if text.endswith('```'):
                text = text[:-3]  # Remove closing ```
            text = text.strip()
            
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback parsing
            lines = response_text.strip().split('\n')
            question = lines[0] if lines else "Tell me about your experience."
            return {
                "question": question,
                "category": "behavioral",
                "duration": "2-3 minutes",
                "criteria": ["Clarity", "Relevance", "Depth"]
            }
    
    def _parse_text_evaluation(self, response_text):
        """Parse text response when JSON parsing fails."""
        lines = response_text.strip().split('\n')
        return {
            "score": 75,
            "feedback": response_text.strip()[:200] + "...",
            "strengths": ["Clear communication"],
            "improvements": ["Add more specific examples"]
        }
    
    def _fallback_question_generation(self, job_role, difficulty):
        """Fallback question generation when API is not available."""
        questions_by_role = {
            "Software Developer": [
                "Tell me about a challenging bug you've had to debug.",
                "How do you approach code reviews with your team?",
                "Describe a time when you had to learn a new technology quickly."
            ],
            "Data Scientist": [
                "Walk me through your approach to a machine learning project.",
                "How do you handle missing data in your datasets?",
                "Describe a time when your model didn't perform as expected."
            ],
            "Product Manager": [
                "How do you prioritize features in a product roadmap?",
                "Tell me about a time you had to make a difficult product decision.",
                "How do you gather and incorporate user feedback?"
            ]
        }
        
        import random
        role_questions = questions_by_role.get(job_role, [
            f"Tell me about your experience in {job_role}.",
            f"What challenges have you faced in {job_role} roles?",
            "Describe a project you're particularly proud of."
        ])
        
        return {
            "question": random.choice(role_questions),
            "category": "behavioral",
            "difficulty": difficulty,
            "duration": "2-3 minutes",
            "criteria": ["Clarity", "Relevance", "Depth"]
        }
    
    def _fallback_answer_evaluation(self, answer_text):
        """Fallback evaluation when API is not available."""
        word_count = len(answer_text.split())
        score = min(100, max(60, word_count * 2 + 20))
        
        feedback = "Your answer demonstrates good communication skills. "
        if word_count < 20:
            feedback += "Consider providing more detailed examples and explanations."
        elif word_count > 100:
            feedback += "Your answer is comprehensive and detailed."
        else:
            feedback += "Good balance of detail and conciseness."
        
        return {
            "score": score,
            "feedback": feedback,
            "strengths": ["Clear communication", "Structured response"],
            "improvements": ["Add specific metrics", "Include more examples"]
        }
    
    async def generate_interview_report(self, report_data):
        """Generate comprehensive interview report using Gemini AI."""
        if not self.api_available:
            return self._fallback_report_generation(report_data)
        
        try:
            prompt = self._build_report_prompt(report_data)
            response = self.model.generate_content(prompt)
            
            # Parse response
            report_analysis = self._parse_report_response(response.text)
            
            # Calculate overall score and placement likelihood
            overall_score = self._calculate_overall_score(report_data, report_analysis)
            placement_likelihood = self._determine_placement_likelihood(overall_score, report_data)
            
            return {
                "overall_score": overall_score,
                "placement_likelihood": placement_likelihood,
                "performance_summary": report_analysis.get("summary", "Performance analysis completed."),
                "strengths": report_analysis.get("strengths", []),
                "development_areas": report_analysis.get("development_areas", []),
                "detailed_feedback": report_analysis.get("detailed_feedback", {}),
                "skill_breakdown": report_analysis.get("skill_breakdown", {}),
                "recommendations": report_analysis.get("recommendations", []),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Report generation error: {str(e)}")
            return self._fallback_report_generation(report_data)
    
    async def generate_comprehensive_evaluation(self, interview_data):
        """Generate comprehensive interview evaluation based on full conversation history."""
        if not self.api_available:
            return self._fallback_comprehensive_evaluation(interview_data)
        
        try:
            prompt = self._build_comprehensive_evaluation_prompt(interview_data)
            
            # Use longer timeout for comprehensive evaluation
            response = self.model.generate_content(prompt)
            
            # Parse the comprehensive evaluation response
            evaluation = self._parse_comprehensive_evaluation_response(response.text)
            
            return {
                "session_id": interview_data.get("session_id", ""),
                "overall_score": evaluation.get("overall_score", 70),
                "placement_likelihood": evaluation.get("placement_likelihood", "Medium"),
                "performance_summary": evaluation.get("performance_summary", "Interview evaluation completed."),
                "strengths": evaluation.get("strengths", []),
                "development_areas": evaluation.get("development_areas", []),
                "detailed_feedback": evaluation.get("detailed_feedback", {}),
                "skill_breakdown": evaluation.get("skill_breakdown", {}),
                "recommendations": evaluation.get("recommendations", []),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation error: {str(e)}")
            return self._fallback_comprehensive_evaluation(interview_data)
    
    def _build_report_prompt(self, report_data):
        """Build comprehensive prompt for interview report generation."""
        responses_summary = ""
        total_score = 0
        question_count = 0
        
        for response in report_data.get('responses', []):
            if 'analysis' in response:
                total_score += response['analysis'].get('score', 0)
                question_count += 1
                responses_summary += f"""
        Q: {response.get('question', 'N/A')[:100]}...
        A: {response.get('response', 'N/A')[:150]}...
        Score: {response['analysis'].get('score', 0)}/100
        Feedback: {response['analysis'].get('feedback', 'N/A')[:100]}...
        """
        
        avg_response_score = (total_score / question_count) if question_count > 0 else 70
        
        prompt = f"""
        You are an expert HR professional and interview coach conducting a comprehensive interview performance analysis.
        
        INTERVIEW PERFORMANCE DATA:
        Job Role: {report_data.get('job_role', 'N/A')}
        Total Questions: {report_data.get('total_questions', 0)}
        Questions Answered: {report_data.get('questions_answered', 0)}
        Questions Skipped: {report_data.get('questions_skipped', 0)}
        Average Response Time: {report_data.get('average_response_time', 0):.1f} seconds
        Average Answer Score: {avg_response_score:.1f}/100
        
        DETAILED RESPONSES:
        {responses_summary}
        
        BEHAVIORAL ANALYSIS:
        """
        
        if report_data.get('facial_analysis_summary'):
            facial = report_data['facial_analysis_summary']
            prompt += f"""
        - Average Confidence: {facial.get('average_confidence', 0.75):.0%}
        - Average Engagement: {facial.get('average_engagement', 0.80):.0%}
        - Average Eye Contact: {facial.get('average_eye_contact', 0.70):.0%}
        """
        
        if report_data.get('voice_analysis_summary'):
            voice = report_data['voice_analysis_summary']
            prompt += f"""
        - Speech Clarity: {voice.get('average_clarity', 0.85):.0%}
        - Speaking Pace: {voice.get('dominant_pace', 'moderate')}
        - Voice Confidence: {voice.get('average_confidence', 0.75):.0%}
        """
        
        prompt += f"""
        
        ANALYSIS REQUIREMENTS:
        Provide a comprehensive analysis in JSON format with:
        
        1. "summary" - 2-3 sentence overall performance summary
        2. "strengths" - List of 3-5 key strengths demonstrated
        3. "development_areas" - List of 3-5 specific areas needing improvement  
        4. "detailed_feedback" - Object with categories:
           - "communication": score and feedback
           - "technical_knowledge": score and feedback
           - "problem_solving": score and feedback
           - "confidence": score and feedback
        5. "skill_breakdown" - Object with skill scores (0-100):
           - "verbal_communication": score
           - "confidence_level": score
           - "technical_competency": score
           - "problem_solving": score
           - "professionalism": score
        6. "recommendations" - List of 4-6 actionable recommendations for improvement
        
        Focus on being constructive, specific, and actionable. Consider the job role when evaluating technical vs soft skills.
        
        JSON Response:
        """
        
        return prompt
    
    def _build_comprehensive_evaluation_prompt(self, interview_data):
        """Build comprehensive prompt for evaluating full conversation history."""
        job_role = interview_data.get('job_role', 'Software Developer')
        responses = interview_data.get('responses', [])
        
        # Build conversation history
        conversation_summary = ""
        total_questions = len(responses)
        
        for i, response_data in enumerate(responses, 1):
            question = response_data.get('question', 'N/A')
            answer = response_data.get('answer', 'N/A')
            duration = response_data.get('duration', 0)
            
            conversation_summary += f"""

QUESTION {i}: {question}

CANDIDATE RESPONSE ({duration:.1f} seconds): {answer}

---"""
        
        # Behavioral metrics
        behavioral_summary = ""
        if interview_data.get('confidence') is not None:
            behavioral_summary += f"""
BEHAVIORAL ANALYSIS:
- Average Confidence Level: {interview_data.get('confidence', 0.5):.0%}
- Average Engagement: {interview_data.get('engagement', 0.5):.0%} 
- Average Eye Contact: {interview_data.get('eyeContact', 0.5):.0%}
- Questions Answered: {interview_data.get('questionsAnswered', 0)}/{interview_data.get('totalQuestions', 5)}
- Questions Skipped: {interview_data.get('questionsSkipped', 0)}
- Average Response Time: {interview_data.get('averageResponseTime', 60):.1f} seconds"""
        
        prompt = f"""
You are an expert HR professional and senior interview evaluator with 15+ years of experience in talent acquisition and candidate assessment. You are conducting a comprehensive evaluation of a {job_role} interview based on the complete conversation history.

**INTERVIEW DATA:**
Position: {job_role}
Total Questions Asked: {total_questions}
Interview Format: AI-powered behavioral and technical interview

**COMPLETE CONVERSATION HISTORY:**{conversation_summary}

{behavioral_summary}

**EVALUATION REQUIREMENTS:**

You must provide a thorough, professional evaluation that includes:

1. **OVERALL PERFORMANCE SCORE** (0-100): Based on response quality, depth, relevance, and communication skills
2. **PLACEMENT LIKELIHOOD**: "High" (80-100), "Medium" (60-79), or "Low" (0-59) based on job readiness
3. **PERFORMANCE SUMMARY**: 2-3 sentences summarizing overall interview performance
4. **KEY STRENGTHS**: 4-6 specific strengths demonstrated during the interview
5. **DEVELOPMENT AREAS**: 4-6 specific areas needing improvement
6. **DETAILED SKILL ASSESSMENT**: Scores (0-100) and feedback for:
   - Communication & Articulation
   - Technical Knowledge & Competency  
   - Problem-Solving & Analytical Thinking
   - Confidence & Professional Presence
7. **SKILL BREAKDOWN**: Numerical scores (0-100) for:
   - verbal_communication
   - confidence_level
   - technical_competency
   - problem_solving
   - professionalism
8. **ACTIONABLE RECOMMENDATIONS**: 5-7 specific, actionable recommendations for improvement

**EVALUATION CRITERIA:**
- **Response Quality**: Depth, specificity, structure, relevance
- **Communication**: Clarity, articulation, professional language
- **Technical Competency**: Knowledge demonstration, problem-solving approach
- **Behavioral Indicators**: Confidence, engagement, professionalism
- **STAR Method Usage**: Situation, Task, Action, Result structure
- **Concrete Examples**: Use of specific, measurable examples
- **Interview Readiness**: Overall preparedness for {job_role} positions

**SCORING GUIDELINES:**
- 90-100: Exceptional candidate, ready for senior roles
- 80-89: Strong candidate, ready for target role
- 70-79: Good candidate, minor improvements needed
- 60-69: Adequate candidate, moderate improvements needed
- 50-59: Developing candidate, significant improvement needed
- Below 50: Not ready, extensive preparation required

Provide your evaluation in JSON format:

{{
    "overall_score": <score 0-100>,
    "placement_likelihood": "<High/Medium/Low>",
    "performance_summary": "<2-3 sentence summary>",
    "strengths": [
        "<specific strength 1>",
        "<specific strength 2>",
        "<specific strength 3>",
        "<specific strength 4>"
    ],
    "development_areas": [
        "<specific area 1>",
        "<specific area 2>", 
        "<specific area 3>",
        "<specific area 4>"
    ],
    "detailed_feedback": {{
        "communication": {{"score": <0-100>, "feedback": "<specific feedback>"}},
        "technical_knowledge": {{"score": <0-100>, "feedback": "<specific feedback>"}},
        "problem_solving": {{"score": <0-100>, "feedback": "<specific feedback>"}},
        "confidence": {{"score": <0-100>, "feedback": "<specific feedback>"}}
    }},
    "skill_breakdown": {{
        "verbal_communication": <0-100>,
        "confidence_level": <0-100>,
        "technical_competency": <0-100>,
        "problem_solving": <0-100>,
        "professionalism": <0-100>
    }},
    "recommendations": [
        "<actionable recommendation 1>",
        "<actionable recommendation 2>",
        "<actionable recommendation 3>",
        "<actionable recommendation 4>",
        "<actionable recommendation 5>"
    ]
}}

JSON Response:
"""
        
        return prompt
    
    def _parse_comprehensive_evaluation_response(self, response_text):
        """Parse comprehensive evaluation response from Gemini."""
        try:
            # Remove code block markers if present
            text = response_text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            evaluation = json.loads(text)
            
            # Validate and ensure all required fields are present
            required_fields = {
                "overall_score": 70,
                "placement_likelihood": "Medium",
                "performance_summary": "Interview evaluation completed successfully.",
                "strengths": ["Completed the interview", "Professional communication"],
                "development_areas": ["Improve response depth", "Practice more examples"],
                "detailed_feedback": {
                    "communication": {"score": 70, "feedback": "Good communication skills"},
                    "technical_knowledge": {"score": 65, "feedback": "Adequate technical knowledge"},
                    "problem_solving": {"score": 70, "feedback": "Good problem-solving approach"},
                    "confidence": {"score": 65, "feedback": "Room for confidence improvement"}
                },
                "skill_breakdown": {
                    "verbal_communication": 70,
                    "confidence_level": 65,
                    "technical_competency": 65,
                    "problem_solving": 70,
                    "professionalism": 75
                },
                "recommendations": [
                    "Practice the STAR method for behavioral questions",
                    "Prepare more specific examples from experience",
                    "Work on confident body language and voice tone",
                    "Research common interview questions for the role"
                ]
            }
            
            # Fill in any missing fields with defaults
            for field, default_value in required_fields.items():
                if field not in evaluation:
                    evaluation[field] = default_value
            
            return evaluation
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {str(e)}")
            # Fallback parsing for non-JSON responses
            return self._parse_text_comprehensive_evaluation(response_text)
    
    def _parse_text_comprehensive_evaluation(self, response_text):
        """Fallback parsing when JSON parsing fails."""
        # Extract key information from text response
        lines = response_text.strip().split('\n')
        
        # Default evaluation structure
        evaluation = {
            "overall_score": 70,
            "placement_likelihood": "Medium",
            "performance_summary": "Interview evaluation completed with mixed results.",
            "strengths": ["Professional communication", "Completed the interview", "Relevant experience shared"],
            "development_areas": ["Provide more specific examples", "Improve response structure", "Build confidence"],
            "detailed_feedback": {
                "communication": {"score": 70, "feedback": "Generally clear communication with room for improvement"},
                "technical_knowledge": {"score": 65, "feedback": "Adequate technical knowledge demonstrated"},
                "problem_solving": {"score": 70, "feedback": "Good problem-solving approach shown"},
                "confidence": {"score": 65, "feedback": "Could benefit from increased confidence"}
            },
            "skill_breakdown": {
                "verbal_communication": 70,
                "confidence_level": 65,
                "technical_competency": 65,
                "problem_solving": 70,
                "professionalism": 75
            },
            "recommendations": [
                "Practice storytelling using the STAR method",
                "Prepare specific examples from your experience",
                "Work on maintaining confident body language",
                "Research common interview questions for your role",
                "Practice technical concepts relevant to the position"
            ]
        }
        
        # Try to extract score if mentioned in text
        for line in lines:
            if 'score' in line.lower() and any(char.isdigit() for char in line):
                # Extract number from line
                import re
                numbers = re.findall(r'\b\d+\b', line)
                if numbers:
                    score = int(numbers[0])
                    if 0 <= score <= 100:
                        evaluation["overall_score"] = score
                        break
        
        return evaluation
    
    def _fallback_comprehensive_evaluation(self, interview_data):
        """Fallback evaluation when AI is not available."""
        responses = interview_data.get('responses', [])
        
        # Calculate basic metrics
        if responses:
            # Analyze response quality
            total_words = 0
            total_responses = len(responses)
            
            for response in responses:
                answer = response.get('answer', '')
                words = len(answer.split()) if answer else 0
                total_words += words
            
            avg_words_per_response = total_words / total_responses if total_responses > 0 else 0
            
            # Basic scoring based on response length and completion
            completion_rate = interview_data.get('questionsAnswered', 0) / interview_data.get('totalQuestions', 5)
            
            base_score = 50  # Starting score
            if avg_words_per_response >= 50:
                base_score += 20
            elif avg_words_per_response >= 25:
                base_score += 10
            
            base_score += completion_rate * 20
            
            # Behavioral adjustments
            confidence = interview_data.get('confidence', 0.5)
            engagement = interview_data.get('engagement', 0.5)
            eye_contact = interview_data.get('eyeContact', 0.5)
            
            behavioral_bonus = (confidence + engagement + eye_contact) / 3 * 10
            
            overall_score = min(100, int(base_score + behavioral_bonus))
        else:
            overall_score = 60  # Default score
        
        # Determine placement likelihood
        if overall_score >= 80:
            placement_likelihood = "High"
            performance_summary = "Strong interview performance demonstrating good communication skills and relevant experience."
        elif overall_score >= 65:
            placement_likelihood = "Medium"
            performance_summary = "Solid interview performance with some areas for development and improvement."
        else:
            placement_likelihood = "Low"
            performance_summary = "Interview performance indicates need for significant improvement before job readiness."
        
        return {
            "overall_score": overall_score,
            "placement_likelihood": placement_likelihood,
            "performance_summary": performance_summary,
            "strengths": [
                "Participated in the complete interview process",
                "Demonstrated professional communication",
                "Shared relevant experience and background",
                "Maintained engagement throughout the session"
            ],
            "development_areas": [
                "Provide more detailed and specific examples",
                "Improve response structure using STAR method",
                "Build confidence in presentation and delivery",
                "Practice common behavioral interview questions"
            ],
            "detailed_feedback": {
                "communication": {"score": max(50, overall_score - 10), "feedback": "Work on providing clearer, more structured responses"},
                "technical_knowledge": {"score": max(45, overall_score - 15), "feedback": "Continue developing technical skills and knowledge"},
                "problem_solving": {"score": max(50, overall_score - 10), "feedback": "Practice problem-solving scenarios and methodologies"},
                "confidence": {"score": max(40, overall_score - 20), "feedback": "Build confidence through practice and preparation"}
            },
            "skill_breakdown": {
                "verbal_communication": max(50, overall_score - 5),
                "confidence_level": max(40, overall_score - 15),
                "technical_competency": max(45, overall_score - 10),
                "problem_solving": max(50, overall_score - 8),
                "professionalism": max(60, overall_score)
            },
            "recommendations": [
                "Practice the STAR method (Situation, Task, Action, Result) for behavioral questions",
                "Prepare 5-7 specific examples from your experience that showcase different skills",
                "Work on confident body language and voice tone during interviews",
                "Research the company and role thoroughly before interviews",
                "Practice common interview questions with a friend or mentor",
                "Focus on quantifiable results and achievements in your responses"
            ]
        }
    
    def _parse_report_response(self, response_text):
        """Parse Gemini response for report data."""
        try:
            # Remove code block markers if present
            text = response_text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()
            
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "summary": "Interview performance analysis completed with mixed results.",
                "strengths": ["Clear communication", "Professional demeanor", "Relevant experience"],
                "development_areas": ["Provide more specific examples", "Improve technical explanations", "Practice common interview questions"],
                "detailed_feedback": {
                    "communication": {"score": 75, "feedback": "Generally clear with room for improvement"},
                    "technical_knowledge": {"score": 70, "feedback": "Adequate knowledge demonstrated"},
                    "problem_solving": {"score": 72, "feedback": "Good approach to problem-solving"},
                    "confidence": {"score": 68, "feedback": "Could benefit from more confidence"}
                },
                "skill_breakdown": {
                    "verbal_communication": 75,
                    "confidence_level": 68,
                    "technical_competency": 70,
                    "problem_solving": 72,
                    "professionalism": 80
                },
                "recommendations": [
                    "Practice storytelling using the STAR method",
                    "Research common interview questions for your role",
                    "Work on maintaining eye contact during responses",
                    "Prepare specific examples from your experience"
                ]
            }
    
    def _calculate_overall_score(self, report_data, report_analysis):
        """Calculate comprehensive overall score out of 100."""
        # Base score from answer quality (40% weight)
        response_scores = []
        for response in report_data.get('responses', []):
            if 'analysis' in response:
                response_scores.append(response['analysis'].get('score', 70))
        
        answer_score = sum(response_scores) / len(response_scores) if response_scores else 70
        
        # Completion rate score (20% weight)
        total_questions = report_data.get('total_questions', 5)
        answered = report_data.get('questions_answered', 0)
        completion_score = (answered / total_questions) * 100 if total_questions > 0 else 0
        
        # Behavioral analysis score (25% weight)
        behavioral_score = 75  # Default
        if report_data.get('facial_analysis_summary'):
            facial = report_data['facial_analysis_summary']
            behavioral_score = (
                facial.get('average_confidence', 0.75) * 100 * 0.4 +
                facial.get('average_engagement', 0.80) * 100 * 0.4 +
                facial.get('average_eye_contact', 0.70) * 100 * 0.2
            )
        
        # Response timing score (15% weight)
        avg_time = report_data.get('average_response_time', 120)
        timing_score = 100
        if avg_time < 30:  # Too quick
            timing_score = 60
        elif avg_time > 180:  # Too long
            timing_score = 70
        elif 60 <= avg_time <= 120:  # Optimal range
            timing_score = 90
        
        # Calculate weighted overall score
        overall_score = (
            answer_score * 0.40 +
            completion_score * 0.20 +
            behavioral_score * 0.25 +
            timing_score * 0.15
        )
        
        return max(0, min(100, int(overall_score)))
    
    def _determine_placement_likelihood(self, overall_score, report_data):
        """Determine placement likelihood based on performance."""
        # Base likelihood on overall score
        if overall_score >= 85:
            base_likelihood = "High"
        elif overall_score >= 70:
            base_likelihood = "Medium"
        else:
            base_likelihood = "Low"
        
        # Adjust based on completion rate
        completion_rate = report_data.get('questions_answered', 0) / report_data.get('total_questions', 5)
        if completion_rate < 0.6:  # Less than 60% completion
            if base_likelihood == "High":
                base_likelihood = "Medium"
            elif base_likelihood == "Medium":
                base_likelihood = "Low"
        
        return base_likelihood
    
    def _fallback_report_generation(self, report_data):
        """Fallback report generation when AI is not available."""
        # Calculate basic score
        response_scores = []
        for response in report_data.get('responses', []):
            if 'analysis' in response:
                response_scores.append(response['analysis'].get('score', 70))
        
        avg_score = sum(response_scores) / len(response_scores) if response_scores else 70
        completion_rate = report_data.get('questions_answered', 0) / report_data.get('total_questions', 5)
        
        overall_score = int(avg_score * 0.7 + completion_rate * 100 * 0.3)
        
        if overall_score >= 80:
            placement_likelihood = "High"
            performance_summary = "Strong interview performance with good technical knowledge and communication skills."
        elif overall_score >= 65:
            placement_likelihood = "Medium"
            performance_summary = "Solid interview performance with some areas for improvement."
        else:
            placement_likelihood = "Low"
            performance_summary = "Interview performance needs significant improvement before job readiness."
        
        return {
            "overall_score": overall_score,
            "placement_likelihood": placement_likelihood,
            "performance_summary": performance_summary,
            "strengths": ["Completed the interview", "Demonstrated relevant experience", "Professional communication"],
            "development_areas": ["Practice common interview questions", "Improve response depth", "Build confidence"],
            "detailed_feedback": {
                "communication": {"score": max(60, avg_score - 10), "feedback": "Focus on clear, structured responses"},
                "technical_knowledge": {"score": max(50, avg_score - 15), "feedback": "Continue developing technical skills"},
                "problem_solving": {"score": max(55, avg_score - 12), "feedback": "Practice problem-solving scenarios"},
                "confidence": {"score": max(50, avg_score - 20), "feedback": "Work on building interview confidence"}
            },
            "skill_breakdown": {
                "verbal_communication": max(60, avg_score - 5),
                "confidence_level": max(50, avg_score - 15),
                "technical_competency": max(55, avg_score - 10),
                "problem_solving": max(60, avg_score - 8),
                "professionalism": max(70, avg_score)
            },
            "recommendations": [
                "Practice the STAR method for behavioral questions",
                "Research the company and role thoroughly",
                "Prepare specific examples from your experience",
                "Work on maintaining confident body language",
                "Practice technical concepts relevant to the role"
            ],
            "generated_at": datetime.now().isoformat()
        }

# Initialize real analyzers
facial_analyzer = RealFacialAnalyzer()
voice_analyzer = RealVoiceAnalyzer()
gemini_client = RealGeminiClient()

# WebSocket Connection Manager for Live Analysis
class ConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.session_data = {}
    
    async def connect(self, websocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        self.session_data[session_id] = {
            "start_time": datetime.now(),
            "frame_count": 0,
            "analysis_history": []
        }
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, websocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
                if session_id in self.session_data:
                    del self.session_data[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def process_live_analysis(self, message: dict, session_id: str):
        """Process real-time analysis data with enhanced metrics."""
        analysis_type = message.get("type")
        data = message.get("data")
        timestamp = datetime.now().isoformat()
        
        try:
            if analysis_type == "facial":
                # Process facial analysis
                image_data = data.get("image")
                if image_data:
                    # Decode base64 image
                    image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    image_np = np.array(image)
                    
                    # Perform facial analysis
                    analysis = facial_analyzer.analyze_frame(image_np)
                    
                    # Add additional performance metrics
                    session_data = self.session_data[session_id]
                    
                    # Calculate trends (improvement/decline over time)
                    recent_analyses = [h for h in session_data["analysis_history"][-10:] if h["type"] == "facial_analysis"]
                    trends = self._calculate_trends(recent_analyses, "facial")
                    
                    # Add real-time feedback
                    feedback = self._generate_real_time_feedback(analysis, trends)
                    
                    result = {
                        "type": "facial_analysis",
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "data": {
                            **analysis,
                            "trends": trends,
                            "real_time_feedback": feedback,
                            "frame_number": session_data["frame_count"] + 1
                        }
                    }
                    
                    # Store in history
                    session_data["analysis_history"].append(result)
                    session_data["frame_count"] += 1
                    
                    return result
            
            elif analysis_type == "voice":
                # Process voice analysis
                audio_data = data.get("audio")
                if audio_data:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
                    
                    # Perform enhanced voice analysis
                    analysis = voice_analyzer.analyze_audio(audio_bytes)
                    
                    # Add voice-specific performance metrics
                    session_data = self.session_data[session_id]
                    
                    # Calculate voice trends
                    recent_voice_analyses = [h for h in session_data["analysis_history"][-10:] if h["type"] == "voice_analysis"]
                    voice_trends = self._calculate_voice_trends(recent_voice_analyses)
                    
                    # Generate voice-specific feedback
                    voice_feedback = self._generate_voice_feedback(analysis, voice_trends)
                    
                    # Add speaking statistics
                    speaking_stats = self._calculate_speaking_stats(recent_voice_analyses, analysis)
                    
                    result = {
                        "type": "voice_analysis",
                        "session_id": session_id,
                        "timestamp": timestamp,
                        "data": {
                            **analysis,
                            "trends": voice_trends,
                            "feedback": voice_feedback,
                            "speaking_stats": speaking_stats
                        }
                    }
                    
                    # Store in history
                    session_data["analysis_history"].append(result)
                    
                    return result
            
            elif analysis_type == "heartbeat":
                # Simple heartbeat to keep connection alive
                return {
                    "type": "heartbeat_response",
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "status": "active"
                }
                
        except Exception as e:
            logger.error(f"Live analysis error: {str(e)}")
            return {
                "type": "error",
                "session_id": session_id,
                "timestamp": timestamp,
                "error": str(e)
            }
        
        return {
            "type": "unknown",
            "session_id": session_id,
            "timestamp": timestamp,
            "message": "Unknown analysis type"
        }
    
    async def send_analysis_result(self, result: dict, session_id: str):
        """Send analysis result to all connected clients in the session."""
        if session_id in self.active_connections:
            message = json.dumps(result)
            for connection in self.active_connections[session_id].copy():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message: {str(e)}")
                    self.active_connections[session_id].remove(connection)
    
    def get_session_summary(self, session_id: str):
        """Get summary of analysis for a session."""
        if session_id not in self.session_data:
            return None
        
        session = self.session_data[session_id]
        history = session["analysis_history"]
        
        # Calculate averages
        facial_analyses = [h for h in history if h["type"] == "facial_analysis"]
        voice_analyses = [h for h in history if h["type"] == "voice_analysis"]
        
        summary = {
            "session_id": session_id,
            "duration": (datetime.now() - session["start_time"]).total_seconds(),
            "total_frames": session["frame_count"],
            "facial_analyses": len(facial_analyses),
            "voice_analyses": len(voice_analyses)
        }
        
        # Calculate average metrics
        if facial_analyses:
            avg_confidence = np.mean([a["data"].get("confidence", 0) for a in facial_analyses])
            avg_engagement = np.mean([a["data"].get("engagement", 0) for a in facial_analyses])
            avg_eye_contact = np.mean([a["data"].get("eye_contact", 0) for a in facial_analyses])
            
            summary["average_metrics"] = {
                "confidence": float(avg_confidence),
                "engagement": float(avg_engagement),
                "eye_contact": float(avg_eye_contact)
            }
        
        if voice_analyses:
            avg_clarity = np.mean([a["data"].get("clarity", 0) for a in voice_analyses if isinstance(a["data"].get("clarity"), (int, float))])
            summary["voice_metrics"] = {
                "average_clarity": float(avg_clarity) if not np.isnan(avg_clarity) else 0.8
            }
        
        return summary
    
    def _calculate_trends(self, recent_analyses, analysis_type):
        """Calculate performance trends over recent analyses."""
        if len(recent_analyses) < 2:
            return {"confidence": "stable", "engagement": "stable", "eye_contact": "stable"}
        
        trends = {}
        
        if analysis_type == "facial":
            metrics = ["confidence", "engagement", "eye_contact"]
            
            for metric in metrics:
                values = [a["data"].get(metric, 0) for a in recent_analyses if "data" in a]
                if len(values) >= 3:
                    # Calculate trend using linear regression
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    if slope > 0.02:
                        trends[metric] = "improving"
                    elif slope < -0.02:
                        trends[metric] = "declining"
                    else:
                        trends[metric] = "stable"
                else:
                    trends[metric] = "stable"
        
        return trends
    
    def _generate_real_time_feedback(self, analysis, trends):
        """Generate real-time feedback based on current analysis and trends."""
        feedback = {
            "suggestions": [],
            "alerts": [],
            "encouragements": []
        }
        
        # Confidence feedback
        confidence = analysis.get("confidence", 0)
        if confidence < 0.4:
            feedback["suggestions"].append("Try to sit up straighter and maintain steady eye contact")
            feedback["alerts"].append("Low confidence detected")
        elif confidence > 0.8:
            feedback["encouragements"].append("Great confidence level!")
        
        # Engagement feedback
        engagement = analysis.get("engagement", 0)
        if engagement < 0.4:
            feedback["suggestions"].append("Look directly at the camera and show more interest")
            feedback["alerts"].append("Low engagement detected")
        elif engagement > 0.8:
            feedback["encouragements"].append("Excellent engagement!")
        
        # Eye contact feedback
        eye_contact = analysis.get("eye_contact", 0)
        if eye_contact < 0.3:
            feedback["suggestions"].append("Make more direct eye contact with the camera")
            feedback["alerts"].append("Poor eye contact")
        elif eye_contact > 0.8:
            feedback["encouragements"].append("Perfect eye contact!")
        
        # Trend-based feedback
        if trends.get("confidence") == "declining":
            feedback["suggestions"].append("Your confidence seems to be dropping - take a deep breath")
        elif trends.get("confidence") == "improving":
            feedback["encouragements"].append("Your confidence is improving!")
        
        # Emotion-based feedback
        emotions = analysis.get("emotions", {})
        if emotions.get("nervous", 0) > 0.5:
            feedback["suggestions"].append("You seem nervous - try to relax and speak slowly")
        elif emotions.get("happy", 0) > 0.4:
            feedback["encouragements"].append("Great positive energy!")
        
        return feedback
    
    def _calculate_voice_trends(self, recent_voice_analyses):
        """Calculate voice performance trends."""
        if len(recent_voice_analyses) < 2:
            return {"clarity": "stable", "confidence": "stable", "pace": "stable"}
        
        trends = {}
        
        # Clarity trend
        clarity_values = [a["data"].get("clarity", 0.8) for a in recent_voice_analyses if "data" in a]
        if len(clarity_values) >= 3:
            x = np.arange(len(clarity_values))
            slope = np.polyfit(x, clarity_values, 1)[0]
            if slope > 0.02:
                trends["clarity"] = "improving"
            elif slope < -0.02:
                trends["clarity"] = "declining"
            else:
                trends["clarity"] = "stable"
        else:
            trends["clarity"] = "stable"
        
        # Confidence trend
        confidence_values = [a["data"].get("confidence_score", 0.7) for a in recent_voice_analyses if "data" in a]
        if len(confidence_values) >= 3:
            x = np.arange(len(confidence_values))
            slope = np.polyfit(x, confidence_values, 1)[0]
            if slope > 0.02:
                trends["confidence"] = "improving"
            elif slope < -0.02:
                trends["confidence"] = "declining"
            else:
                trends["confidence"] = "stable"
        else:
            trends["confidence"] = "stable"
        
        # Speaking pace trend
        pace_values = []
        for a in recent_voice_analyses:
            if "data" in a and "pace" in a["data"]:
                pace = a["data"]["pace"]
                if pace == "slow":
                    pace_values.append(0.3)
                elif pace == "moderate":
                    pace_values.append(0.6)
                else:  # fast
                    pace_values.append(0.9)
        
        if len(pace_values) >= 3:
            recent_pace = np.mean(pace_values[-3:])
            earlier_pace = np.mean(pace_values[-6:-3]) if len(pace_values) >= 6 else recent_pace
            
            if abs(recent_pace - 0.6) < abs(earlier_pace - 0.6):
                trends["pace"] = "improving"
            elif abs(recent_pace - 0.6) > abs(earlier_pace - 0.6):
                trends["pace"] = "declining"
            else:
                trends["pace"] = "stable"
        else:
            trends["pace"] = "stable"
        
        return trends
    
    def _generate_voice_feedback(self, analysis, trends):
        """Generate voice-specific feedback."""
        feedback = {
            "suggestions": [],
            "alerts": [],
            "encouragements": []
        }
        
        # Clarity feedback
        clarity = analysis.get("clarity", 0.8)
        if clarity < 0.5:
            feedback["suggestions"].append("Speak more clearly and articulate your words")
            feedback["alerts"].append("Low speech clarity detected")
        elif clarity > 0.9:
            feedback["encouragements"].append("Excellent speech clarity!")
        
        # Pace feedback
        pace = analysis.get("pace", "moderate")
        if pace == "fast":
            feedback["suggestions"].append("Try to slow down your speaking pace")
            feedback["alerts"].append("Speaking too fast")
        elif pace == "slow":
            feedback["suggestions"].append("You can speak a bit faster to sound more engaging")
        elif pace == "moderate":
            feedback["encouragements"].append("Perfect speaking pace!")
        
        # Volume feedback
        volume = analysis.get("volume", "appropriate")
        if volume == "low":
            feedback["suggestions"].append("Speak a bit louder to project confidence")
            feedback["alerts"].append("Voice volume is too low")
        elif volume == "high":
            feedback["suggestions"].append("Lower your voice slightly for better comfort")
        elif volume == "appropriate":
            feedback["encouragements"].append("Great voice volume!")
        
        # Confidence feedback
        confidence = analysis.get("confidence_score", 0.7)
        if confidence < 0.4:
            feedback["suggestions"].append("Try to sound more confident and steady")
            feedback["alerts"].append("Voice confidence is low")
        elif confidence > 0.8:
            feedback["encouragements"].append("Very confident voice!")
        
        # Emotional state feedback
        emotions = analysis.get("emotional_state", {})
        if emotions.get("nervous", 0) > 0.6:
            feedback["suggestions"].append("Take a deep breath and try to relax your voice")
        elif emotions.get("confident", 0) > 0.6:
            feedback["encouragements"].append("Your voice sounds very confident!")
        
        # Trend-based feedback
        if trends.get("clarity") == "declining":
            feedback["suggestions"].append("Your speech clarity is declining - slow down and enunciate")
        elif trends.get("clarity") == "improving":
            feedback["encouragements"].append("Your speech is becoming clearer!")
        
        return feedback
    
    def _calculate_speaking_stats(self, recent_analyses, current_analysis):
        """Calculate speaking statistics."""
        stats = {
            "average_clarity": 0.8,
            "average_confidence": 0.7,
            "dominant_pace": "moderate",
            "total_speaking_time": 0,
            "words_per_minute": 120
        }
        
        if not recent_analyses:
            return stats
        
        # Calculate averages
        clarity_values = [a["data"].get("clarity", 0.8) for a in recent_analyses if "data" in a]
        confidence_values = [a["data"].get("confidence_score", 0.7) for a in recent_analyses if "data" in a]
        
        if clarity_values:
            stats["average_clarity"] = float(np.mean(clarity_values))
        if confidence_values:
            stats["average_confidence"] = float(np.mean(confidence_values))
        
        # Dominant pace
        pace_counts = {"slow": 0, "moderate": 0, "fast": 0}
        for a in recent_analyses:
            if "data" in a and "pace" in a["data"]:
                pace = a["data"]["pace"]
                if pace in pace_counts:
                    pace_counts[pace] += 1
        
        stats["dominant_pace"] = max(pace_counts.items(), key=lambda x: x[1])[0]
        
        # Estimate speaking time (rough approximation)
        stats["total_speaking_time"] = len(recent_analyses) * 2  # Assume 2 seconds per analysis
        
        # Words per minute (from speech rate if available)
        if "speech_rate" in current_analysis:
            stats["words_per_minute"] = int(current_analysis["speech_rate"])
        
        return stats

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    image_data: Optional[str] = None  # Base64 encoded image
    audio_data: Optional[str] = None  # Base64 encoded audio
    session_id: str

class QuestionRequest(BaseModel):
    job_role: str
    resume_data: Optional[dict] = None
    difficulty: Optional[str] = "medium"
    category: Optional[str] = "behavioral"
    session_id: str

class AnalysisResponse(BaseModel):
    confidence: float
    engagement: float
    eye_contact: float
    emotions: dict
    session_id: str

class QuestionResponse(BaseModel):
    question: str
    category: str
    difficulty: str
    expected_duration: str
    evaluation_criteria: List[str]

class InterviewReportRequest(BaseModel):
    session_id: str
    job_role: str
    total_questions: int
    questions_answered: int
    questions_skipped: int
    average_response_time: float
    responses: List[dict]  # Individual question responses with scores
    facial_analysis_summary: Optional[dict] = None
    voice_analysis_summary: Optional[dict] = None

class InterviewReportResponse(BaseModel):
    session_id: str
    overall_score: int  # Out of 100
    placement_likelihood: str  # High/Medium/Low
    performance_summary: str
    strengths: List[str]
    development_areas: List[str]
    detailed_feedback: dict
    skill_breakdown: dict
    recommendations: List[str]
    generated_at: str

def register_routes(app: FastAPI):
    
    @app.get("/")
    async def root():
        return {"message": "AI Interview Prep Services API", "status": "online"}
    
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "services": {
                "facial_analysis": "online",
                "voice_analysis": "online", 
                "gemini_integration": "online"
            }
        }
    
    @app.post("/analyze/facial", response_model=AnalysisResponse)
    async def analyze_facial(request: AnalysisRequest):
        """Analyze facial expressions from image data"""
        try:
            if not request.image_data:
                raise HTTPException(status_code=400, detail="No image data provided")
            
            # Decode base64 image
            image_bytes = base64.b64decode(request.image_data.split(',')[1] if ',' in request.image_data else request.image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Perform facial analysis
            analysis = facial_analyzer.analyze_frame(image_np)
            
            return AnalysisResponse(
                confidence=analysis.get("confidence", 0.75),
                engagement=analysis.get("engagement", 0.80),
                eye_contact=analysis.get("eye_contact", 0.70),
                emotions=analysis.get("emotions", {"neutral": 0.8}),
                session_id=request.session_id
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Facial analysis failed: {str(e)}")
    
    @app.post("/analyze/voice")
    async def analyze_voice(request: AnalysisRequest):
        """Analyze voice/speech patterns from audio data"""
        try:
            if not request.audio_data:
                raise HTTPException(status_code=400, detail="No audio data provided")
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(request.audio_data.split(',')[1] if ',' in request.audio_data else request.audio_data)
            
            # Perform voice analysis
            analysis = voice_analyzer.analyze_audio(audio_bytes)
            
            return {
                "clarity": analysis.get("clarity", 0.85),
                "pace": analysis.get("pace", "moderate"),
                "tone": analysis.get("tone", "confident"),
                "volume": analysis.get("volume", "appropriate"),
                "session_id": request.session_id
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Voice analysis failed: {str(e)}")
    
    @app.post("/generate/question", response_model=QuestionResponse)
    async def generate_question(request: QuestionRequest):
        """Generate interview question using Gemini AI"""
        try:
            # Generate question based on job role and resume
            question_data = await gemini_client.generate_question(
                job_role=request.job_role,
                resume_data=request.resume_data,
                difficulty=request.difficulty,
                category=request.category
            )
            
            return QuestionResponse(
                question=question_data.get("question", f"Tell me about your experience in {request.job_role}."),
                category=question_data.get("category", "behavioral"),
                difficulty=request.difficulty,
                expected_duration=question_data.get("duration", "2-3 minutes"),
                evaluation_criteria=question_data.get("criteria", ["Clarity", "Relevance", "Depth"])
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")
    
    @app.post("/evaluate/answer")
    async def evaluate_answer(answer_text: str, question: str, session_id: str):
        """Evaluate an interview answer using Gemini AI"""
        try:
            evaluation = await gemini_client.evaluate_answer(answer_text, question)
            
            return {
                "score": evaluation.get("score", 75),
                "feedback": evaluation.get("feedback", "Good answer with room for improvement."),
                "strengths": evaluation.get("strengths", ["Clear communication"]),
                "improvements": evaluation.get("improvements", ["Add specific examples"]),
                "session_id": session_id
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Answer evaluation failed: {str(e)}")
    
    @app.post("/generate/report", response_model=InterviewReportResponse)
    async def generate_interview_report(request: InterviewReportRequest):
        """Generate comprehensive interview report"""
        try:
            # Generate report using AI analysis
            report_data = await gemini_client.generate_interview_report({
                "session_id": request.session_id,
                "job_role": request.job_role,
                "total_questions": request.total_questions,
                "questions_answered": request.questions_answered,
                "questions_skipped": request.questions_skipped,
                "average_response_time": request.average_response_time,
                "responses": request.responses,
                "facial_analysis_summary": request.facial_analysis_summary,
                "voice_analysis_summary": request.voice_analysis_summary
            })
            
            return InterviewReportResponse(
                session_id=request.session_id,
                overall_score=report_data.get("overall_score", 70),
                placement_likelihood=report_data.get("placement_likelihood", "Medium"),
                performance_summary=report_data.get("performance_summary", "Interview completed successfully."),
                strengths=report_data.get("strengths", []),
                development_areas=report_data.get("development_areas", []),
                detailed_feedback=report_data.get("detailed_feedback", {}),
                skill_breakdown=report_data.get("skill_breakdown", {}),
                recommendations=report_data.get("recommendations", []),
                generated_at=report_data.get("generated_at", datetime.now().isoformat())
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    
    @app.post("/generate/comprehensive-evaluation", response_model=InterviewReportResponse)
    async def generate_comprehensive_evaluation(interview_data: dict):
        """Generate comprehensive interview evaluation based on full conversation history"""
        try:
            logger.info(f"Generating comprehensive evaluation for session: {interview_data.get('sessionId', 'unknown')}")
            
            # Generate comprehensive evaluation using full conversation history
            evaluation_data = await gemini_client.generate_comprehensive_evaluation(interview_data)
            
            logger.info(f"Evaluation completed with score: {evaluation_data.get('overall_score', 'unknown')}")
            
            return InterviewReportResponse(
                session_id=evaluation_data.get("session_id", ""),
                overall_score=evaluation_data.get("overall_score", 70),
                placement_likelihood=evaluation_data.get("placement_likelihood", "Medium"),
                performance_summary=evaluation_data.get("performance_summary", "Interview evaluation completed."),
                strengths=evaluation_data.get("strengths", []),
                development_areas=evaluation_data.get("development_areas", []),
                detailed_feedback=evaluation_data.get("detailed_feedback", {}),
                skill_breakdown=evaluation_data.get("skill_breakdown", {}),
                recommendations=evaluation_data.get("recommendations", []),
                generated_at=evaluation_data.get("generated_at", datetime.now().isoformat())
            )
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Comprehensive evaluation failed: {str(e)}")
    
    @app.post("/upload/resume")
    async def upload_resume(file: UploadFile = File(...)):
        """Process uploaded resume file"""
        try:
            content = await file.read()
            
            # Process resume based on file type
            if file.content_type == "application/pdf":
                # Handle PDF processing
                resume_data = {"type": "pdf", "processed": True}
            elif file.content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                # Handle Word document processing
                resume_data = {"type": "docx", "processed": True}
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
            
            return {
                "status": "processed",
                "filename": file.filename,
                "size": len(content),
                "data": resume_data
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Resume processing failed: {str(e)}")
    
    @app.get("/session/{session_id}/summary")
    async def get_session_analysis_summary(session_id: str):
        """Get analysis summary for a session."""
        from main import manager  # Import the connection manager
        
        try:
            summary = manager.get_session_summary(session_id)
            if summary:
                return summary
            else:
                raise HTTPException(status_code=404, detail="Session not found or no analysis data available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get session summary: {str(e)}")
