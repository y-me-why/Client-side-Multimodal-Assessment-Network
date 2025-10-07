import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FacialExpressionAnalyzer:
    def __init__(self):
        """Initialize facial expression analyzer with MediaPipe."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Key landmark indices for expression analysis
        self.landmark_indices = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        }
        
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame for facial expressions.
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            Dictionary containing expression analysis results
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {"detected": False, "expressions": {}}
            
            # Extract landmarks
            landmarks = results.multi_face_landmarks[0]
            expression_scores = self._calculate_expressions(landmarks, frame.shape)
            
            return {
                "detected": True,
                "expressions": expression_scores,
                "confidence_level": self._calculate_confidence(expression_scores),
                "engagement_score": self._calculate_engagement(expression_scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {str(e)}")
            return {"detected": False, "error": str(e)}
    
    def _calculate_expressions(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Calculate expression scores from facial landmarks."""
        h, w = frame_shape[:2]
        
        # Convert landmarks to pixel coordinates
        landmark_points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmark_points.append((x, y))
        
        expressions = {}
        
        # Eye openness (confidence indicator)
        expressions['eye_openness'] = self._calculate_eye_openness(landmark_points)
        
        # Smile detection
        expressions['smile_intensity'] = self._calculate_smile_intensity(landmark_points)
        
        # Eyebrow position (surprise/concern)
        expressions['eyebrow_raise'] = self._calculate_eyebrow_position(landmark_points)
        
        # Overall facial tension
        expressions['facial_tension'] = self._calculate_facial_tension(landmark_points)
        
        return expressions
    
    def _calculate_eye_openness(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate eye openness ratio (0-1 scale)."""
        # Simplified eye aspect ratio calculation
        # In a real implementation, you'd use proper eye landmarks
        return 0.8  # Placeholder
    
    def _calculate_smile_intensity(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate smile intensity (0-1 scale)."""
        # Simplified smile detection based on mouth corner positions
        return 0.5  # Placeholder
    
    def _calculate_eyebrow_position(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate eyebrow raise level (0-1 scale)."""
        return 0.3  # Placeholder
    
    def _calculate_facial_tension(self, landmarks: List[Tuple[int, int]]) -> float:
        """Calculate overall facial tension (0-1 scale)."""
        return 0.2  # Placeholder
    
    def _calculate_confidence(self, expressions: Dict[str, float]) -> float:
        """Calculate overall confidence score based on expressions."""
        eye_confidence = expressions.get('eye_openness', 0) * 0.4
        smile_confidence = expressions.get('smile_intensity', 0) * 0.3
        tension_penalty = expressions.get('facial_tension', 0) * -0.2
        eyebrow_factor = min(expressions.get('eyebrow_raise', 0), 0.5) * 0.1
        
        confidence = max(0, min(1, eye_confidence + smile_confidence + tension_penalty + eyebrow_factor))
        return confidence
    
    def _calculate_engagement(self, expressions: Dict[str, float]) -> float:
        """Calculate engagement score based on facial activity."""
        # Higher engagement with appropriate eye contact and facial expressiveness
        eye_engagement = expressions.get('eye_openness', 0) * 0.5
        expression_variety = (expressions.get('smile_intensity', 0) + 
                            expressions.get('eyebrow_raise', 0)) * 0.3
        
        engagement = max(0, min(1, eye_engagement + expression_variety))
        return engagement