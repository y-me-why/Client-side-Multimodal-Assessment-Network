import React, { useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree, useLoader } from '@react-three/fiber';
import { 
  OrbitControls,
} from '@react-three/drei';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { Box as MUIBox, Typography, IconButton } from '@mui/material';
import { VolumeUp, VolumeOff } from '@mui/icons-material';

interface Avatar3DProps {
  isListening?: boolean;
  isSpeaking?: boolean;
  currentText?: string;
  userPosition?: { x: number; y: number; z: number };
  onAvatarReady?: () => void;
}


const GLTFModel: React.FC<{
  isListening: boolean;
  isSpeaking: boolean;
  userPosition: { x: number; y: number; z: number };
  mouthShape: number;
  onLoadingChange: (loading: boolean) => void;
  onError: (error: string | null) => void;
}> = ({ isListening, isSpeaking, userPosition, mouthShape, onLoadingChange, onError }) => {
  const [gltf, setGltf] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const modelRef = useRef<THREE.Group>(null);
  const [model, setModel] = useState<THREE.Group | null>(null);
  
  useEffect(() => {
    const loader = new GLTFLoader();
    loader.load(
      "https://models.readyplayer.me/68987a7f7c6c17df6663f495.glb",
      (loadedGltf) => {
        setGltf(loadedGltf);
        setLoading(false);
        onLoadingChange(false);
      },
      (progress) => {
        console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
      },
      (err) => {
        console.error('Error loading GLTF model:', err);
        const errorMsg = 'Failed to load avatar model';
        setError(errorMsg);
        setLoading(false);
        onLoadingChange(false);
        onError(errorMsg);
      }
    );
  }, []);

  useEffect(() => {
    if (gltf && gltf.scene) {
      const clonedScene = gltf.scene.clone();
      clonedScene.position.set(0, -1.5, 0);
      clonedScene.scale.set(1.2, 1.2, 1.2);
      setModel(clonedScene);
    }
  }, [gltf]);

  // Animation loop for the GLTF model
  useFrame((state) => {
    if (!modelRef.current) return;

    const time = state.clock.getElapsedTime();
    
    // Breathing animation
    const breathingScale = 1 + Math.sin(time * 2) * 0.01;
    modelRef.current.scale.y = 1.2 * breathingScale;

    // Listening animation - subtle head nod
    if (isListening) {
      modelRef.current.rotation.x = Math.sin(time * 3) * 0.02;
    } else {
      modelRef.current.rotation.x = THREE.MathUtils.lerp(
        modelRef.current.rotation.x,
        0,
        0.1
      );
    }

    // Speaking animation - slight head movement
    if (isSpeaking) {
      modelRef.current.rotation.y = Math.sin(time * 5) * 0.01;
      modelRef.current.rotation.z = Math.sin(time * 4) * 0.01;
    } else {
      modelRef.current.rotation.y = THREE.MathUtils.lerp(
        modelRef.current.rotation.y,
        0,
        0.1
      );
      modelRef.current.rotation.z = THREE.MathUtils.lerp(
        modelRef.current.rotation.z,
        0,
        0.1
      );
    }
  });

  if (loading) {
    return (
      <group>
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[1, 1.5, 0.8]} />
          <meshStandardMaterial color="#e0e0e0" opacity={0.7} transparent />
        </mesh>
        {/* Loading indicator */}
        <mesh position={[0, 1, 0]}>
          <sphereGeometry args={[0.1]} />
          <meshStandardMaterial color="#2196f3" emissive="#2196f3" emissiveIntensity={0.5} />
        </mesh>
      </group>
    );
  }
  
  if (error || !model) {
    return (
      <group>
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[1, 1.5, 0.8]} />
          <meshStandardMaterial color="#ff5722" opacity={0.8} transparent />
        </mesh>
        {/* Error indicator */}
        <mesh position={[0, 1, 0]}>
          <sphereGeometry args={[0.1]} />
          <meshStandardMaterial color="#f44336" emissive="#f44336" emissiveIntensity={0.5} />
        </mesh>
      </group>
    );
  }

  return (
    <group ref={modelRef}>
      <primitive object={model} />
    </group>
  );
};

const Avatar3D: React.FC<Avatar3DProps> = ({
  isListening = false,
  isSpeaking = false,
  currentText = '',
  userPosition = { x: 0, y: 0, z: 5 },
  onAvatarReady,
}) => {
  const [mouthShape, setMouthShape] = useState(0);
  const [speechSynthesis, setSpeechSynthesis] = useState<SpeechSynthesis | null>(null);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [currentUtterance, setCurrentUtterance] = useState<SpeechSynthesisUtterance | null>(null);
  const [avatarLoading, setAvatarLoading] = useState(true);
  const [avatarError, setAvatarError] = useState<string | null>(null);

  // Initialize speech synthesis
  useEffect(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      setSpeechSynthesis(window.speechSynthesis);
      onAvatarReady?.();
    }
  }, [onAvatarReady]);

  // Simple lip sync based on speech synthesis events
  useEffect(() => {
    if (!currentText || !speechSynthesis || !isAudioEnabled) return;

    const utterance = new SpeechSynthesisUtterance(currentText);
    utterance.rate = 0.9;
    utterance.pitch = 1.1;
    utterance.volume = 0.8;

    // Try to get a female voice
    const voices = speechSynthesis.getVoices();
    const femaleVoice = voices.find(voice => 
      voice.name.toLowerCase().includes('female') || 
      voice.name.toLowerCase().includes('zira') ||
      voice.name.toLowerCase().includes('hazel')
    );
    if (femaleVoice) {
      utterance.voice = femaleVoice;
    }

    let animationInterval: NodeJS.Timeout;

    utterance.onstart = () => {
      // Animate mouth during speech
      animationInterval = setInterval(() => {
        const randomShape = Math.random() * 0.8 + 0.2;
        setMouthShape(randomShape);
      }, 150);
    };

    utterance.onend = () => {
      clearInterval(animationInterval);
      setMouthShape(0);
      setCurrentUtterance(null);
    };

    utterance.onerror = () => {
      clearInterval(animationInterval);
      setMouthShape(0);
      setCurrentUtterance(null);
    };

    setCurrentUtterance(utterance);
    speechSynthesis.speak(utterance);

    return () => {
      clearInterval(animationInterval);
      speechSynthesis.cancel();
    };
  }, [currentText, speechSynthesis, isAudioEnabled]);

  const toggleAudio = () => {
    setIsAudioEnabled(!isAudioEnabled);
    if (currentUtterance && speechSynthesis) {
      speechSynthesis.cancel();
    }
  };

  return (
    <MUIBox component="div" sx={{ position: 'relative', width: '100%', height: '400px' }}>
      {/* Loading/Error Status */}
      {(avatarLoading || avatarError) && (
        <MUIBox
          sx={{
            position: 'absolute',
            top: 16,
            left: 16,
            zIndex: 10,
            backgroundColor: avatarLoading ? 'rgba(33, 150, 243, 0.9)' : 'rgba(244, 67, 54, 0.9)',
            color: 'white',
            px: 2,
            py: 1,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
          }}
        >
          <MUIBox
            sx={{
              width: 8,
              height: 8,
              backgroundColor: 'white',
              borderRadius: '50%',
              animation: avatarLoading ? 'pulse 1s infinite' : 'none',
              '@keyframes pulse': {
                '0%': { opacity: 1 },
                '50%': { opacity: 0.5 },
                '100%': { opacity: 1 },
              },
            }}
          />
          <Typography variant="caption">
            {avatarLoading ? 'Loading Avatar...' : 'Avatar Load Error'}
          </Typography>
        </MUIBox>
      )}
      {/* Audio Control */}
      <IconButton
        onClick={toggleAudio}
        sx={{
          position: 'absolute',
          top: 8,
          right: 8,
          zIndex: 10,
          backgroundColor: 'rgba(255, 255, 255, 0.8)',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.9)',
          },
        }}
      >
        {isAudioEnabled ? <VolumeUp /> : <VolumeOff />}
      </IconButton>

      <Canvas
        camera={{ position: [0, 0, 5], fov: 45 }}
        style={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
      >
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <pointLight position={[-5, 5, 5]} intensity={0.4} />
        
        <GLTFModel
          isListening={isListening}
          isSpeaking={isSpeaking}
          userPosition={userPosition}
          mouthShape={mouthShape}
          onLoadingChange={setAvatarLoading}
          onError={setAvatarError}
        />
        
        {/* Subtle camera controls */}
        <OrbitControls 
          enableZoom={false}
          enablePan={false}
          enableRotate={true}
          maxPolarAngle={Math.PI / 2}
          minPolarAngle={Math.PI / 4}
          maxAzimuthAngle={Math.PI / 6}
          minAzimuthAngle={-Math.PI / 6}
        />
      </Canvas>

      {/* Status indicator */}
      {isSpeaking && (
        <MUIBox
          sx={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            backgroundColor: 'rgba(76, 175, 80, 0.9)',
            color: 'white',
            px: 2,
            py: 1,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
          }}
        >
          <MUIBox
            sx={{
              width: 8,
              height: 8,
              backgroundColor: 'white',
              borderRadius: '50%',
              animation: 'pulse 1s infinite',
              '@keyframes pulse': {
                '0%': { opacity: 1 },
                '50%': { opacity: 0.5 },
                '100%': { opacity: 1 },
              },
            }}
          />
          <Typography variant="caption">Speaking...</Typography>
        </MUIBox>
      )}

      {isListening && (
        <MUIBox
          sx={{
            position: 'absolute',
            bottom: 16,
            right: 16,
            backgroundColor: 'rgba(33, 150, 243, 0.9)',
            color: 'white',
            px: 2,
            py: 1,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
          }}
        >
          <MUIBox
            sx={{
              width: 8,
              height: 8,
              backgroundColor: 'white',
              borderRadius: '50%',
              animation: 'pulse 1s infinite',
              '@keyframes pulse': {
                '0%': { opacity: 1 },
                '50%': { opacity: 0.5 },
                '100%': { opacity: 1 },
              },
            }}
          />
          <Typography variant="caption">Listening...</Typography>
        </MUIBox>
      )}
    </MUIBox>
  );
};

export default Avatar3D;