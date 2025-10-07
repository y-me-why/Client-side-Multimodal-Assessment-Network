import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Paper,
  LinearProgress,
  Alert,
  Chip,
  IconButton,
} from '@mui/material';
import {
  CloudUpload,
  Description,
  Delete,
  CheckCircle,
} from '@mui/icons-material';
import * as pdfjsLib from 'pdfjs-dist';
import mammoth from 'mammoth';

// Set up PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

interface ResumeUploadProps {
  onResumeAnalyzed: (resumeData: {
    text: string;
    skills: string[];
    experience: string[];
    projects: string[];
  }) => void;
}

const ResumeUpload: React.FC<ResumeUploadProps> = ({ onResumeAnalyzed }) => {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [resumeData, setResumeData] = useState<any>(null);

  const extractTextFromPDF = async (file: File): Promise<string> => {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = '';

    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map((item: any) => item.str)
        .join(' ');
      fullText += pageText + ' ';
    }

    return fullText.trim();
  };

  const extractTextFromWord = async (file: File): Promise<string> => {
    const arrayBuffer = await file.arrayBuffer();
    const result = await mammoth.extractRawText({ arrayBuffer });
    return result.value;
  };

  const analyzeResumeText = (text: string) => {
    // Simple text analysis - in real implementation, this would be done by Gemini AI
    const lines = text.toLowerCase().split(/[\n\r.]+/);
    
    const skillKeywords = [
      'javascript', 'typescript', 'react', 'node.js', 'python', 'java', 'c++',
      'html', 'css', 'sql', 'mongodb', 'postgresql', 'aws', 'azure', 'docker',
      'kubernetes', 'git', 'agile', 'scrum', 'machine learning', 'ai', 'data science'
    ];
    
    const experienceKeywords = [
      'years of experience', 'worked at', 'employment', 'position', 'role',
      'responsible for', 'managed', 'led', 'developed', 'implemented'
    ];

    const skills = skillKeywords.filter(skill => 
      text.toLowerCase().includes(skill)
    );

    const experience = lines.filter(line => 
      experienceKeywords.some(keyword => line.includes(keyword))
    ).slice(0, 5); // Take first 5 experience-related lines

    const projects = lines.filter(line => 
      line.includes('project') || line.includes('built') || line.includes('created')
    ).slice(0, 3); // Take first 3 project-related lines

    return {
      text,
      skills,
      experience,
      projects
    };
  };

  const processFile = useCallback(async (file: File) => {
    setIsProcessing(true);
    setError(null);
    setSuccess(false);

    try {
      let text = '';
      
      if (file.type === 'application/pdf') {
        text = await extractTextFromPDF(file);
      } else if (
        file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
        file.type === 'application/msword'
      ) {
        text = await extractTextFromWord(file);
      } else {
        throw new Error('Unsupported file format');
      }

      const analyzed = analyzeResumeText(text);
      setResumeData(analyzed);
      setSuccess(true);
      onResumeAnalyzed(analyzed);

    } catch (err) {
      setError(`Failed to process resume: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
    }
  }, [onResumeAnalyzed]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setFile(file);
      processFile(file);
    }
  }, [processFile]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const removeFile = () => {
    setFile(null);
    setResumeData(null);
    setError(null);
    setSuccess(false);
  };

  return (
    <Box component="div" sx={{ mb: 3 }}>
      <Typography variant="h6" gutterBottom>
        Upload Your Resume (Optional)
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Upload your resume to get personalized questions based on your experience and skills
      </Typography>

      {!file && (
        <Paper
          {...getRootProps()}
          sx={{
            p: 4,
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            backgroundColor: isDragActive ? 'primary.light' : 'grey.50',
            cursor: 'pointer',
            textAlign: 'center',
            transition: 'all 0.3s ease',
            '&:hover': {
              borderColor: 'primary.main',
              backgroundColor: 'primary.light',
            },
          }}
        >
          <input {...getInputProps()} />
          <CloudUpload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            {isDragActive ? 'Drop your resume here' : 'Drag & drop your resume'}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            or click to browse files
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Supports PDF, DOC, DOCX files (max 10MB)
          </Typography>
        </Paper>
      )}

      {file && (
        <Paper sx={{ p: 3 }}>
          <Box component="div" sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Description sx={{ mr: 2, color: 'primary.main' }} />
            <Box component="div" sx={{ flexGrow: 1 }}>
              <Typography variant="subtitle1">{file.name}</Typography>
              <Typography variant="caption" color="text.secondary">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </Typography>
            </Box>
            {success && <CheckCircle sx={{ color: 'success.main', mr: 1 }} />}
            <IconButton onClick={removeFile} color="error">
              <Delete />
            </IconButton>
          </Box>

          {isProcessing && (
            <Box component="div" sx={{ mb: 2 }}>
              <Typography variant="body2" sx={{ mb: 1 }}>
                Processing resume...
              </Typography>
              <LinearProgress />
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {success && resumeData && (
            <Box component="div">
              <Alert severity="success" sx={{ mb: 2 }}>
                Resume processed successfully! Your interview questions will be personalized.
              </Alert>
              
              <Box component="div" sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Detected Skills:
                </Typography>
                <Box component="div" sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {resumeData.skills.map((skill: string, index: number) => (
                    <Chip key={index} label={skill} size="small" color="primary" />
                  ))}
                </Box>
              </Box>

              {resumeData.experience.length > 0 && (
                <Box component="div" sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Experience Highlights:
                  </Typography>
                  {resumeData.experience.slice(0, 2).map((exp: string, index: number) => (
                    <Typography key={index} variant="body2" color="text.secondary">
                      • {exp.substring(0, 100)}...
                    </Typography>
                  ))}
                </Box>
              )}
            </Box>
          )}
        </Paper>
      )}
    </Box>
  );
};

export default ResumeUpload;