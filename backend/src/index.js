const express = require('express');
const cors = require('cors');
const http = require('http');
const socketIo = require('socket.io');
const dotenv = require('dotenv');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const fetch = require('node-fetch');

// Load environment variables
dotenv.config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: process.env.FRONTEND_URL || "http://localhost:3000",
        methods: ["GET", "POST"]
    }
});

// Security middleware
app.use(helmet());
app.use(morgan('combined'));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// CORS middleware
app.use(cors({
    origin: process.env.FRONTEND_URL || "http://localhost:3000",
    credentials: true
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Routes
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        services: {
            api: 'online',
            websocket: 'online',
            ai_services: 'checking...'
        }
    });
});

// Interview session management
const activeSessions = new Map();

// Enhanced fallback question generation with resume awareness and no repetition
function generateFallbackQuestion(jobRole, category, difficulty, resumeData = null, askedQuestions = []) {
    const questionSets = {
        'Software Developer': {
            'behavioral': [
                'Tell me about a time when you had to debug a particularly challenging issue.',
                'Describe a situation where you had to learn a new technology quickly.',
                'Walk me through a time when you had to work with a difficult team member.',
                'Tell me about a project you worked on that you are particularly proud of.',
                'Describe a time when you had to make a technical decision under pressure.',
                'Tell me about a time when you had to meet a tight deadline.',
                'Describe a situation where you had to explain technical concepts to non-technical stakeholders.',
                'Walk me through how you approach problem-solving in your development work.',
                'Tell me about a time when you had to refactor legacy code.',
                'Describe your experience working in an Agile development environment.'
            ],
            'technical': [
                'How would you optimize a slow-running database query?',
                'Explain your approach to designing a scalable web application.',
                'How do you ensure code quality in your development process?',
                'Walk me through how you would implement a caching strategy.',
                'Describe your experience with microservices architecture.',
                'How do you handle API design and versioning?',
                'Explain your approach to database design and normalization.',
                'How would you implement security best practices in your applications?',
                'Describe your experience with continuous integration and deployment.',
                'How do you approach testing in your development workflow?'
            ],
            'situational': [
                'How would you handle a situation where your code causes a production outage?',
                'What would you do if you disagreed with a technical decision made by your manager?',
                'How would you approach mentoring a junior developer?',
                'What would you do if you discovered a security vulnerability in your application?',
                'How would you handle conflicting requirements from different stakeholders?',
                'What would you do if you were asked to implement a feature with an unrealistic timeline?',
                'How would you handle a situation where you need to work with unfamiliar technology?',
                'What would you do if your team was consistently missing sprint goals?'
            ]
        },
        'Data Scientist': {
            'behavioral': [
                'Tell me about a machine learning project that didn\'t go as expected.',
                'Describe a time when you had to explain complex data insights to non-technical stakeholders.',
                'Walk me through your approach to a challenging data cleaning task.',
                'Tell me about a time when your analysis changed a business decision.',
                'Describe a situation where you had to work with incomplete or poor-quality data.',
                'Tell me about a time when you had to choose between different modeling approaches.',
                'Walk me through how you validate your model\'s performance.'
            ],
            'technical': [
                'How do you handle missing data in your datasets?',
                'Explain your approach to feature selection and engineering.',
                'How do you validate and test your machine learning models?',
                'Describe your experience with A/B testing and experimental design.',
                'How do you approach model interpretability and explainability?',
                'Explain your process for dealing with overfitting.',
                'How do you handle imbalanced datasets?'
            ]
        },
        'Product Manager': {
            'behavioral': [
                'Tell me about a time when you had to make a difficult product decision.',
                'Describe how you handled a situation where engineering pushed back on requirements.',
                'Walk me through a time when you had to pivot a product strategy.',
                'Tell me about a feature launch that didn\'t meet expectations.',
                'Describe a time when you had to balance competing stakeholder priorities.',
                'Tell me about how you\'ve used data to drive product decisions.',
                'Walk me through a challenging user research project you led.'
            ],
            'technical': [
                'How do you prioritize features in your product roadmap?',
                'Explain your approach to gathering and analyzing user feedback.',
                'How do you work with data to make product decisions?',
                'Describe your experience with agile development methodologies.',
                'How do you measure product success and define KPIs?',
                'Explain your process for competitive analysis.',
                'How do you approach product pricing strategy?'
            ]
        }
    };
    
    // Get questions for the role and category, with fallbacks
    const roleQuestions = questionSets[jobRole] || questionSets['Software Developer'];
    const categoryQuestions = roleQuestions[category] || roleQuestions['behavioral'];
    
    // Filter out already asked questions
    const availableQuestions = categoryQuestions.filter(q => !askedQuestions.includes(q));
    
    // If all questions have been asked, reset the pool but add variation
    const questionsToChooseFrom = availableQuestions.length > 0 ? availableQuestions : categoryQuestions;
    
    // Select a random question
    let selectedQuestion = questionsToChooseFrom[Math.floor(Math.random() * questionsToChooseFrom.length)];
    
    // Personalize question based on resume data if available
    if (resumeData) {
        selectedQuestion = personalizeQuestionWithResume(selectedQuestion, resumeData, jobRole);
    }
    
    return {
        id: Date.now(),
        question: selectedQuestion,
        category,
        difficulty,
        jobRole,
        expectedDuration: difficulty === 'easy' ? '1-2 minutes' : difficulty === 'hard' ? '3-5 minutes' : '2-3 minutes',
        keyPoints: [],
        evaluationCriteria: ['Clarity of explanation', 'Specific examples', 'Problem-solving approach', 'Results achieved']
    };
}

// Personalize questions based on resume data
function personalizeQuestionWithResume(baseQuestion, resumeData, jobRole) {
    if (!resumeData) return baseQuestion;
    
    const skills = resumeData.skills || [];
    const experience = resumeData.experience || [];
    const projects = resumeData.projects || [];
    
    // If resume has specific skills, incorporate them
    if (skills.length > 0) {
        const randomSkill = skills[Math.floor(Math.random() * Math.min(skills.length, 3))];
        
        // Customize question based on skills
        if (baseQuestion.includes('technology') || baseQuestion.includes('technical')) {
            return baseQuestion + ` Specifically, can you share your experience with ${randomSkill}?`;
        }
        
        if (baseQuestion.includes('project')) {
            return baseQuestion + ` Please focus on a project where you used ${randomSkill}.`;
        }
    }
    
    // If resume has experience, reference it
    if (experience.length > 0) {
        const recentExperience = experience[0]; // Assume first is most recent
        if (baseQuestion.includes('experience') || baseQuestion.includes('previous role')) {
            return baseQuestion + ` Drawing from your experience at ${recentExperience.company || 'your previous company'}.`;
        }
    }
    
    // If resume has projects, reference them
    if (projects.length > 0) {
        const randomProject = projects[Math.floor(Math.random() * Math.min(projects.length, 2))];
        if (baseQuestion.includes('project') || baseQuestion.includes('work')) {
            return baseQuestion + ` You can reference ${randomProject.name || 'one of your projects'} if relevant.`;
        }
    }
    
    return baseQuestion;
}

// Fallback response evaluation when AI service is unavailable
function generateFallbackEvaluation(response) {
    const wordCount = response.split(' ').filter(word => word.trim().length > 0).length;
    const sentenceCount = response.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    
    // Simple scoring based on response length and structure
    let score = 60; // Base score
    
    // Word count scoring
    if (wordCount < 10) {
        score -= 20;
    } else if (wordCount >= 30 && wordCount <= 100) {
        score += 15;
    } else if (wordCount > 100 && wordCount <= 200) {
        score += 10;
    } else if (wordCount > 200) {
        score -= 5; // Too lengthy
    }
    
    // Structure scoring
    if (sentenceCount >= 2) {
        score += 10; // Good structure
    }
    
    // Look for key indicators
    const indicators = {
        examples: /\b(example|instance|experience|situation|time when)\b/gi,
        results: /\b(result|outcome|achieved|accomplished|successful|improved)\b/gi,
        actions: /\b(did|implemented|developed|created|led|managed|coordinated)\b/gi,
        learning: /\b(learned|realized|discovered|understood|gained)\b/gi
    };
    
    Object.values(indicators).forEach(regex => {
        if (regex.test(response)) {
            score += 5;
        }
    });
    
    // Cap the score
    score = Math.min(100, Math.max(30, score));
    
    // Generate feedback based on score
    let feedback = 'Thank you for your response. ';
    const strengths = [];
    const improvements = [];
    
    if (score >= 80) {
        feedback += 'Your answer demonstrates strong communication skills and provides good detail.';
        strengths.push('Clear and comprehensive response');
        strengths.push('Good use of examples');
    } else if (score >= 60) {
        feedback += 'Your answer shows good understanding with room for enhancement.';
        strengths.push('Clear communication');
        improvements.push('Consider adding more specific examples');
    } else {
        feedback += 'Your answer could benefit from more detail and specific examples.';
        improvements.push('Provide more detailed explanations');
        improvements.push('Include specific examples from your experience');
    }
    
    if (wordCount < 20) {
        improvements.push('Expand your answer with more details');
    } else if (wordCount > 200) {
        improvements.push('Try to be more concise while maintaining detail');
    }
    
    return {
        score,
        feedback,
        strengths: strengths.length > 0 ? strengths : ['Completed the response'],
        improvements: improvements.length > 0 ? improvements : ['Continue practicing detailed responses']
    };
}

// Socket.IO connection handling
io.on('connection', (socket) => {
    console.log('New client connected:', socket.id);
    
    // Join interview session
    socket.on('join-session', (sessionId) => {
        socket.join(sessionId);
        
        // Initialize session if not exists
        if (!activeSessions.has(sessionId)) {
            activeSessions.set(sessionId, {
                id: sessionId,
                participants: new Set(),
                startTime: Date.now(),
                currentQuestion: null,
                responses: [],
                analysisData: [],
                askedQuestions: [], // Track asked questions to avoid repetition
                questionCount: 0,
                jobRole: null,
                resumeData: null
            });
        }
        
        const session = activeSessions.get(sessionId);
        session.participants.add(socket.id);
        
        socket.emit('session-joined', {
            sessionId,
            participantCount: session.participants.size
        });
        
        console.log(`Client ${socket.id} joined session ${sessionId}`);
    });
    
    // Handle interview question requests
    socket.on('request-question', async (data) => {
        try {
            const { sessionId, category = 'behavioral', difficulty = 'medium', jobRole = 'Software Developer', resumeData = null } = data;
            
            // Get or update session with job role and resume data
            let session = activeSessions.get(sessionId);
            if (session) {
                session.jobRole = jobRole;
                session.resumeData = resumeData;
                session.questionCount++;
            }
            
            // Call AI services for question generation
            let questionData;
            try {
                const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8001';
                
                // Enhanced prompt for personalized questions
                const enhancedResumeData = resumeData ? {
                    ...resumeData,
                    asked_questions: session?.askedQuestions || [],
                    question_count: session?.questionCount || 0
                } : null;
                
                const response = await fetch(`${aiServiceUrl}/generate/question`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        job_role: jobRole,
                        resume_data: enhancedResumeData,
                        difficulty: difficulty,
                        session_id: sessionId,
                        category: category,
                        context: {
                            asked_questions: session?.askedQuestions || [],
                            interview_stage: session?.questionCount || 1,
                            previous_responses: session?.responses?.slice(-2) || [] // Last 2 responses for context
                        }
                    })
                });
                
                if (response.ok) {
                    const aiQuestion = await response.json();
                    questionData = {
                        id: Date.now(),
                        question: aiQuestion.question,
                        category: aiQuestion.category || category,
                        difficulty: aiQuestion.difficulty || difficulty,
                        jobRole,
                        expectedDuration: aiQuestion.expected_duration || '2-3 minutes',
                        keyPoints: [],
                        evaluationCriteria: aiQuestion.evaluation_criteria || ['Clarity', 'Relevance', 'Depth']
                    };
                } else {
                    throw new Error(`AI service responded with status ${response.status}`);
                }
            } catch (aiError) {
                console.warn('AI service unavailable, using fallback question generation:', aiError.message);
                // Fallback to improved question generation with tracking
                const askedQuestions = session?.askedQuestions || [];
                questionData = generateFallbackQuestion(jobRole, category, difficulty, resumeData, askedQuestions);
            }
            
            // Update session with new question
            if (session) {
                session.currentQuestion = questionData;
                session.askedQuestions.push(questionData.question);
            }
            
            io.to(sessionId).emit('question-generated', questionData);
            console.log(`Generated ${session?.askedQuestions?.length > 0 ? 'personalized' : 'standard'} question for session ${sessionId}:`, questionData.question.substring(0, 100) + '...');
            
        } catch (error) {
            console.error('Error generating question:', error);
            socket.emit('error', { message: 'Failed to generate question' });
        }
    });
    
    // Handle user responses
    socket.on('submit-response', async (data) => {
        try {
            const { sessionId, questionId, response, duration, question } = data;
            
            if (activeSessions.has(sessionId)) {
                const session = activeSessions.get(sessionId);
                
                let evaluation = null;
                
                // Call AI service for response evaluation
                try {
                    const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8001';
                    const evalResponse = await fetch(`${aiServiceUrl}/evaluate/answer?answer_text=${encodeURIComponent(response)}&question=${encodeURIComponent(question || session.currentQuestion?.question || '')}&session_id=${sessionId}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        }
                    });
                    
                    if (evalResponse.ok) {
                        evaluation = await evalResponse.json();
                    } else {
                        console.warn('AI evaluation service unavailable, using fallback');
                        evaluation = generateFallbackEvaluation(response);
                    }
                } catch (aiError) {
                    console.warn('AI evaluation failed, using fallback:', aiError.message);
                    evaluation = generateFallbackEvaluation(response);
                }
                
                const responseData = {
                    id: Date.now(),
                    questionId,
                    response,
                    duration,
                    timestamp: Date.now(),
                    analysis: evaluation
                };
                
                session.responses.push(responseData);
                
                // Emit response received confirmation with evaluation
                socket.emit('response-received', { 
                    responseId: responseData.id,
                    evaluation: evaluation
                });
                
                console.log(`Response evaluated for session ${sessionId}:`, response.substring(0, 50) + '...');
            }
            
        } catch (error) {
            console.error('Error handling response:', error);
            socket.emit('error', { message: 'Failed to process response' });
        }
    });
    
    // Handle real-time analysis data (facial/voice)
    socket.on('analysis-data', (data) => {
        try {
            const { sessionId, type, analysisResult } = data;
            
            if (activeSessions.has(sessionId)) {
                const session = activeSessions.get(sessionId);
                
                const analysisEntry = {
                    timestamp: Date.now(),
                    type, // 'facial' or 'voice'
                    data: analysisResult
                };
                
                session.analysisData.push(analysisEntry);
                
                // Broadcast real-time analysis to session participants
                io.to(sessionId).emit('live-analysis', analysisEntry);
            }
            
        } catch (error) {
            console.error('Error handling analysis data:', error);
        }
    });
    
    // Handle follow-up question requests
    socket.on('request-followup', async (data) => {
        try {
            const { sessionId, originalQuestion, userResponse } = data;
            
            // Mock follow-up question generation
            const followUpQuestion = {
                id: Date.now(),
                question: `That's interesting. Can you elaborate on the specific steps you took and what the outcome was?`,
                reasoning: 'To gather more specific details about the situation',
                timestamp: Date.now()
            };
            
            io.to(sessionId).emit('followup-generated', followUpQuestion);
            console.log(`Generated follow-up for session ${sessionId}`);
            
        } catch (error) {
            console.error('Error generating follow-up:', error);
            socket.emit('error', { message: 'Failed to generate follow-up question' });
        }
    });
    
    // Handle session end
    socket.on('end-session', (sessionId) => {
        if (activeSessions.has(sessionId)) {
            const session = activeSessions.get(sessionId);
            
            // Generate session summary
            const summary = {
                sessionId,
                duration: Date.now() - session.startTime,
                questionsAsked: session.responses.length,
                totalResponses: session.responses.length,
                analysisPoints: session.analysisData.length,
                endTime: Date.now()
            };
            
            io.to(sessionId).emit('session-summary', summary);
            
            // Clean up session after a delay
            setTimeout(() => {
                activeSessions.delete(sessionId);
                console.log(`Session ${sessionId} cleaned up`);
            }, 60000); // 1 minute delay
        }
    });
    
    // Handle disconnection
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
        
        // Remove from active sessions
        activeSessions.forEach((session, sessionId) => {
            if (session.participants.has(socket.id)) {
                session.participants.delete(socket.id);
                
                // If no participants left, mark session for cleanup
                if (session.participants.size === 0) {
                    setTimeout(() => {
                        if (activeSessions.has(sessionId) && activeSessions.get(sessionId).participants.size === 0) {
                            activeSessions.delete(sessionId);
                            console.log(`Empty session ${sessionId} cleaned up`);
                        }
                    }, 30000); // 30 second delay
                }
            }
        });
    });
});

// API Routes
app.get('/api/sessions/active', (req, res) => {
    const sessionList = Array.from(activeSessions.values()).map(session => ({
        id: session.id,
        participants: session.participants.size,
        startTime: session.startTime,
        questionsAsked: session.responses.length
    }));
    
    res.json(sessionList);
});

app.get('/api/sessions/:sessionId', (req, res) => {
    const { sessionId } = req.params;
    
    if (activeSessions.has(sessionId)) {
        const session = activeSessions.get(sessionId);
        res.json({
            id: session.id,
            participants: session.participants.size,
            startTime: session.startTime,
            currentQuestion: session.currentQuestion,
            totalResponses: session.responses.length,
            totalAnalysisPoints: session.analysisData.length
        });
    } else {
        res.status(404).json({ error: 'Session not found' });
    }
});

// REST API endpoint for question generation
app.post('/api/generate-question', async (req, res) => {
    try {
        const { sessionId, jobRole = 'Software Developer', category = 'behavioral', difficulty = 'medium', resumeData = null } = req.body;
        
        // Get or create session
        let session = activeSessions.get(sessionId);
        if (!session) {
            session = {
                id: sessionId,
                participants: new Set(),
                startTime: Date.now(),
                currentQuestion: null,
                responses: [],
                analysisData: [],
                askedQuestions: [],
                questionCount: 0,
                jobRole: jobRole,
                resumeData: resumeData
            };
            activeSessions.set(sessionId, session);
        }
        
        // Update session info
        session.jobRole = jobRole;
        session.resumeData = resumeData;
        session.questionCount++;
        
        let questionData;
        try {
            const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8001';
            const response = await fetch(`${aiServiceUrl}/generate/question`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    job_role: jobRole,
                    resume_data: resumeData,
                    difficulty: difficulty,
                    session_id: sessionId
                })
            });
            
            if (response.ok) {
                const aiQuestion = await response.json();
                questionData = {
                    id: Date.now(),
                    question: aiQuestion.question,
                    category: aiQuestion.category || category,
                    difficulty: aiQuestion.difficulty || difficulty,
                    jobRole,
                    expectedDuration: aiQuestion.expected_duration || '2-3 minutes',
                    keyPoints: [],
                    evaluationCriteria: aiQuestion.evaluation_criteria || ['Clarity', 'Relevance', 'Depth']
                };
            } else {
                throw new Error(`AI service responded with status ${response.status}`);
            }
        } catch (aiError) {
            console.warn('AI service unavailable, using fallback question generation:', aiError.message);
            const askedQuestions = session?.askedQuestions || [];
            questionData = generateFallbackQuestion(jobRole, category, difficulty, resumeData, askedQuestions);
        }
        
        // Update session with new question
        session.currentQuestion = questionData;
        session.askedQuestions.push(questionData.question);
        
        res.json(questionData);
        console.log(`Generated question via API for session ${sessionId}:`, questionData.question);
        
    } catch (error) {
        console.error('Error generating question:', error);
        res.status(500).json({ error: 'Failed to generate question' });
    }
});

// REST API endpoint for response evaluation
app.post('/api/evaluate-response', async (req, res) => {
    try {
        const { sessionId, response, question, duration } = req.body;
        
        let evaluation = null;
        
        // Call AI service for response evaluation
        try {
            const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8001';
            const evalResponse = await fetch(`${aiServiceUrl}/evaluate/answer?answer_text=${encodeURIComponent(response)}&question=${encodeURIComponent(question || '')}&session_id=${sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (evalResponse.ok) {
                evaluation = await evalResponse.json();
            } else {
                console.warn('AI evaluation service unavailable, using fallback');
                evaluation = generateFallbackEvaluation(response);
            }
        } catch (aiError) {
            console.warn('AI evaluation failed, using fallback:', aiError.message);
            evaluation = generateFallbackEvaluation(response);
        }
        
        // Store response data if session exists
        if (activeSessions.has(sessionId)) {
            const session = activeSessions.get(sessionId);
            const responseData = {
                id: Date.now(),
                questionId: Date.now(),
                response,
                duration,
                timestamp: Date.now(),
                analysis: evaluation
            };
            session.responses.push(responseData);
        }
        
        res.json(evaluation);
        console.log(`Response evaluated via API for session ${sessionId}:`, response.substring(0, 50) + '...');
        
    } catch (error) {
        console.error('Error evaluating response:', error);
        res.status(500).json({ error: 'Failed to evaluate response' });
    }
});

// REST API endpoint for interview report generation
app.post('/api/generate-report', async (req, res) => {
    try {
        const { sessionId } = req.body;
        
        if (!activeSessions.has(sessionId)) {
            return res.status(404).json({ error: 'Session not found' });
        }
        
        const session = activeSessions.get(sessionId);
        
        // Prepare report data
        const reportData = {
            session_id: sessionId,
            job_role: session.jobRole || 'Software Developer',
            total_questions: 5, // MAX_QUESTIONS from frontend
            questions_answered: session.responses.length,
            questions_skipped: Math.max(0, 5 - session.responses.length),
            average_response_time: session.responses.length > 0 ? 
                session.responses.reduce((acc, r) => acc + (r.duration || 0), 0) / session.responses.length : 0,
            responses: session.responses.map(r => ({
                question: r.question || session.currentQuestion?.question || 'Question not recorded',
                response: r.response || '',
                duration: r.duration || 0,
                analysis: r.analysis || { score: 70, feedback: 'No analysis available' }
            })),
            facial_analysis_summary: calculateFacialAnalysisSummary(session.analysisData),
            voice_analysis_summary: calculateVoiceAnalysisSummary(session.analysisData)
        };
        
        // Call AI service for report generation
        try {
            const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8001';
            const reportResponse = await fetch(`${aiServiceUrl}/generate/report`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(reportData)
            });
            
            if (reportResponse.ok) {
                const report = await reportResponse.json();
                
                // Store report in session for potential retrieval
                session.finalReport = report;
                
                res.json(report);
                console.log(`Interview report generated for session ${sessionId}`);
            } else {
                throw new Error(`AI service responded with status ${reportResponse.status}`);
            }
        } catch (aiError) {
            console.warn('AI report generation failed, using fallback:', aiError.message);
            
            // Fallback report generation
            const fallbackReport = generateFallbackReport(reportData);
            session.finalReport = fallbackReport;
            res.json(fallbackReport);
        }
        
    } catch (error) {
        console.error('Error generating report:', error);
        res.status(500).json({ error: 'Failed to generate interview report' });
    }
});

// Helper function to calculate facial analysis summary
function calculateFacialAnalysisSummary(analysisData) {
    const facialData = analysisData.filter(d => d.type === 'facial');
    if (facialData.length === 0) return null;
    
    const avgConfidence = facialData.reduce((acc, d) => acc + (d.data.confidence || 0.75), 0) / facialData.length;
    const avgEngagement = facialData.reduce((acc, d) => acc + (d.data.engagement || 0.80), 0) / facialData.length;
    const avgEyeContact = facialData.reduce((acc, d) => acc + (d.data.eye_contact || 0.70), 0) / facialData.length;
    
    return {
        average_confidence: avgConfidence,
        average_engagement: avgEngagement,
        average_eye_contact: avgEyeContact,
        total_frames_analyzed: facialData.length
    };
}

// Helper function to calculate voice analysis summary
function calculateVoiceAnalysisSummary(analysisData) {
    const voiceData = analysisData.filter(d => d.type === 'voice');
    if (voiceData.length === 0) return null;
    
    const avgClarity = voiceData.reduce((acc, d) => acc + (d.data.clarity || 0.85), 0) / voiceData.length;
    const avgConfidence = voiceData.reduce((acc, d) => acc + (d.data.confidence_score || 0.75), 0) / voiceData.length;
    
    // Get most common pace
    const paceCount = { slow: 0, moderate: 0, fast: 0 };
    voiceData.forEach(d => {
        const pace = d.data.pace || 'moderate';
        if (paceCount[pace] !== undefined) paceCount[pace]++;
    });
    const dominantPace = Object.keys(paceCount).reduce((a, b) => paceCount[a] > paceCount[b] ? a : b);
    
    return {
        average_clarity: avgClarity,
        average_confidence: avgConfidence,
        dominant_pace: dominantPace,
        total_audio_analyzed: voiceData.length
    };
}

// Fallback report generation
function generateFallbackReport(reportData) {
    const avgScore = reportData.responses.length > 0 ? 
        reportData.responses.reduce((acc, r) => acc + (r.analysis.score || 70), 0) / reportData.responses.length : 70;
    
    const completionRate = reportData.questions_answered / reportData.total_questions;
    const overallScore = Math.round(avgScore * 0.7 + completionRate * 100 * 0.3);
    
    let placementLikelihood = 'Medium';
    if (overallScore >= 80) placementLikelihood = 'High';
    else if (overallScore < 65) placementLikelihood = 'Low';
    
    return {
        session_id: reportData.session_id,
        overall_score: overallScore,
        placement_likelihood: placementLikelihood,
        performance_summary: `Interview completed with ${reportData.questions_answered}/${reportData.total_questions} questions answered. Overall performance shows ${placementLikelihood.toLowerCase()} potential for placement.`,
        strengths: ['Professional communication', 'Completed interview process', 'Demonstrated relevant experience'],
        development_areas: ['Practice more interview questions', 'Provide more detailed examples', 'Build confidence'],
        detailed_feedback: {
            communication: { score: Math.max(60, avgScore - 5), feedback: 'Focus on clear, structured responses' },
            technical_knowledge: { score: Math.max(55, avgScore - 10), feedback: 'Continue developing role-specific skills' },
            problem_solving: { score: Math.max(60, avgScore - 8), feedback: 'Practice systematic problem-solving approaches' },
            confidence: { score: Math.max(50, avgScore - 15), feedback: 'Work on building interview confidence' }
        },
        skill_breakdown: {
            verbal_communication: Math.max(60, avgScore - 5),
            confidence_level: Math.max(50, avgScore - 15),
            technical_competency: Math.max(55, avgScore - 10),
            problem_solving: Math.max(60, avgScore - 8),
            professionalism: Math.max(70, avgScore)
        },
        recommendations: [
            'Practice common interview questions for your role',
            'Prepare specific examples using the STAR method',
            'Research the company and position thoroughly',
            'Work on confident body language and eye contact',
            'Practice explaining technical concepts clearly'
        ],
        generated_at: new Date().toISOString()
    };
}

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

const PORT = process.env.PORT || 5000;

server.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
    console.log(`📊 Health check: http://localhost:${PORT}/health`);
    console.log(`🔌 WebSocket server ready`);
    console.log(`🎯 Active sessions tracking enabled`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('Received SIGTERM signal, shutting down gracefully...');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});