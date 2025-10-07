import os
import google.generativeai as genai
from typing import Dict, List, Optional, Any
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        """Initialize Gemini client with API key."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Interview question templates and contexts
        self.question_categories = {
            "behavioral": {
                "description": "Questions about past experiences and behavior",
                "prompts": ["Tell me about a time when...", "Describe a situation where...", "How did you handle..."]
            },
            "technical": {
                "description": "Technical skills and problem-solving questions",
                "prompts": ["Explain how you would...", "What is your approach to...", "Solve this problem..."]
            },
            "situational": {
                "description": "Hypothetical scenario questions",
                "prompts": ["What would you do if...", "How would you approach...", "In this situation..."]
            },
            "leadership": {
                "description": "Leadership and management questions",
                "prompts": ["Describe your leadership style...", "How do you motivate a team...", "Tell me about a difficult team member..."]
            }
        }
    
    async def generate_interview_question(self, 
                                        category: str = "behavioral",
                                        job_role: str = "Software Developer",
                                        difficulty: str = "medium",
                                        context: Dict = None,
                                        resume_data: Dict = None) -> Dict:
        """
        Generate a realistic interview question using Gemini.
        
        Args:
            category: Type of question (behavioral, technical, situational, leadership)
            job_role: The job role for which the interview is being conducted
            difficulty: Question difficulty (easy, medium, hard)
            context: Additional context about the interview
            
        Returns:
            Dictionary containing the generated question and metadata
        """
        try:
            prompt = self._build_question_prompt(category, job_role, difficulty, context, resume_data)
            
            response = self.model.generate_content(prompt)
            
            # Parse the response
            question_data = self._parse_question_response(response.text)
            
            return {
                "question": question_data["question"],
                "category": category,
                "difficulty": difficulty,
                "job_role": job_role,
                "expected_duration": question_data.get("expected_duration", "2-3 minutes"),
                "key_points": question_data.get("key_points", []),
                "evaluation_criteria": question_data.get("evaluation_criteria", []),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating interview question: {str(e)}")
            return {"error": str(e)}
    
    async def generate_follow_up_question(self, 
                                        original_question: str,
                                        user_response: str,
                                        analysis_results: Dict = None) -> Dict:
        """
        Generate a follow-up question based on the user's response.
        
        Args:
            original_question: The original interview question
            user_response: User's response to the original question
            analysis_results: Results from facial/voice analysis
            
        Returns:
            Dictionary containing the follow-up question
        """
        try:
            prompt = self._build_followup_prompt(original_question, user_response, analysis_results)
            
            response = self.model.generate_content(prompt)
            
            return {
                "follow_up_question": response.text.strip(),
                "reasoning": "Based on your previous response",
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating follow-up question: {str(e)}")
            return {"error": str(e)}
    
    async def generate_interview_scenario(self, 
                                        scenario_type: str = "client_meeting",
                                        job_role: str = "Software Developer",
                                        complexity: str = "medium") -> Dict:
        """
        Generate a complete interview scenario with background and questions.
        
        Args:
            scenario_type: Type of scenario (client_meeting, team_conflict, project_deadline, etc.)
            job_role: The job role context
            complexity: Scenario complexity (simple, medium, complex)
            
        Returns:
            Dictionary containing the scenario details
        """
        try:
            prompt = f"""
            Create a realistic interview scenario for a {job_role} position. 
            
            Scenario Type: {scenario_type}
            Complexity: {complexity}
            
            Please provide:
            1. A detailed background scenario (2-3 paragraphs)
            2. 3-5 progressive interview questions related to this scenario
            3. Key challenges the candidate should address
            4. Success criteria for the response
            
            Format the response as JSON with the following structure:
            {{
                "scenario": {{
                    "title": "Scenario title",
                    "background": "Detailed background description",
                    "setting": "Where this takes place",
                    "stakeholders": ["list of people involved"]
                }},
                "questions": [
                    {{
                        "question": "The interview question",
                        "focus_area": "What this question tests",
                        "difficulty": "easy/medium/hard"
                    }}
                ],
                "key_challenges": ["list of main challenges"],
                "success_criteria": ["what makes a good response"]
            }}
            """
            
            response = self.model.generate_content(prompt)
            scenario_data = json.loads(response.text.strip())
            
            scenario_data["generated_at"] = datetime.now().isoformat()
            scenario_data["job_role"] = job_role
            scenario_data["complexity"] = complexity
            
            return scenario_data
            
        except Exception as e:
            logger.error(f"Error generating interview scenario: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_response_quality(self, 
                                     question: str,
                                     user_response: str,
                                     analysis_context: Dict = None) -> Dict:
        """
        Analyze the quality of a user's response to an interview question.
        
        Args:
            question: The interview question that was asked
            user_response: The user's response
            analysis_context: Additional context from facial/voice analysis
            
        Returns:
            Dictionary containing response analysis
        """
        try:
            prompt = f"""
            Analyze this interview response and provide detailed feedback:
            
            Question: {question}
            Response: {user_response}
            
            Please provide:
            1. Overall score (1-10)
            2. Strengths of the response
            3. Areas for improvement
            4. Specific suggestions for better answers
            5. Missing key elements
            
            Additional context: {json.dumps(analysis_context) if analysis_context else "None"}
            
            Format as JSON:
            {{
                "overall_score": 0-10,
                "strengths": ["list of strengths"],
                "improvements": ["areas to improve"],
                "suggestions": ["specific suggestions"],
                "missing_elements": ["what was missing"],
                "detailed_feedback": "paragraph of detailed feedback"
            }}
            """
            
            response = self.model.generate_content(prompt)
            feedback_data = json.loads(response.text.strip())
            
            feedback_data["analyzed_at"] = datetime.now().isoformat()
            
            return feedback_data
            
        except Exception as e:
            logger.error(f"Error analyzing response: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_resume_for_questions(self, resume_text: str, job_role: str = "Software Developer") -> Dict:
        """
        Analyze resume text and extract key information for generating targeted questions.
        
        Args:
            resume_text: Full text content of the resume
            job_role: Target job role for the interview
            
        Returns:
            Dictionary containing analyzed resume data and question suggestions
        """
        try:
            prompt = f"""
            Analyze this resume and extract key information for generating personalized interview questions:
            
            RESUME TEXT:
            {resume_text}
            
            TARGET JOB ROLE: {job_role}
            
            Please analyze and return the following in JSON format:
            {{
                "skills": ["list of technical skills mentioned"],
                "experience_years": "estimated years of experience",
                "key_achievements": ["list of notable achievements"],
                "technologies": ["list of technologies/tools mentioned"],
                "domains": ["list of industry domains/sectors"],
                "projects": ["list of significant projects mentioned"],
                "education": "highest education level and field",
                "certifications": ["list of certifications if any"],
                "potential_question_areas": [
                    "list of 5-7 specific areas where questions can be generated based on this resume"
                ],
                "strength_areas": ["areas where candidate seems strongest"],
                "growth_areas": ["areas where candidate might need development"]
            }}
            
            Focus on extracting information that would be relevant for a {job_role} interview.
            """
            
            response = self.model.generate_content(prompt)
            analysis_data = json.loads(response.text.strip())
            
            analysis_data["analyzed_at"] = datetime.now().isoformat()
            analysis_data["job_role"] = job_role
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            return {"error": str(e)}
    
    def _build_question_prompt(self, category: str, job_role: str, difficulty: str, context: Dict = None, resume_data: Dict = None) -> str:
        """Build a prompt for generating interview questions."""
        base_prompt = f"""
        Generate a realistic {category} interview question for a {job_role} position.
        Difficulty level: {difficulty}
        
        Requirements:
        - The question should be professional and commonly asked in real interviews
        - Include context about what the interviewer is looking for
        - Specify expected response duration
        - List 3-5 key points a good answer should cover
        - Provide evaluation criteria
        
        """
        
        # Add resume-specific context if available
        if resume_data:
            skills = resume_data.get('skills', [])
            experience = resume_data.get('experience', [])
            projects = resume_data.get('projects', [])
            
            base_prompt += f"""
            RESUME CONTEXT (use this to personalize the question):
            - Skills: {', '.join(skills[:10]) if skills else 'None specified'}
            - Experience highlights: {'. '.join(experience[:3]) if experience else 'None specified'}
            - Projects: {'. '.join(projects[:2]) if projects else 'None specified'}
            
            Generate a question that references specific skills or experiences from the resume when relevant.
            Make the question feel personalized and directly related to their background.
            
            """
        
        if context:
            base_prompt += f"Additional context: {json.dumps(context)}\n"
        
        category_info = self.question_categories.get(category, {})
        if category_info:
            base_prompt += f"Category focus: {category_info['description']}\n"
        
        base_prompt += """
        Format the response as JSON:
        {
            "question": "The actual interview question",
            "expected_duration": "expected response time",
            "key_points": ["list of key points to cover"],
            "evaluation_criteria": ["what makes a good response"]
        }
        """
        
        return base_prompt
    
    def _build_followup_prompt(self, original_question: str, user_response: str, analysis_results: Dict = None) -> str:
        """Build a prompt for generating follow-up questions."""
        prompt = f"""
        Based on this interview exchange, generate an appropriate follow-up question:
        
        Original Question: {original_question}
        User's Response: {user_response}
        
        """
        
        if analysis_results:
            prompt += f"Analysis Results: {json.dumps(analysis_results)}\n"
        
        prompt += """
        Generate a follow-up question that:
        - Digs deeper into their response
        - Tests their knowledge further
        - Is natural and conversational
        - Helps evaluate their suitability for the role
        
        Provide only the follow-up question, nothing else.
        """
        
        return prompt
    
    def _parse_question_response(self, response_text: str) -> Dict:
        """Parse the Gemini response for question generation."""
        try:
            # Remove code block markers if present
            text = response_text.strip()
            if text.startswith('```json'):
                text = text[7:]  # Remove ```json
            if text.endswith('```'):
                text = text[:-3]  # Remove closing ```
            text = text.strip()
            
            # Try to parse as JSON
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Fallback parsing if JSON fails
            lines = response_text.strip().split('\n')
            return {
                "question": lines[0] if lines else "Could you tell me about yourself?",
                "expected_duration": "2-3 minutes",
                "key_points": [],
                "evaluation_criteria": []
            }
