"""
Learning Service for IQRAi
Handles quiz generation, Socratic dialogue, and study mode
"""
import os
import json
import uuid
from typing import List, Dict, Optional
from dotenv import load_dotenv
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
from app.utils.document_processor import DocumentProcessor

load_dotenv()


class LearningService:
    """Service for learning features (quiz generation, Socratic mode, etc.)"""
    
    def __init__(self):
        """Initialize learning service with agents."""
        self.quiz_agent = self._initialize_quiz_agent()
        self.socratic_agent = self._initialize_socratic_agent()
        self.study_agent = self._initialize_study_agent()
    
    def _initialize_quiz_agent(self) -> Agent:
        """Initialize agent for quiz generation."""
        instructions = """You are an expert quiz generator for educational content.

Your role is to create high-quality, relevant quizzes based on provided content.

QUIZ GENERATION RULES:
1. Read the provided content carefully
2. Generate questions that test understanding, not just memorization
3. Ensure questions are clear and unambiguous
4. For MCQs: Provide 4 options with only 1 correct answer
5. For True/False: Make statements that are definitively true or false
6. For Short Answer: Ask questions that require brief, specific responses

OUTPUT FORMAT:
Always return valid JSON in this exact format:

For MCQ:
{
  "quiz_type": "mcq",
  "questions": [
    {
      "question": "What is the main concept?",
      "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
      "correct_answer": "B",
      "explanation": "Brief explanation of why B is correct"
    }
  ]
}

For True/False:
{
  "quiz_type": "true_false",
  "questions": [
    {
      "question": "Statement to verify",
      "correct_answer": "True",
      "explanation": "Why this is true/false"
    }
  ]
}

For Short Answer:
{
  "quiz_type": "short_answer",
  "questions": [
    {
      "question": "Question requiring brief response",
      "expected_answer": "Key points that should be in the answer",
      "explanation": "Full answer explanation"
    }
  ]
}

IMPORTANT: Return ONLY valid JSON, no additional text."""

        return Agent(
            name="Quiz Generator",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    def _initialize_socratic_agent(self) -> Agent:
        """Initialize agent for Socratic dialogue."""
        instructions = """You are a Socratic teaching assistant for IQRAi University.

Your role is to help students learn through questioning and guided discovery, NOT by directly giving answers.

SOCRATIC METHOD PRINCIPLES:
1. Never directly state the answer - guide students to discover it
2. Ask probing questions that make students think
3. Use examples and analogies to illustrate concepts
4. Break complex topics into smaller, manageable questions
5. Acknowledge correct thinking and gently redirect misconceptions
6. Be encouraging and patient

DIALOGUE FLOW:
- Start by asking what the student already knows
- Build on their existing knowledge
- Use counter-examples to challenge assumptions
- Lead them to conclusions through logical steps

EXAMPLES:
Student: "What is recursion?"
You: "Great question! Before we dive in, have you ever seen a mirror reflecting another mirror? What do you notice about the reflections?"

Student: "They repeat infinitely"
You: "Exactly! Now, if a function calls itself, what do you think might happen?"

Student: "It would repeat like the mirrors?"
You: "You're on the right track! That's the essence of recursion - a function calling itself. But what do you think we need to prevent it from repeating forever?"

BE CONVERSATIONAL, ENCOURAGING, AND THOUGHT-PROVOKING."""

        return Agent(
            name="Socratic Tutor",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    def _initialize_study_agent(self) -> Agent:
        """Initialize agent for study mode (AI asks questions)."""
        instructions = """You are a study quiz master for IQRAi University.

Your role is to quiz students on content they've uploaded to help them study.

STUDY MODE RULES:
1. Ask questions one at a time based on the provided content
2. Questions should test different aspects: understanding, application, analysis
3. After student answers, provide feedback:
   - If correct: Praise and optionally expand
   - If incorrect: Gently correct and explain the right answer
4. Track progress and vary difficulty
5. Be encouraging and educational

QUESTION FORMAT:
Ask clear, direct questions that test understanding of the material.

FEEDBACK FORMAT:
ALWAYS start your feedback with a rating out of 10 in this EXACT format:
"[Rating: X/10]"

Example:
"[Rating: 7/10] Good effort! You correctly identified..."

Rating Guidelines:
- 9-10: Completely correct, demonstrates deep understanding
- 7-8: Mostly correct with minor omissions
- 5-6: Partially correct, missing key elements
- 3-4: Incorrect but shows some understanding
- 1-2: Incorrect with minimal understanding

After the rating, provide detailed feedback."""

        return Agent(
            name="Study Quiz Master",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    async def generate_quiz(
        self,
        content: str,
        quiz_type: str,
        num_questions: int,
        total_marks: int = 0,
        content_type: str = "quiz"
    ) -> Dict:
        """
        Generate a quiz based on content with marks-based complexity.
        
        Args:
            content: The educational content to generate quiz from
            quiz_type: Type of quiz (mcq, true_false, short_answer)
            num_questions: Number of questions to generate
            total_marks: Total marks for the quiz/assignment
            content_type: "quiz" or "assignment"
            
        Returns:
            Dict with quiz data
        """
        # Calculate marks per question
        marks_per_question = total_marks / num_questions if total_marks > 0 else 0
        
        # Determine complexity level and question style based on marks
        if marks_per_question <= 1:
            complexity = "basic recall"
            question_depth = "simple factual or definitional questions"
            question_style = "recall basic facts, definitions, or concepts"
        elif marks_per_question <= 2:
            complexity = "simple understanding"
            question_depth = "short factual questions requiring basic reasoning"
            question_style = "list examples, identify key terms, or simple classification"
        elif marks_per_question <= 5:
            complexity = "short explanation or application"
            question_depth = "require explanation with examples or basic application"
            question_style = "explain concepts with examples or apply basic principles"
        elif marks_per_question <= 10:
            complexity = "detailed explanation or comparison"
            question_depth = "require structured analysis with explanations and comparisons"
            question_style = "explain multiple aspects, compare concepts, or analyze relationships"
        elif marks_per_question <= 15:
            complexity = "comprehensive analytical"
            question_depth = "require critical thinking and multi-perspective evaluation"
            question_style = "evaluate from multiple perspectives, analyze causes and effects, or assess implications"
        else:  # > 15 marks
            complexity = "in-depth essay or case study"
            question_depth = "require argumentative reasoning, evaluation, and synthesis"
            question_style = "develop arguments, synthesize knowledge, evaluate trade-offs, or analyze case studies"
        
        # Adjust based on content type
        if content_type == "assignment":
            if marks_per_question <= 2:
                question_depth = "brief but require understanding"
            elif marks_per_question <= 5:
                question_depth = "require detailed explanation and critical thinking"
            else:
                question_depth = "require comprehensive analysis, examples, comparisons, and critical evaluation"
        
        prompt = f"""Generate a {content_type} with {num_questions} questions based on this content:

CONTENT:
{content}

Number of Questions: {num_questions}
Total Marks: {total_marks}
Marks per Question: {marks_per_question:.1f}

QUESTION COMPLEXITY REQUIREMENTS (CRITICAL):
- Marks per question: {marks_per_question:.1f} marks
- Complexity level: {complexity}
- Question depth: {question_depth}
- Question style: {question_style}

For {marks_per_question:.1f} marks per question, each question must:
1. Test the appropriate cognitive level based on mark weight
2. Require {question_style}
3. Have depth matching: {question_depth}
4. Reflect complexity: {complexity}

QUESTION STYLING BY MARKS:
- 1 mark: Factual recall only (e.g., "What is X?", "Define Y")
- 2 marks: Basic understanding (e.g., "List two examples", "Identify key terms")
- 3-5 marks: Short explanation (e.g., "Explain X with an example", "Describe how Y works")
- 6-10 marks: Detailed analysis (e.g., "Compare X and Y", "Discuss implications of Z")
- 11-15 marks: Comprehensive evaluation (e.g., "Evaluate X from multiple perspectives", "Assess the impact of Y")
- 16+ marks: Essay-style synthesis (e.g., "Critically analyze X", "Evaluate trade-offs between Y and Z")

Generate questions that test different cognitive levels to ensure comprehensive assessment.

OUTPUT FORMAT:
Always return valid JSON in this exact format:

For MCQ:
{{
  "quiz_type": "mcq",
  "questions": [
    {{
      "question": "Question text",
      "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
      "correct_answer": "B",
      "marks": {marks_per_question:.1f},
      "explanation": "Brief explanation of why B is correct"
    }}
  ]
}}

For True/False:
{{
  "quiz_type": "true_false",
  "questions": [
    {{
      "question": "Statement to verify",
      "correct_answer": "True",
      "marks": {marks_per_question:.1f},
      "explanation": "Why this is true/false"
    }}
  ]
}}

For Short Answer:
{{
  "quiz_type": "short_answer",
  "questions": [
    {{
      "question": "Question requiring response matching complexity level",
      "expected_answer": "Key points that should be in the answer",
      "marks": {marks_per_question:.1f},
      "explanation": "Full answer explanation"
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON, no additional text. Each question must include a "marks" field with value {marks_per_question:.1f}."""

        try:
            result = Runner.run_streamed(
                self.quiz_agent,
                input=[{"role": "user", "content": prompt}],
            )
            
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            # Parse JSON from response
            json_start = full_response.find('{')
            json_end = full_response.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = full_response[json_start:json_end]
                quiz_data = json.loads(json_str)
                
                # Ensure all questions have marks field
                if 'questions' in quiz_data:
                    for q in quiz_data['questions']:
                        if 'marks' not in q:
                            q['marks'] = marks_per_question
                
                return quiz_data
            else:
                raise ValueError("No valid JSON in response")
                
        except Exception as e:
            raise ValueError(f"Quiz generation failed: {str(e)}")
    
    async def socratic_dialogue(
        self,
        topic: str,
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Generate Socratic dialogue response.
        
        Args:
            topic: The topic or question student is asking about
            conversation_history: Previous messages in conversation
            
        Returns:
            Socratic response string
        """
        if conversation_history is None:
            conversation_history = []
        
        # Add current topic to conversation
        messages = conversation_history + [{"role": "user", "content": topic}]
        
        try:
            result = Runner.run_streamed(
                self.socratic_agent,
                input=messages,
            )
            
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            return full_response.strip()
                
        except Exception as e:
            raise ValueError(f"Socratic dialogue failed: {str(e)}")
    
    async def study_mode_question(
        self,
        content: str,
        previous_qa: List[Dict] = None
    ) -> str:
        """
        Generate a study mode question based on content.
        
        Args:
            content: The study content
            previous_qa: Previous questions and answers in this session
            
        Returns:
            Question string
        """
        if previous_qa is None:
            previous_qa = []
        
        context = f"""Study Content:
{content}

Previous Q&A in this session:
{json.dumps(previous_qa) if previous_qa else "None yet"}

Generate the next question to quiz the student on this content. Ask something they haven't been asked yet."""

        try:
            result = Runner.run_streamed(
                self.study_agent,
                input=[{"role": "user", "content": context}],
            )
            
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            return full_response.strip()
                
        except Exception as e:
            raise ValueError(f"Study mode question generation failed: {str(e)}")
    
    async def evaluate_study_answer(
        self,
        question: str,
        student_answer: str,
        content: str
    ) -> str:
        """
        Evaluate student's answer in study mode.
        
        Args:
            question: The question asked
            student_answer: Student's response
            content: Original study content
            
        Returns:
            Feedback string
        """
        prompt = f"""Study Content:
{content}

Question Asked: {question}
Student's Answer: {student_answer}

Evaluate the student's answer. Provide encouraging feedback indicating if they're correct, incorrect, or partially correct. Explain the right answer briefly."""

        try:
            result = Runner.run_streamed(
                self.study_agent,
                input=[{"role": "user", "content": prompt}],
            )
            
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            return full_response.strip()
                
        except Exception as e:
            raise ValueError(f"Answer evaluation failed: {str(e)}")

