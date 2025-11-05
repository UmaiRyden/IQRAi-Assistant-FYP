"""
OBE (Outcome-Based Education) Verification Service
Checks exam papers against Bloom's Taxonomy verb requirements using AI Agent
"""
import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
from openai import BadRequestError
from app.utils.document_processor import DocumentProcessor

load_dotenv()


class OBEService:
    """Service for verifying exam papers against Bloom's Taxonomy verbs using AI Agent."""
    
    def __init__(self):
        """Initialize OBE service, load verb database, and create agent."""
        self.verbs_data = self._load_verbs()
        self.agent = self._initialize_agent()
        self.rewriter_agent = self._initialize_rewriter_agent()
    
    def _load_verbs(self) -> Dict:
        """Load OBE verbs from JSON file."""
        verbs_path = os.path.join("knowledge", "obe_verbs.json")
        try:
            with open(verbs_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"OBE verbs file not found at: {verbs_path}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in OBE verbs file")
    
    def _initialize_agent(self) -> Agent:
        """Initialize OpenAI Agent for OBE verification."""
        
        # Create comprehensive instructions for the agent
        instructions = """You are an expert in Outcome-Based Education (OBE) and Bloom's Taxonomy assessment.
Your role is to analyze exam question papers and verify if they contain appropriate action verbs according to the selected Bloom's Taxonomy domain and level.

CRITICAL REQUIREMENT: You must ONLY match verbs that are in the provided required verb list for the target level. Do NOT match verbs from lower levels.

TASK:
You will be provided with:
1. A domain (Cognitive, Psychomotor, or Affective)
2. A level (e.g., C1-C6 for Cognitive, P1-P4 for Psychomotor, A1-A4 for Affective)
3. A list of required verbs for that domain and level
4. The extracted text from an exam question paper

INSTRUCTIONS:
1. Carefully read and understand the exam questions
2. Identify all action verbs used in the questions (even if they are in different forms like "explain", "explaining", "explained")
3. **CRITICAL:** Match these verbs ONLY against the required verb list provided for the target level
4. **Do NOT match lower-level verbs:** For example, if checking C2 level, do NOT match C1 verbs like "Define", "List", "State", "Name" - these should be EXCLUDED
5. Consider verb variations and tenses (e.g., "analyze" matches "analyzing", "analyzed", "analysis")
6. Look for verbs in context - questions like "Students should be able to explain..." contain the verb "explain"
7. Be strict - if the paper uses predominantly lower-level verbs, the match should be LOW and status should be FAIL

OUTPUT FORMAT (MUST BE VALID JSON):
{
  "matched_verbs": ["Only verbs from the required list"],
  "excluded_verbs": ["Lower-level verbs found but excluded"],
  "verb_details": [
    {
      "verb": "Explain",
      "found_in_question": "Explain the concept of...",
      "variation_used": "explaining"
    }
  ],
  "analysis": "Brief analysis of the question paper's alignment with the selected level. Mention if lower-level verbs were found.",
  "suggestions": ["Suggestions for improvement if applicable"],
  "confidence_score": 85
}

IMPORTANT RULES:
- **STRICT MATCHING:** Only match verbs from the provided required verb list
- **Exclude lower-level verbs:** If you find C1 verbs (Define, List, State, Name) when checking C2, do NOT include them in matched_verbs
- **Low confidence for mismatched levels:** If paper uses wrong-level verbs, confidence should be LOW (0-30)
- Consider the educational context - questions that ask students to perform an action contain that verb
- Provide helpful feedback about which level the paper actually aligns with
- Your response MUST be valid JSON format only, no additional text

BLOOM'S TAXONOMY LEVELS (for context):
Cognitive Domain:
- C1 (Remember): Recall facts and basic concepts
- C2 (Understand): Explain ideas or concepts
- C3 (Apply): Use information in new situations
- C4 (Analyze): Draw connections among ideas
- C5 (Evaluate): Justify a stand or decision
- C6 (Create): Produce new or original work

Psychomotor Domain:
- P1 (Perception): Use sensory cues to guide activity
- P2 (Set): Demonstrate readiness to act
- P3 (Guided Response): Know steps and perform with guidance
- P4 (Mechanism): Perform skillfully and with confidence

Affective Domain:
- A1 (Receiving): Show awareness and willingness to hear
- A2 (Responding): Active participation and reaction
- A3 (Valuing): Attach value and express commitment
- A4 (Organization): Integrate values into a system

Be professional, accurate, and helpful in your analysis."""

        return Agent(
            name="OBE Verification Specialist",
            instructions=instructions,
            model="gpt-4o-mini"  # Using GPT-4 for better reasoning
        )
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains."""
        return list(self.verbs_data.keys())
    
    def get_levels_for_domain(self, domain: str) -> List[str]:
        """Get available levels for a specific domain."""
        if domain not in self.verbs_data:
            return []
        return list(self.verbs_data[domain].keys())
    
    def get_verbs_for_level(self, domain: str, level: str) -> List[str]:
        """Get verbs for a specific domain and level."""
        if domain not in self.verbs_data:
            return []
        if level not in self.verbs_data[domain]:
            return []
        return self.verbs_data[domain][level]
    
    
    def _split_into_questions(self, text: str) -> List[Dict[str, str]]:
        """
        Split document text into individual questions.
        
        Returns:
            List of dicts with 'number' and 'text' keys
        """
        import re
        
        questions = []
        
        # Pattern to match question numbers: "1.", "Q1.", "Question 1:", etc.
        # Also handles formats like "1)", "(1)", "Q.1", etc.
        patterns = [
            r'(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.|$)',  # 1. Question text
            r'(?:^|\n)\s*Q\.?\s*(\d+)[\.:)]\s+(.+?)(?=\n\s*Q\.?\s*\d+|$)',  # Q1. or Q.1 or Q1:
            r'(?:^|\n)\s*Question\s+(\d+)[\.:]\s+(.+?)(?=\n\s*Question\s+\d+|$)',  # Question 1:
            r'(?:^|\n)\s*\((\d+)\)\s+(.+?)(?=\n\s*\(\d+\)|$)',  # (1) Question
            r'(?:^|\n)\s*(\d+)\)\s+(.+?)(?=\n\s*\d+\)|$)',  # 1) Question
        ]
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if matches:
                for num, question_text in matches:
                    cleaned_text = question_text.strip()
                    if cleaned_text and len(cleaned_text) > 5:  # Ignore very short matches
                        questions.append({
                            'number': num,
                            'text': cleaned_text
                        })
                if questions:  # If we found questions with this pattern, stop trying others
                    break
        
        # If no numbered questions found, try splitting by question marks
        if not questions:
            parts = text.split('?')
            for i, part in enumerate(parts[:-1], 1):  # Exclude last part after final ?
                # Take last sentence before the question mark
                sentences = part.split('.')
                question_text = sentences[-1].strip() + '?'
                if len(question_text) > 10:
                    questions.append({
                        'number': str(i),
                        'text': question_text
                    })
        
        # If still no questions, treat entire document as one question
        if not questions:
            questions.append({
                'number': '1',
                'text': text.strip()
            })
        
        return questions
    
    def _detect_level_for_question(self, question_text: str, domain: str) -> Dict:
        """
        Detect which level(s) a question belongs to based on verbs.
        Only matches verbs that are clearly used as action verbs (typically at start of question).
        
        Returns:
            Dict with detected levels and matched verbs
        """
        question_lower = question_text.lower().strip()
        detected_levels = {}
        
        # Common question starters to strip before checking verbs
        question_starters = [
            'write a', 'write an', 'give a', 'give an', 'provide a', 'provide an',
            'create a', 'create an', 'develop a', 'develop an', 'make a', 'make an',
            'what is', 'what are', 'how does', 'how do', 'why is', 'why are',
            'when is', 'when are', 'where is', 'where are', 'who is', 'who are',
            'which is', 'which are', 'can you', 'could you', 'should you',
            'students should', 'you should', 'you must', 'students must',
            'you are required to', 'students are required to'
        ]
        
        # Remove common question starters to focus on the action verb
        cleaned_question = question_lower
        for starter in question_starters:
            if cleaned_question.startswith(starter):
                cleaned_question = cleaned_question[len(starter):].strip()
                break
        
        # Check each level in the domain
        for level, verbs in self.verbs_data[domain].items():
            matched_verbs = []
            
            for verb in verbs:
                # Skip level identifiers (C1, C2, etc.)
                if verb in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
                           'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7',
                           'A1', 'A2', 'A3', 'A4', 'A5']:
                    continue
                
                # Skip very short/common words that might cause false matches
                if len(verb) < 3:
                    continue
                
                verb_lower = verb.lower()
                
                # Method 1: Check if verb is at the start of the question (most reliable)
                if cleaned_question.startswith(verb_lower):
                    # Make sure it's followed by a space or word boundary
                    if len(cleaned_question) == len(verb_lower) or cleaned_question[len(verb_lower)] in ' \t\n,.;:':
                        if verb not in matched_verbs:
                            matched_verbs.append(verb)
                        continue
                
                # Method 2: Check if verb appears with common suffixes (ing, ed, s)
                verb_patterns = [
                    r'\b' + re.escape(verb_lower) + r'\b',  # exact match
                    r'\b' + re.escape(verb_lower) + r'ing\b',  # verb+ing
                    r'\b' + re.escape(verb_lower) + r'ed\b',   # verb+ed
                    r'\b' + re.escape(verb_lower) + r's\b',    # verb+s
                ]
                
                # Only match at the beginning of the cleaned question or after "to "
                search_text = cleaned_question
                for pattern in verb_patterns:
                    # Check at start of cleaned question
                    match = re.match(pattern, search_text)
                    if match:
                        if verb not in matched_verbs:
                            matched_verbs.append(verb)
                        break
                    
                    # Check after "to " (e.g., "How to explain...")
                    if search_text.startswith('to '):
                        match = re.match(pattern, search_text[3:])
                        if match:
                            if verb not in matched_verbs:
                                matched_verbs.append(verb)
                            break
            
            # Only add level if we found matching verbs
            if matched_verbs:
                detected_levels[level] = matched_verbs
        
        return detected_levels
    
    async def verify_document_auto_detect(
        self,
        file_path: str,
        domain: str
    ) -> Dict:
        """
        Verify a document with automatic level detection for each question.
        
        Args:
            file_path: Path to the document file
            domain: Domain to check (Cognitive, Psychomotor, Affective)
            
        Returns:
            Dictionary with per-question analysis and detected levels
        """
        # Validate domain
        if domain not in self.verbs_data:
            raise ValueError(f"Invalid domain: {domain}. Valid domains: {self.get_available_domains()}")
        
        # Extract text from document
        try:
            document_text = DocumentProcessor.extract_text(file_path)
        except Exception as e:
            raise ValueError(f"Failed to extract text from document: {str(e)}")
        
        if not document_text or len(document_text.strip()) < 10:
            raise ValueError("Document appears to be empty or has insufficient content")
        
        # Split into questions
        questions = self._split_into_questions(document_text)
        
        # Analyze each question
        question_analysis = []
        total_questions = len(questions)
        questions_with_match = 0
        all_detected_levels = set()
        
        for q in questions:
            detected_levels = self._detect_level_for_question(q['text'], domain)
            
            if detected_levels:
                questions_with_match += 1
                all_detected_levels.update(detected_levels.keys())
            
            question_analysis.append({
                'question_number': q['number'],
                'question_text': q['text'][:200] + '...' if len(q['text']) > 200 else q['text'],
                'detected_levels': detected_levels if detected_levels else None,
                'status': 'Match Found' if detected_levels else 'No CLO-level match found'
            })
        
        # Overall statistics
        match_percentage = (questions_with_match / total_questions * 100) if total_questions > 0 else 0
        
        return {
            'domain': domain,
            'total_questions': total_questions,
            'questions_with_match': questions_with_match,
            'questions_without_match': total_questions - questions_with_match,
            'match_percentage': round(match_percentage, 2),
            'detected_levels': sorted(list(all_detected_levels)),
            'question_analysis': question_analysis,
            'status': 'Pass' if questions_with_match > 0 else 'Fail'
        }
    
    async def verify_document(
        self,
        file_path: str,
        domain: str,
        level: str = None
    ) -> Dict:
        """
        Verify a document against OBE verb requirements using AI Agent.
        
        Args:
            file_path: Path to the document file
            domain: Domain to check (Cognitive, Psychomotor, Affective)
            level: Level to check (C1-C6, P1-P4, A1-A4)
            
        Returns:
            Dictionary with verification results
        """
        # Validate domain and level
        if domain not in self.verbs_data:
            raise ValueError(f"Invalid domain: {domain}. Valid domains: {self.get_available_domains()}")
        
        if level not in self.verbs_data[domain]:
            raise ValueError(f"Invalid level: {level}. Valid levels for {domain}: {self.get_levels_for_domain(domain)}")
        
        # Extract text from document
        try:
            document_text = DocumentProcessor.extract_text(file_path)
        except Exception as e:
            raise ValueError(f"Failed to extract text from document: {str(e)}")
        
        if not document_text or len(document_text.strip()) < 10:
            raise ValueError("Document appears to be empty or has insufficient content")
        
        # Get required verbs for the selected level
        required_verbs = self.get_verbs_for_level(domain, level)
        
        # Prepare prompt for the agent
        prompt = f"""DOMAIN: {domain}
LEVEL: {level}

REQUIRED VERBS FOR {level}:
{json.dumps(required_verbs, indent=2)}

EXAM QUESTION PAPER TEXT:
---
{document_text}
---

Please analyze this exam question paper and identify which of the required verbs are present in the questions. Consider verb variations, tenses, and contextual usage. Provide your analysis in the specified JSON format."""

        try:
            # Run the agent
            result = Runner.run_streamed(
                self.agent,
                input=[{"role": "user", "content": prompt}],
            )
            
            # Collect the full response
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            # Parse the JSON response from the agent
            try:
                # Extract JSON from response (in case there's extra text)
                json_start = full_response.find('{')
                json_end = full_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = full_response[json_start:json_end]
                    agent_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in agent response")
                
                # Extract matched verbs
                matched_verbs = agent_result.get("matched_verbs", [])
                excluded_verbs = agent_result.get("excluded_verbs", [])
                verb_details = agent_result.get("verb_details", [])
                analysis = agent_result.get("analysis", "")
                suggestions = agent_result.get("suggestions", [])
                confidence_score = agent_result.get("confidence_score", 0)
                
                # ADDITIONAL STRICT VALIDATION: Ensure matched verbs are actually in required list
                # This adds a safety check in case AI makes mistakes
                validated_matched_verbs = []
                for verb in matched_verbs:
                    # Case-insensitive check if verb is in required list
                    if any(verb.lower() == req_verb.lower() for req_verb in required_verbs):
                        validated_matched_verbs.append(verb)
                
                # If AI matched verbs that aren't in the required list, override with validated list
                if len(validated_matched_verbs) != len(matched_verbs):
                    print(f"Warning: AI matched verbs not in required list. Filtered from {len(matched_verbs)} to {len(validated_matched_verbs)}")
                    matched_verbs = validated_matched_verbs
                
                # If we found lower-level verbs but no target-level verbs, lower confidence
                if excluded_verbs and len(matched_verbs) == 0:
                    confidence_score = min(confidence_score, 30)  # Cap at 30% if only wrong-level verbs found
                    analysis += f" Note: Found {len(excluded_verbs)} lower-level verbs that don't match the target level requirements."
                
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: simple keyword matching if agent response parsing fails
                print(f"Agent response parsing failed: {e}. Using fallback method.")
                matched_verbs = self._fallback_keyword_match(document_text, required_verbs)
                excluded_verbs = []  # Can't detect excluded verbs in fallback mode
                verb_details = []
                analysis = "Analysis performed using keyword matching (agent parsing failed)"
                suggestions = ["Consider rephrasing questions for clarity"]
                confidence_score = 50
            
            # Calculate statistics
            total_required = len(required_verbs)
            total_matched = len(matched_verbs)
            match_percentage = (total_matched / total_required * 100) if total_required > 0 else 0
            
            # Determine status (Pass if at least one verb is found)
            status = "Pass" if total_matched > 0 else "Fail"
            
            # Return comprehensive results
            return {
                "domain": domain,
                "level": level,
                "matched_verbs": matched_verbs,
                "excluded_verbs": excluded_verbs,  # Lower-level verbs found but excluded
                "matched_count": total_matched,
                "total_required": total_required,
                "match_percentage": round(match_percentage, 2),
                "status": status,
                "verb_details": verb_details,
                "analysis": analysis,
                "suggestions": suggestions,
                "confidence_score": confidence_score,
                "document_preview": document_text[:300] + "..." if len(document_text) > 300 else document_text
            }
            
        except BadRequestError as e:
            raise ValueError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Verification failed: {str(e)}")
    
    def _fallback_keyword_match(self, text: str, required_verbs: List[str]) -> List[str]:
        """Fallback method for simple keyword matching if agent fails."""
        text_lower = text.lower()
        matched = []
        for verb in required_verbs:
            if verb.lower() in text_lower:
                matched.append(verb)
        return matched
    
    def _initialize_rewriter_agent(self) -> Agent:
        """Initialize OpenAI Agent for rewriting questions to match Bloom's taxonomy levels."""
        
        instructions = """You are an expert academic question writer specializing in Bloom's Taxonomy alignment.
Your role is to rewrite exam questions to match specific Bloom's Taxonomy cognitive levels while maintaining the original subject matter and educational intent.

TASK:
Rewrite the provided question to align with the specified Bloom's Taxonomy level using appropriate action verbs from the provided verb list.

CRITICAL REQUIREMENTS:
1. Use ONLY verbs from the provided verb list for the target level
2. Maintain the core subject/topic context from the original question
3. Keep the same educational objective and difficulty
4. Ensure the rewritten question is clear, grammatically correct, and academically appropriate
5. Return ONLY the rewritten question text - no explanations, no meta-commentary, just the question itself

OUTPUT FORMAT:
Return only the improved question text. Do not include:
- Explanations
- Meta-commentary
- Labels like "Improved question:"
- Any text before or after the question

Example output format:
Explain the key principles of machine learning and how they apply to real-world scenarios.

Bloom's Taxonomy Levels (for context):
- C1 (Remember): Recall facts and basic concepts
- C2 (Understand): Explain ideas or concepts
- C3 (Apply): Use information in new situations
- C4 (Analyze): Draw connections among ideas
- C5 (Evaluate): Justify a stand or decision
- C6 (Create): Produce new or original work

Be precise, professional, and ensure the question uses the appropriate cognitive level verbs."""
        
        return Agent(
            name="obe_question_rewriter_agent",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    async def rewrite_question(
        self,
        question: str,
        domain: str,
        level: str,
        context: Optional[str] = None
    ) -> str:
        """
        Rewrite a question to match a specific Bloom's Taxonomy level.
        
        Args:
            question: Original question text
            domain: Domain (Cognitive, Psychomotor, Affective)
            level: Target level (C1-C6, P1-P4, A1-A4)
            context: Optional additional context for the question
            
        Returns:
            Improved question text
        """
        # Validate inputs
        if domain not in self.verbs_data:
            raise ValueError(f"Invalid domain: {domain}")
        
        if level not in self.verbs_data[domain]:
            raise ValueError(f"Invalid level: {level} for domain: {domain}")
        
        # Get verbs for the target level
        required_verbs = self.get_verbs_for_level(domain, level)
        
        # Filter out level identifiers (C1, C2, etc.)
        action_verbs = [v for v in required_verbs if v not in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
                                                               'P1', 'P2', 'P3', 'P4', 
                                                               'A1', 'A2', 'A3', 'A4']]
        
        # Build prompt
        verbs_list = ", ".join(action_verbs[:20])  # Use first 20 verbs to avoid token limits
        if len(action_verbs) > 20:
            verbs_list += f" (and {len(action_verbs) - 20} more verbs)"
        
        prompt = f"""Rewrite the following exam question to match Bloom's taxonomy level {level}.

REQUIRED VERBS FOR {level}:
{verbs_list}

ORIGINAL QUESTION:
{question}

OPTIONAL CONTEXT:
{context if context else "None provided"}

INSTRUCTIONS:
1. Identify the core topic/subject from the original question
2. Rewrite the question using verbs from the provided {level} verb list
3. Maintain the educational context and subject matter
4. Ensure the question is clear and academically appropriate
5. Return ONLY the rewritten question text (no explanations)

Rewritten question:"""
        
        try:
            # Run the agent
            result = Runner.run_streamed(
                self.rewriter_agent,
                input=[{"role": "user", "content": prompt}],
            )
            
            # Collect the full response
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            # Clean the response - remove any prefixes or meta-text
            cleaned_response = full_response.strip()
            
            # Remove common prefixes if present
            prefixes_to_remove = [
                "Improved question:",
                "Rewritten question:",
                "Here's the improved question:",
                "Question:",
                "The rewritten question:"
            ]
            
            for prefix in prefixes_to_remove:
                if cleaned_response.lower().startswith(prefix.lower()):
                    cleaned_response = cleaned_response[len(prefix):].strip()
                    # Remove leading quotes if present
                    if cleaned_response.startswith('"') or cleaned_response.startswith("'"):
                        cleaned_response = cleaned_response[1:]
                    if cleaned_response.endswith('"') or cleaned_response.endswith("'"):
                        cleaned_response = cleaned_response[:-1]
                    break
            
            return cleaned_response.strip()
            
        except BadRequestError as e:
            raise ValueError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Question rewriting failed: {str(e)}")
    
    def get_domain_info(self) -> Dict:
        """Get information about all domains and their levels."""
        info = {}
        for domain in self.verbs_data:
            info[domain] = {
                "levels": list(self.verbs_data[domain].keys()),
                "total_verbs": sum(len(verbs) for verbs in self.verbs_data[domain].values())
            }
        return info
    
    def _initialize_generator_agent(self) -> Agent:
        """Initialize OpenAI Agent for generating OBE-aligned questions from lecture content."""
        
        instructions = """You are an expert academic question writer specializing in creating Bloom's Taxonomy-aligned exam questions from lecture content.

TASK:
Generate academic exam questions based on the provided lecture/topic content. Each question must be aligned with specific Bloom's Taxonomy levels using appropriate action verbs.

CRITICAL REQUIREMENTS:
1. Each question must use verbs appropriate for its assigned Bloom's Taxonomy level
2. Questions must be contextually accurate to the lecture content
3. Questions should test understanding at the appropriate cognitive level
4. Return questions in the specified format (JSON with question and level)
5. Distribute questions across the requested Bloom levels
6. Ensure questions are clear, grammatically correct, and academically appropriate

OUTPUT FORMAT:
Return a JSON array with objects containing:
{
  "question": "The question text",
  "level": "C1" (or C2, C3, C4, C5, C6, P1, P2, P3, P4, A1, A2, A3, A4)
}

Return ONLY valid JSON - no explanations, no meta-commentary.

Bloom's Taxonomy Levels (for context):
Cognitive Domain:
- C1 (Remember): Recall facts and basic concepts - Use verbs: Define, List, State, Name, Identify
- C2 (Understand): Explain ideas or concepts - Use verbs: Explain, Describe, Summarize, Classify
- C3 (Apply): Use information in new situations - Use verbs: Apply, Use, Demonstrate, Solve
- C4 (Analyze): Draw connections among ideas - Use verbs: Analyze, Compare, Contrast, Distinguish
- C5 (Evaluate): Justify a stand or decision - Use verbs: Evaluate, Critique, Assess, Judge
- C6 (Create): Produce new or original work - Use verbs: Create, Design, Construct, Develop

Psychomotor Domain:
- P1 (Perception): Use sensory cues to guide activity
- P2 (Set): Demonstrate readiness to act
- P3 (Guided Response): Know steps and perform with guidance
- P4 (Mechanism): Perform skillfully and with confidence

Affective Domain:
- A1 (Receiving): Show awareness and willingness to hear
- A2 (Responding): Active participation and reaction
- A3 (Valuing): Attach value and express commitment
- A4 (Organization): Integrate values into a system

Be precise, professional, and ensure questions use appropriate verbs for each level."""
        
        return Agent(
            name="obe_question_generator_agent",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    async def generate_questions(
        self,
        lecture_text: str,
        domain: str,
        levels: List[str],
        count: int
    ) -> List[Dict[str, str]]:
        """
        Generate OBE-aligned questions from lecture content.
        
        Args:
            lecture_text: Text content from lecture/material
            domain: Domain (Cognitive, Psychomotor, Affective)
            levels: List of Bloom levels to target (e.g., ['C1', 'C2', 'C3'])
            count: Number of questions to generate
            
        Returns:
            List of dicts with 'question' and 'level' keys
        """
        # Validate inputs
        if domain not in self.verbs_data:
            raise ValueError(f"Invalid domain: {domain}")
        
        # Validate levels
        for level in levels:
            if level not in self.verbs_data[domain]:
                raise ValueError(f"Invalid level: {level} for domain: {domain}")
        
        # Initialize generator agent if not already done
        if not hasattr(self, 'generator_agent'):
            self.generator_agent = self._initialize_generator_agent()
        
        # Get verbs for each level
        level_verbs_map = {}
        for level in levels:
            verbs = self.get_verbs_for_level(domain, level)
            # Filter out level identifiers
            action_verbs = [v for v in verbs if v not in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
                                                          'P1', 'P2', 'P3', 'P4', 
                                                          'A1', 'A2', 'A3', 'A4']]
            level_verbs_map[level] = action_verbs[:15]  # Limit to 15 verbs per level
        
        # Build prompt
        verbs_info = "\n".join([
            f"{level}: {', '.join(level_verbs_map[level][:10])}..."
            for level in levels
        ])
        
        # Distribute questions across levels
        questions_per_level = count // len(levels) if len(levels) > 0 else count
        remainder = count % len(levels)
        
        all_questions = []
        
        for idx, level in enumerate(levels):
            # Add one extra question to first few levels if there's a remainder
            num_for_level = questions_per_level + (1 if idx < remainder else 0)
            if num_for_level == 0:
                continue
                
            verbs_list = ", ".join(level_verbs_map[level])
            
            prompt = f"""Generate {num_for_level} exam question(s) based on the following lecture content.

LECTURE CONTENT:
{lecture_text[:3000]}  # Limit to 3000 chars to avoid token limits

TARGET BLOOM LEVEL: {level}
REQUIRED VERBS FOR {level}:
{verbs_list}

INSTRUCTIONS:
1. Generate {num_for_level} question(s) that align with Bloom's Taxonomy level {level}
2. Use verbs from the provided {level} verb list
3. Base questions on the lecture content provided
4. Ensure questions are contextually accurate and test appropriate understanding
5. Return ONLY a JSON array in this format:
[
  {{"question": "Question text here", "level": "{level}"}},
  {{"question": "Another question", "level": "{level}"}}
]

Return only the JSON array - no explanations."""
            
            try:
                # Run the agent
                result = Runner.run_streamed(
                    self.generator_agent,
                    input=[{"role": "user", "content": prompt}],
                )
                
                # Collect the full response
                full_response = ""
                async for event in result.stream_events():
                    if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                        token = event.data.delta
                        full_response += token
                
                # Extract JSON from response
                json_start = full_response.find('[')
                json_end = full_response.rfind(']') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = full_response[json_start:json_end]
                    questions = json.loads(json_str)
                    
                    # Validate format and add to all_questions
                    for q in questions:
                        if isinstance(q, dict) and 'question' in q and 'level' in q:
                            all_questions.append({
                                'question': q['question'].strip(),
                                'level': q['level']
                            })
                else:
                    # Fallback: try to parse as single question
                    cleaned = full_response.strip()
                    # Remove common prefixes
                    for prefix in ["Question:", "Q:", "Generated question:"]:
                        if cleaned.lower().startswith(prefix.lower()):
                            cleaned = cleaned[len(prefix):].strip()
                    if cleaned:
                        all_questions.append({
                            'question': cleaned,
                            'level': level
                        })
                        
            except Exception as e:
                print(f"Error generating questions for level {level}: {str(e)}")
                continue
        
        # Return up to requested count
        return all_questions[:count]

