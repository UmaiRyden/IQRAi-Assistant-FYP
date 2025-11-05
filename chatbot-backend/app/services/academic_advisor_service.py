"""
Academic Advisor Service for generating study plans and career guidance
"""
import os
import json
from typing import List, Dict, Optional
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
from app.services.course_advisor_service import CourseAdvisorService
from app.utils.document_processor import DocumentProcessor


class AcademicAdvisorService:
    """Service for academic advising including study plans and career guidance."""
    
    def __init__(self):
        """Initialize the academic advisor service."""
        self.course_advisor = CourseAdvisorService()
        self.document_processor = DocumentProcessor()
        self.youtube_channels = self._load_youtube_channels()
        self.course_data = self._load_course_data()
    
    def _load_youtube_channels(self) -> Dict[str, List[str]]:
        """Load YouTube channels mapping from JSON file."""
        possible_paths = [
            os.path.join("data", "youtube_channels.json"),
            os.path.join("chatbot-backend", "data", "youtube_channels.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "youtube_channels.json"),
        ]
        
        for data_path in possible_paths:
            if os.path.exists(data_path):
                try:
                    with open(data_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        print(f"Loaded YouTube channels from: {data_path}")
                        return data
                except Exception as e:
                    print(f"Error loading YouTube channels from {data_path}: {e}")
                    continue
        
        print("WARNING: YouTube channels file not found. Using defaults.")
        return {}
    
    def _load_course_data(self) -> List[Dict]:
        """Load course data from JSON file."""
        return self.course_advisor.course_data if hasattr(self.course_advisor, 'course_data') else []
    
    def _initialize_study_plan_agent(self) -> Agent:
        """Initialize OpenAI Agent for study plan generation."""
        return Agent(
            name="study_plan_generator",
            instructions="""You are an expert academic advisor specializing in creating personalized study plans for university students.
Your role is to:
1. Analyze course content and difficulty
2. Recommend appropriate study priorities (High/Medium/Low)
3. Suggest relevant YouTube channels for learning
4. Recommend practice platforms (LeetCode, Coursera, Udemy, Kaggle, etc.)
5. Create a detailed weekly study outline that balances all courses

Guidelines:
- Prioritize courses based on difficulty, prerequisites, and exam schedules
- Recommend YouTube channels that match the course content
- Suggest platforms that offer hands-on practice
- Create realistic weekly schedules (consider 2-3 hours per course per week minimum)
- Balance study time across all courses
- Include review sessions and practice time

Format your response as JSON with clear structure for each course.""",
            model="gpt-4o-mini",
        )
    
    def _initialize_career_agent(self) -> Agent:
        """Initialize OpenAI Agent for career analysis."""
        return Agent(
            name="career_advisor",
            instructions="""You are an expert career counselor specializing in computer science education and career planning.
Your role is to:
1. Analyze student transcripts to identify strengths and weaknesses
2. Match career interests with required skills and courses
3. Recommend relevant electives from the curriculum
4. Create learning pathways for skill development
5. Suggest resources (YouTube channels, courses, platforms)
6. Provide guidance on master's preparation and career paths

Guidelines:
- Be specific about which courses build strong foundations
- Identify gaps in knowledge that need improvement
- Recommend electives that align with career goals
- Create step-by-step learning pathways
- Suggest practical resources (YouTube, Coursera, Udemy, etc.)
- Consider international master's programs when relevant
- Provide academic, professional tone

Format your response as structured JSON with clear sections.""",
            model="gpt-4o-mini",
        )
    
    def _initialize_chat_agent(self) -> Agent:
        """Initialize OpenAI Agent for interactive career chat."""
        return Agent(
            name="career_chat_advisor",
            instructions="""You are a friendly and knowledgeable career advisor for computer science students.
Your role is to:
1. Answer questions about career paths, electives, and course selection
2. Provide guidance on master's programs (especially international)
3. Help students understand how courses relate to career goals
4. Suggest improvements to their academic plan
5. Explain prerequisites and course relationships

Guidelines:
- Be conversational but professional
- Reference specific course codes when relevant
- Provide actionable advice
- Consider the student's transcript and career interest context
- Be encouraging and supportive
- If asked about master's programs, provide specific guidance on prerequisites and preparation

Keep responses concise but informative.""",
            model="gpt-4o-mini",
        )
    
    async def generate_study_plan(
        self,
        courses: List[Dict[str, str]],
        auto_save: bool = False
    ) -> Dict[str, List[Dict]]:
        """Generate a personalized study plan for semester courses."""
        try:
            # Get course information from course_data.json
            course_details = []
            for course in courses:
                code = course.get("code", "").upper()
                name = course.get("name", "")
                
                # Find course in course_data
                course_info = None
                for c in self.course_data:
                    if c.get("course_code", "").upper() == code:
                        course_info = c
                        break
                
                course_details.append({
                    "code": code,
                    "name": name,
                    "info": course_info
                })
            
            # Get YouTube channels for courses
            channels_map = {}
            for course in course_details:
                code = course["code"]
                # Try exact match first
                if code in self.youtube_channels:
                    channels_map[code] = self.youtube_channels[code]
                else:
                    # Try general categories
                    if "data" in code.lower() or "ml" in code.lower() or "ai" in code.lower():
                        channels_map[code] = self.youtube_channels.get("ai_ml", [])
                    elif "web" in code.lower() or "dev" in code.lower():
                        channels_map[code] = self.youtube_channels.get("web_dev", [])
                    else:
                        channels_map[code] = self.youtube_channels.get("general_programming", [])
            
            # Prepare prompt for AI agent
            courses_text = "\n".join([
                f"- {c['code']}: {c['name']}" + 
                (f" (Semester {c['info'].get('semester', 'N/A')})" if c['info'] else "")
                for c in course_details
            ])
            
            prompt = f"""Generate a comprehensive study plan for the following courses:

{courses_text}

For each course, provide:
1. Study Priority: High/Medium/Low (based on difficulty, prerequisites, and importance)
2. Recommended YouTube Channels: {json.dumps(channels_map, indent=2)}
3. Practice Platforms: Suggest from LeetCode, Coursera, Udemy, Kaggle, HackerRank, Codecademy, etc.
4. Weekly Study Outline: A detailed week-by-week breakdown (16 weeks) covering:
   - Topics to cover each week
   - Practice exercises
   - Review sessions
   - Assignment deadlines (estimate)
   - Exam preparation

Return ONLY a valid JSON array in this exact format:
[
  {{
    "course_name": "Course Code - Course Name",
    "study_priority": "High|Medium|Low",
    "recommended_channels": ["Channel 1", "Channel 2", ...],
    "practice_platforms": ["Platform 1", "Platform 2", ...],
    "weekly_outline": "Week 1: ...\nWeek 2: ...\n..."
  }},
  ...
]

Ensure:
- Priorities are realistic (not all High)
- Channels match the course content
- Platforms offer relevant practice
- Weekly outline is detailed and achievable"""
            
            agent = self._initialize_study_plan_agent()
            
            # Run the agent using Runner.run_streamed
            result = Runner.run_streamed(
                agent,
                input=[{"role": "user", "content": prompt}],
            )
            
            # Collect the full response
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            response_text = full_response.strip()
            
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                study_plan = json.loads(json_text)
            else:
                # Fallback: try to parse entire response
                study_plan = json.loads(response_text)
            
            # Enhance with YouTube channels if missing
            for i, plan in enumerate(study_plan):
                course_code = course_details[i]["code"] if i < len(course_details) else ""
                if course_code and course_code in channels_map:
                    if not plan.get("recommended_channels"):
                        plan["recommended_channels"] = channels_map[course_code]
                if not plan.get("practice_platforms"):
                    plan["practice_platforms"] = ["LeetCode", "Coursera", "Udemy"]
            
            return {
                "study_plan": study_plan,
                "auto_saved": auto_save
            }
            
        except Exception as e:
            print(f"Error generating study plan: {e}")
            raise Exception(f"Failed to generate study plan: {str(e)}")
    
    async def analyze_career_path(
        self,
        transcript_text: Optional[str],
        area_of_interest: str,
        completed_courses: Optional[List[str]] = None
    ) -> Dict:
        """Analyze career path and provide recommendations."""
        try:
            # If transcript text provided, extract courses
            if transcript_text and not completed_courses:
                completed_courses = self.course_advisor.extract_courses_from_text(transcript_text)
            
            # Get eligible courses and analyze
            eligible_courses = []
            if completed_courses:
                eligible_courses = self.course_advisor.get_eligible_courses(completed_courses)
            
            # Get electives related to area of interest
            interest_keywords = {
                "Artificial Intelligence": ["AI", "ML", "Machine Learning", "Neural", "Deep Learning"],
                "Data Science": ["Data", "Statistics", "Analytics", "ML"],
                "Cybersecurity": ["Security", "Network", "Cryptography", "Ethical"],
                "Software Engineering": ["Software", "Engineering", "Design", "Development"],
                "Web / Mobile Development": ["Web", "Mobile", "Development", "App"],
                "Game Development": ["Game", "Graphics", "Animation"],
                "Cloud / DevOps": ["Cloud", "DevOps", "Infrastructure", "Deployment"]
            }
            
            # Find relevant electives
            relevant_electives = []
            keywords = interest_keywords.get(area_of_interest, [])
            for course in self.course_data:
                course_name = course.get("course_name", "").lower()
                course_code = course.get("course_code", "").upper()
                
                # Check if elective (not in completed courses)
                if completed_courses and course_code in completed_courses:
                    continue
                
                # Check if matches interest
                matches = any(keyword.lower() in course_name or keyword.lower() in course_code.lower() 
                            for keyword in keywords)
                if matches:
                    relevant_electives.append({
                        "code": course_code,
                        "name": course.get("course_name", ""),
                        "reason": f"Aligns with {area_of_interest} career path"
                    })
            
            # Prepare prompt for AI agent
            completed_text = ", ".join(completed_courses) if completed_courses else "None specified"
            
            prompt = f"""Analyze the student's academic profile and provide career guidance:

Area of Interest: {area_of_interest}
Completed Courses: {completed_text}

Provide a comprehensive career analysis including:

1. Course Strength Analysis:
   - List 5-8 key courses from transcript
   - For each: course code, estimated grade (if available), and strength assessment (Strong/Average/Weak)
   - Base assessment on relevance to career interest

2. Weak Areas to Improve:
   - Identify 2-4 areas where student needs improvement
   - Be specific about skills or knowledge gaps

3. Suggested Electives:
   - Recommend 3-5 electives from the curriculum
   - For each: course code, name, and reason for recommendation
   - Prioritize courses that build toward career goal

4. Learning Pathway:
   - Create a step-by-step learning pathway (5-8 steps)
   - Each step should be actionable and specific
   - Include skill development and course recommendations

5. Recommended Resources:
   - Group by platform (YouTube, Coursera, Udemy, etc.)
   - Provide specific resource names or links
   - Focus on {area_of_interest} related content

Return ONLY a valid JSON object in this exact format:
{{
  "strength_analysis": [
    {{"course": "CSC101", "grade": "A", "strength": "Strong"}},
    ...
  ],
  "weak_areas": [
    "Area 1 description",
    "Area 2 description",
    ...
  ],
  "suggested_electives": [
    {{"code": "CSCXXX", "name": "Course Name", "reason": "Why this course helps..."}},
    ...
  ],
  "learning_pathway": [
    "Step 1: ...",
    "Step 2: ...",
    ...
  ],
  "recommended_resources": [
    {{"platform": "YouTube", "links": ["Channel 1", "Channel 2", ...]}},
    {{"platform": "Coursera", "links": ["Course 1", "Course 2", ...]}},
    ...
  ]
}}

Ensure all recommendations are specific and actionable."""
            
            agent = self._initialize_career_agent()
            
            # Run the agent using Runner.run_streamed
            result = Runner.run_streamed(
                agent,
                input=[{"role": "user", "content": prompt}],
            )
            
            # Collect the full response
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            response_text = full_response.strip()
            print(f"📝 Raw AI response (first 500 chars): {response_text[:500]}")
            
            # Extract JSON from response
            analysis = None
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                try:
                    analysis = json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing error: {e}")
                    print(f"📝 Attempted to parse: {json_text[:200]}...")
                    # Try to extract just the analysis part
                    raise Exception(f"Failed to parse AI response as JSON: {str(e)}")
            else:
                # Try to parse entire response
                try:
                    analysis = json.loads(response_text)
                except json.JSONDecodeError:
                    raise Exception(f"AI response does not contain valid JSON. Response: {response_text[:200]}...")
            
            # Validate analysis structure
            if not isinstance(analysis, dict):
                raise Exception("AI response is not a valid JSON object")
            
            # Ensure all required fields exist
            if "strength_analysis" not in analysis:
                analysis["strength_analysis"] = []
            if "weak_areas" not in analysis:
                analysis["weak_areas"] = []
            if "suggested_electives" not in analysis:
                analysis["suggested_electives"] = []
            if "learning_pathway" not in analysis:
                analysis["learning_pathway"] = []
            if "recommended_resources" not in analysis:
                analysis["recommended_resources"] = []
            
            # Merge with electives we found
            if relevant_electives:
                existing_codes = {e.get("code", "") for e in analysis.get("suggested_electives", [])}
                for elective in relevant_electives:
                    if elective["code"] not in existing_codes:
                        analysis.setdefault("suggested_electives", []).append(elective)
            
            return {
                "analysis": analysis
            }
            
        except Exception as e:
            print(f"Error analyzing career path: {e}")
            raise Exception(f"Failed to analyze career path: {str(e)}")
    
    async def chat_with_advisor(
        self,
        message: str,
        context: Optional[Dict] = None,
        area_of_interest: Optional[str] = None
    ) -> Dict[str, str]:
        """Chat with career advisor."""
        try:
            # Build context string
            context_str = ""
            if context:
                context_str += f"\nCareer Analysis Context:\n{json.dumps(context, indent=2)}\n"
            if area_of_interest:
                context_str += f"\nArea of Interest: {area_of_interest}\n"
            
            prompt = f"""Student Question: {message}
{context_str}

Provide a helpful, specific answer to the student's question. Reference courses, electives, and career paths when relevant.
Be conversational but professional. Keep the response concise (2-4 paragraphs)."""
            
            agent = self._initialize_chat_agent()
            
            # Run the agent using Runner.run_streamed
            result = Runner.run_streamed(
                agent,
                input=[{"role": "user", "content": prompt}],
            )
            
            # Collect the full response
            full_response = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    token = event.data.delta
                    full_response += token
            
            response_text = full_response.strip()
            
            return {
                "response": response_text.strip()
            }
            
        except Exception as e:
            print(f"Error in advisor chat: {e}")
            raise Exception(f"Failed to get advisor response: {str(e)}")

