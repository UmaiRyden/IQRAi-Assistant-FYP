"""
Course Advisor Service for analyzing student transcripts and recommending courses
"""
import os
import json
import re
from typing import List, Dict, Set, Optional
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent


class CourseAdvisorService:
    """Service for course recommendation based on transcript analysis."""
    
    def __init__(self):
        """Initialize the course advisor service."""
        self.course_data = self._load_course_data()
        if not self.course_data:
            print("WARNING: No course data loaded! Course advisor may not work correctly.")
        self.agent = self._initialize_agent()
    
    def _load_course_data(self) -> List[Dict]:
        """Load course data from JSON file."""
        # Try multiple possible paths
        possible_paths = [
            os.path.join("data", "course_data.json"),
            os.path.join("chatbot-backend", "data", "course_data.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "course_data.json")
        ]
        
        for data_path in possible_paths:
            if os.path.exists(data_path):
                try:
                    print(f"Loading course data from: {data_path}")
                    with open(data_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        # Handle both formats: direct array or nested in "courses"
                        if isinstance(data, list):
                            print(f"Loaded {len(data)} courses from JSON array")
                            return data
                        elif isinstance(data, dict) and "courses" in data:
                            courses = data.get("courses", [])
                            print(f"Loaded {len(courses)} courses from JSON object")
                            return courses
                        else:
                            print(f"Warning: Unexpected data format in {data_path}")
                            return []
                except json.JSONDecodeError as e:
                    print(f"Error parsing course data from {data_path}: {e}")
                    continue
                except Exception as e:
                    print(f"Error reading course data from {data_path}: {e}")
                    continue
        
        # If we get here, none of the paths worked
        print(f"ERROR: Course data file not found. Tried: {possible_paths}")
        return []
    
    def _initialize_agent(self) -> Agent:
        """Initialize OpenAI Agent for course advising."""
        instructions = """You are an AI Course Advisor at Iqra University. Your role is to help students plan their academic journey.

When providing course recommendations:
1. Be friendly and encouraging
2. Explain why specific courses are recommended
3. Emphasize the importance of completing prerequisites for Final Year Project (CSC441)
4. Provide clear guidance on course sequencing
5. Highlight any critical missing prerequisites that would block FYP eligibility

Always prioritize helping students understand how to unlock CSC441 (Final Year Project I) by Semester 7.

Be concise but informative in your responses."""
        
        return Agent(
            name="course_advisor_agent",
            instructions=instructions,
            model="gpt-4o-mini"
        )
    
    def extract_courses_from_text(self, text: str) -> List[str]:
        """
        Extract course codes from transcript text.
        
        Args:
            text: Transcript text content
            
        Returns:
            List of course codes found (e.g., ['CSC101', 'CSC102'])
        """
        # Pattern to match course codes like CSC101, CSC-101, CSC 101, CSC101L, etc.
        patterns = [
            r'\b([A-Z]{2,4})\s*[-]?\s*(\d{3}[A-Z]?)\b',  # CSC101, CSC-101, CSC 101, CSC101L
            r'\b([A-Z]{2,4})(\d{3}[A-Z]?)\b',  # CSC101 (no space), CSC101L
        ]
        
        found_courses = set()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                prefix = match.group(1).upper()
                number_suffix = match.group(2)
                course_code = f"{prefix}{number_suffix}"
                found_courses.add(course_code)
        
        # Also try to find courses mentioned in a table/list format
        lines = text.split('\n')
        for line in lines:
            # Look for lines with course codes (including lab codes with L suffix)
            line_matches = re.findall(r'\b([A-Z]{2,4})\s*[-]?\s*(\d{3}[A-Z]?)\b', line, re.IGNORECASE)
            for prefix, number_suffix in line_matches:
                course_code = f"{prefix.upper()}{number_suffix}"
                found_courses.add(course_code)
        
        # Filter to only include courses that exist in our course data
        valid_course_codes = {c.get("course_code") for c in self.course_data if c.get("course_code")}
        filtered_courses = [c for c in found_courses if c in valid_course_codes]
        
        print(f"Course extraction: Found {len(found_courses)} potential courses, {len(filtered_courses)} valid courses")
        
        return sorted(filtered_courses)
    
    def get_eligible_courses(self, completed_courses: List[str]) -> List[Dict]:
        """
        Get courses that the student is eligible to take.
        
        Args:
            completed_courses: List of completed course codes
            
        Returns:
            List of eligible course dictionaries
        """
        completed_set = set(completed_courses)
        eligible = []
        
        for course in self.course_data:
            course_code = course.get("course_code")
            if not course_code:
                continue  # Skip courses without a code
            
            # Skip if already completed
            if course_code in completed_set:
                continue
            
            # Check prerequisites
            prerequisites = course.get("prerequisites", [])
            if not prerequisites:
                # No prerequisites - always eligible
                eligible.append(course)
            else:
                # Extract prerequisite codes (handle both string and object formats)
                prereq_codes = []
                for prereq in prerequisites:
                    if isinstance(prereq, str):
                        prereq_codes.append(prereq)
                    elif isinstance(prereq, dict):
                        prereq_code = prereq.get("prerequisite_code")
                        if prereq_code:
                            prereq_codes.append(prereq_code)
                
                # Check if all prerequisites are met
                if all(prereq_code in completed_set for prereq_code in prereq_codes):
                    eligible.append(course)
        
        return eligible
    
    def get_fyp_prerequisites(self) -> Set[str]:
        """
        Get all prerequisites needed for Final Year Project (CSC441).
        
        Returns:
            Set of course codes that are prerequisites for CSC441 (recursively)
        """
        fyp_code = "CSC441"
        prerequisites_set = set()
        
        def get_prereqs_recursive(course_code: str, visited: Set[str]):
            """Recursively collect all prerequisites."""
            if course_code in visited:
                return
            
            visited.add(course_code)
            
            # Find course in data
            course = next((c for c in self.course_data if c.get("course_code") == course_code), None)
            if not course:
                # Course not found - skip silently (might be elective or missing)
                return
            
            prereqs = course.get("prerequisites", [])
            # Extract prerequisite codes (handle both string and object formats)
            for prereq in prereqs:
                if isinstance(prereq, str):
                    prereq_code = prereq
                elif isinstance(prereq, dict):
                    prereq_code = prereq.get("prerequisite_code")
                    if not prereq_code:
                        continue
                else:
                    continue
                
                prerequisites_set.add(prereq_code)
                get_prereqs_recursive(prereq_code, visited)
        
        get_prereqs_recursive(fyp_code, set())
        return prerequisites_set
    
    def analyze_fyp_progress(self, completed_courses: List[str]) -> Dict:
        """
        Analyze progress toward Final Year Project eligibility.
        
        Args:
            completed_courses: List of completed course codes
            
        Returns:
            Dictionary with FYP progress information
        """
        completed_set = set(completed_courses)
        fyp_prereqs = self.get_fyp_prerequisites()
        
        # Check FYP prerequisites
        completed_prereqs = [code for code in fyp_prereqs if code in completed_set]
        remaining_prereqs = [code for code in fyp_prereqs if code not in completed_set]
        
        # Get missing course details
        missing_courses = []
        for code in remaining_prereqs:
            course = next((c for c in self.course_data if c["course_code"] == code), None)
            if course:
                missing_courses.append({
                    "course_code": course["course_code"],
                    "course_name": course["course_name"],
                    "semester": course.get("semester", 0)
                })
        
        # Sort missing courses by semester
        missing_courses.sort(key=lambda x: x["semester"])
        
        return {
            "completed_prereqs": len(completed_prereqs),
            "remaining_prereqs": len(remaining_prereqs),
            "total_prereqs": len(fyp_prereqs),
            "missing_courses": missing_courses,
            "completed_prereq_codes": completed_prereqs,
            "remaining_prereq_codes": [c["course_code"] for c in missing_courses]
        }
    
    def prioritize_courses(self, eligible_courses: List[Dict], completed_courses: List[str]) -> List[Dict]:
        """
        Prioritize courses, especially those needed for FYP.
        
        Args:
            eligible_courses: List of eligible courses
            completed_courses: List of completed courses
            
        Returns:
            Prioritized list of courses (FYP prerequisites first)
        """
        fyp_progress = self.analyze_fyp_progress(completed_courses)
        fyp_remaining = set(fyp_progress["remaining_prereq_codes"])
        
        # Separate courses into FYP-related and others
        fyp_related = []
        other_courses = []
        
        for course in eligible_courses:
            course_code = course["course_code"]
            if course_code in fyp_remaining:
                course["is_fyp_prerequisite"] = True
                fyp_related.append(course)
            else:
                course["is_fyp_prerequisite"] = False
                other_courses.append(course)
        
        # Sort FYP-related by semester (earlier semesters first)
        fyp_related.sort(key=lambda x: x.get("semester", 999))
        
        # Sort other courses by semester
        other_courses.sort(key=lambda x: x.get("semester", 999))
        
        # Return FYP-related first, then others
        return fyp_related + other_courses
    
    async def generate_ai_advice(
        self,
        completed_courses: List[str],
        eligible_courses: List[Dict],
        fyp_progress: Dict
    ) -> str:
        """
        Generate AI-powered course advice.
        
        Args:
            completed_courses: List of completed course codes
            eligible_courses: List of eligible courses
            fyp_progress: FYP progress dictionary
            
        Returns:
            AI-generated advice text
        """
        # Prepare context for the agent
        completed_count = len(completed_courses)
        eligible_count = len(eligible_courses)
        
        # Find top recommended courses (especially FYP-related)
        fyp_related = [c for c in eligible_courses if c.get("is_fyp_prerequisite", False)]
        top_recommendations = fyp_related[:3] if fyp_related else eligible_courses[:3]
        
        prompt = f"""Based on the student's transcript analysis:

COMPLETED COURSES ({completed_count}):
{', '.join(completed_courses) if completed_courses else 'None'}

ELIGIBLE NEXT COURSES ({eligible_count}):
{chr(10).join([f"- {c['course_code']}: {c['course_name']}" for c in top_recommendations])}

FINAL YEAR PROJECT (CSC441) PROGRESS:
- Completed Prerequisites: {fyp_progress['completed_prereqs']}/{fyp_progress['total_prereqs']}
- Remaining Prerequisites: {fyp_progress['remaining_prereqs']}
{f"- Missing: {', '.join(fyp_progress['remaining_prereq_codes'][:5])}" if fyp_progress['remaining_prereq_codes'] else "- All prerequisites completed!"}

Provide a brief, friendly recommendation (2-3 sentences) on:
1. Which courses to take next semester
2. Progress toward Final Year Project eligibility
3. How to ensure FYP can be unlocked by Semester 7

Be encouraging and specific."""
        
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
            
            return full_response.strip()
        except Exception as e:
            print(f"Error generating AI advice: {e}")
            # Fallback response
            return f"Based on your transcript, you have completed {completed_count} courses and are eligible for {eligible_count} courses. To unlock Final Year Project (CSC441) by Semester 7, you need to complete {fyp_progress['remaining_prereqs']} more prerequisite courses."
    
    async def analyze_transcript(self, transcript_text: str) -> Dict:
        """
        Complete transcript analysis pipeline.
        
        Args:
            transcript_text: Text content from uploaded transcript
            
        Returns:
            Complete analysis dictionary
        """
        try:
            # Extract completed courses
            print(f"Extracting courses from transcript text (length: {len(transcript_text)})")
            completed_courses = self.extract_courses_from_text(transcript_text)
            print(f"Found {len(completed_courses)} completed courses: {completed_courses[:5]}...")
            
            # Get eligible courses
            eligible_courses = self.get_eligible_courses(completed_courses)
            print(f"Found {len(eligible_courses)} eligible courses")
            
            # Analyze FYP progress
            fyp_progress = self.analyze_fyp_progress(completed_courses)
            print(f"FYP progress: {fyp_progress.get('completed_prereqs')}/{fyp_progress.get('total_prereqs')} prerequisites")
            
            # Prioritize courses (FYP prerequisites first)
            prioritized_courses = self.prioritize_courses(eligible_courses, completed_courses)
            print(f"Prioritized {len(prioritized_courses)} courses")
            
            # Generate AI advice
            print("Generating AI advice...")
            ai_advice = await self.generate_ai_advice(
                completed_courses,
                prioritized_courses,
                fyp_progress
            )
            print("AI advice generated successfully")
        except Exception as e:
            print(f"Error in analyze_transcript: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Get completed course details
        completed_course_details = []
        for code in completed_courses:
            course = next((c for c in self.course_data if c.get("course_code") == code), None)
            if course:
                completed_course_details.append({
                    "course_code": course.get("course_code", code),
                    "course_name": course.get("course_name", "Unknown Course"),
                    "semester": course.get("semester", 0)
                })
        
        return {
            "completed_courses": completed_courses,
            "completed_course_details": completed_course_details,
            "eligible_courses": prioritized_courses,
            "fyp_progress": fyp_progress,
            "ai_advice": ai_advice
        }

