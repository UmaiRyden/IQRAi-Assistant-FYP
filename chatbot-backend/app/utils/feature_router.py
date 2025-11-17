"""
Feature Router - Detects user intent and routes to appropriate in-app features.

This module provides intelligent routing to guide users to built-in features
instead of providing generic responses.
"""

from typing import Optional, Dict, List
import re


class FeatureRouter:
    """Routes user queries to appropriate in-app features."""
    
    # Feature detection patterns
    EMAIL_KEYWORDS = [
        r'\bsend\s+email\b', r'\bemail\s+to\b', r'\bwrite\s+email\b',
        r'\bdraft\s+email\b', r'\bcompose\s+email\b', r'\bemail\s+template\b',
        r'\bhow\s+to\s+send\s+email\b', r'\bemail\s+student\s+affairs\b',
        r'\bemail\s+department\b', r'\bcontact\s+via\s+email\b',
        r'\bemail\s+administration\b', r'\bemail\s+office\b'
    ]
    
    COURSE_ADVISOR_KEYWORDS = [
        r'\btranscript\b', r'\bcourse\s+planning\b', r'\bsemester\s+plan\b',
        r'\bprerequisite\b', r'\bprerequisites\b', r'\bdegree\s+completion\b',
        r'\bwhat\s+courses\s+to\s+take\b', r'\bnext\s+semester\b',
        r'\bcourse\s+recommendation\b', r'\bcourse\s+suggestion\b',
        r'\bwhich\s+courses\b', r'\beligible\s+courses\b',
        r'\bremaining\s+courses\b', r'\bfyp\s+prerequisite\b',
        r'\bgraduation\s+requirements\b', r'\bcourse\s+selection\b',
        r'\bacademic\s+plan\b', r'\bstudy\s+plan\b'
    ]
    
    OBE_KEYWORDS = [
        r'\bbloom\s+taxonomy\b', r'\bbloom\'?s\s+taxonomy\b',
        r'\bobe\b', r'\boutcome\s+based\s+education\b',
        r'\bquestion\s+improvement\b', r'\bimprove\s+question\b',
        r'\bexam\s+question\b', r'\bverify\s+question\b',
        r'\bquestion\s+verification\b', r'\bexam\s+paper\b',
        r'\bassessment\s+verb\b', r'\baction\s+verb\b',
        r'\bcognitive\s+domain\b', r'\bpsychomotor\b', r'\baffective\b',
        r'\bc1\b', r'\bc2\b', r'\bc3\b', r'\bc4\b', r'\bc5\b', r'\bc6\b',
        r'\bp1\b', r'\bp2\b', r'\bp3\b', r'\bp4\b',
        r'\ba1\b', r'\ba2\b', r'\ba3\b', r'\ba4\b'
    ]
    
    LEARNING_KEYWORDS = [
        r'\bquiz\b', r'\bgenerate\s+quiz\b', r'\bcreate\s+quiz\b',
        r'\bstudy\s+help\b', r'\bstudy\s+mode\b', r'\bsocratic\b',
        r'\bsocratic\s+mode\b', r'\bexplain\s+concept\b',
        r'\blearning\s+material\b', r'\bstudy\s+guide\b',
        r'\bpractice\s+question\b', r'\bflashcard\b',
        r'\bhelp\s+me\s+study\b', r'\bhow\s+to\s+study\b',
        r'\bunderstand\s+this\b', r'\bexplain\s+this\b',
        r'\bassignment\b', r'\bhomework\s+help\b'
    ]
    
    # Feature descriptions for guidance
    FEATURE_GUIDES = {
        'email': {
            'name': 'Send Email',
            'description': 'built-in email drafting and sending feature',
            'location': 'the **Send Email** button in the chat interface',
            'capabilities': [
                'Automatically formats emails with your student information',
                'Pre-fills department contacts',
                'Allows editing before sending',
                'Sends emails directly from the application'
            ],
            'guidance': (
                "I can help you send emails! IQRAi has a built-in **Send Email** feature "
                "that makes it easy to draft and send emails to university departments.\n\n"
                "**To use it:**\n"
                "1. Ask me to draft an email (e.g., 'Draft an email to student affairs about...')\n"
                "2. Once I generate the email, click the **📧 Send Email** button in the chat interface\n"
                "3. Select the department, review/edit the email, and send it directly\n\n"
                "The feature automatically includes your student information and formats everything properly. "
                "Would you like me to draft an email for you now?"
            )
        },
        'course_advisor': {
            'name': 'Course Advisor',
            'description': 'transcript analysis and course recommendation tool',
            'location': 'the **Course Advisor** tab in the sidebar',
            'capabilities': [
                'Analyzes your transcript automatically',
                'Identifies completed courses and prerequisites',
                'Recommends eligible next courses',
                'Tracks FYP (CSC441) prerequisite progress',
                'Provides semester planning suggestions'
            ],
            'guidance': (
                "Great question! IQRAi has a dedicated **Course Advisor** tool that can help with exactly this.\n\n"
                "**To use it:**\n"
                "1. Go to the **Course Advisor** tab in the sidebar\n"
                "2. Upload your transcript (PDF, DOCX, or TXT)\n"
                "3. The system will automatically:\n"
                "   • Analyze your completed courses\n"
                "   • Identify prerequisites you've met\n"
                "   • Recommend eligible next courses\n"
                "   • Show your FYP (CSC441) prerequisite progress\n"
                "   • Suggest semester planning options\n\n"
                "This is much more accurate than manual recommendations because it analyzes your actual transcript. "
                "Would you like to try it? Just navigate to the Course Advisor tab and upload your transcript!"
            )
        },
        'obe': {
            'name': 'OBE Verification',
            'description': 'Bloom\'s Taxonomy and OBE question verification tool',
            'location': 'the **OBE Verification** tab in the sidebar',
            'capabilities': [
                'Verifies exam questions against Bloom\'s Taxonomy levels',
                'Detects appropriate action verbs (C1-C6, P1-P4, A1-A4)',
                'Suggests question improvements',
                'Auto-detects question levels',
                'Rewrites questions to match target levels'
            ],
            'guidance': (
                "Perfect! IQRAi has a specialized **OBE Verification** tool for this.\n\n"
                "**To use it:**\n"
                "1. Go to the **OBE Verification** tab in the sidebar\n"
                "2. Upload your exam paper or enter a question\n"
                "3. Select the domain (Cognitive, Psychomotor, or Affective) and level\n"
                "4. The system will:\n"
                "   • Verify if questions match the target Bloom's Taxonomy level\n"
                "   • Identify appropriate action verbs\n"
                "   • Suggest improvements if needed\n"
                "   • Auto-detect question levels\n"
                "   • Rewrite questions to match specific levels\n\n"
                "This tool is specifically designed for OBE compliance and question quality assurance. "
                "Navigate to the OBE Verification tab to get started!"
            )
        },
        'learning': {
            'name': 'Learning Interface',
            'description': 'quiz generation, study mode, and Socratic dialogue tools',
            'location': 'the **Learning** tab in the sidebar',
            'capabilities': [
                'Generate quizzes from uploaded materials',
                'Study Mode: AI asks questions to test your understanding',
                'Socratic Mode: Interactive dialogue to deepen understanding',
                'Multiple question types (MCQ, True/False, Short Answer)',
                'Automatic feedback and explanations'
            ],
            'guidance': (
                "Excellent! IQRAi has a comprehensive **Learning** interface with multiple study tools.\n\n"
                "**To use it:**\n"
                "1. Go to the **Learning** tab in the sidebar\n"
                "2. Upload your course materials (PDF, DOCX, PPTX, TXT)\n"
                "3. Choose from these features:\n"
                "   • **Quiz Generator**: Create quizzes from your materials\n"
                "   • **Study Mode**: AI asks you questions to test understanding\n"
                "   • **Socratic Mode**: Interactive dialogue to explore concepts deeply\n\n"
                "All tools work with your uploaded materials and provide personalized learning support. "
                "Navigate to the Learning tab to get started!"
            )
        }
    }
    
    @classmethod
    def detect_feature(cls, user_message: str) -> Optional[str]:
        """
        Detect which feature the user is asking about.
        
        Args:
            user_message: The user's message/query
            
        Returns:
            Feature name ('email', 'course_advisor', 'obe', 'learning') or None
        """
        message_lower = user_message.lower()
        
        # Check email features
        for pattern in cls.EMAIL_KEYWORDS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return 'email'
        
        # Check course advisor features
        for pattern in cls.COURSE_ADVISOR_KEYWORDS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return 'course_advisor'
        
        # Check OBE features
        for pattern in cls.OBE_KEYWORDS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return 'obe'
        
        # Check learning features
        for pattern in cls.LEARNING_KEYWORDS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                return 'learning'
        
        return None
    
    @classmethod
    def get_feature_guidance(cls, feature_name: str) -> Optional[str]:
        """
        Get guidance message for a specific feature.
        
        Args:
            feature_name: Name of the feature ('email', 'course_advisor', 'obe', 'learning')
            
        Returns:
            Guidance message string or None if feature not found
        """
        return cls.FEATURE_GUIDES.get(feature_name, {}).get('guidance')
    
    @classmethod
    def get_feature_info(cls, feature_name: str) -> Optional[Dict]:
        """
        Get full feature information.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature info dictionary or None
        """
        return cls.FEATURE_GUIDES.get(feature_name)
    
    @classmethod
    def should_redirect(cls, user_message: str, has_document_context: bool = False) -> bool:
        """
        Determine if user should be redirected to a feature.
        
        Args:
            user_message: The user's message
            has_document_context: Whether user has uploaded documents
            
        Returns:
            True if redirect is appropriate, False otherwise
        """
        # If user is asking about uploaded document content, don't redirect
        if has_document_context:
            return False
        
        # Check if message matches any feature pattern
        detected = cls.detect_feature(user_message)
        return detected is not None

