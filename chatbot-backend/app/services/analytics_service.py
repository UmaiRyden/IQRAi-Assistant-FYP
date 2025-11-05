"""
Analytics tracking service for user activity logging
"""
import os
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

ANALYTICS_FILE = "analytics.json"


class AnalyticsService:
    """Service for tracking and retrieving user analytics."""
    
    def __init__(self):
        """Initialize analytics service."""
        self.analytics_file = ANALYTICS_FILE
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create analytics file if it doesn't exist."""
        if not os.path.exists(self.analytics_file):
            with open(self.analytics_file, 'w') as f:
                json.dump([], f)
    
    def track_event(
        self,
        user: str,
        action: str,
        meta: Optional[Dict] = None
    ):
        """
        Track a user action/event.
        
        Args:
            user: User identifier (email or session_id)
            action: Action type (ai_chat, document_upload, quiz_generated, study_session, obe_verification, obe_rewrite)
            meta: Optional metadata about the action
        """
        event = {
            "user": user,
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "meta": meta or {}
        }
        
        try:
            # Read existing events
            with open(self.analytics_file, 'r') as f:
                events = json.load(f)
            
            # Append new event
            events.append(event)
            
            # Keep only last 10000 events to avoid file bloat
            if len(events) > 10000:
                events = events[-10000:]
            
            # Write back
            with open(self.analytics_file, 'w') as f:
                json.dump(events, f, indent=2)
        except Exception as e:
            print(f"Error tracking event: {e}")
    
    def get_recent_events(self, limit: int = 20) -> List[Dict]:
        """
        Get recent events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of recent events (most recent first)
        """
        try:
            with open(self.analytics_file, 'r') as f:
                events = json.load(f)
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return events[:limit]
        except Exception as e:
            print(f"Error reading events: {e}")
            return []
    
    def get_dashboard_stats(self) -> Dict:
        """
        Get aggregated statistics for dashboard.
        
        Returns:
            Dictionary with aggregated statistics
        """
        try:
            with open(self.analytics_file, 'r') as f:
                events = json.load(f)
            
            # Initialize counters
            stats = {
                "total_chats": 0,
                "obe_verifications": 0,
                "obe_rewrites": 0,
                "quizzes_generated": 0,
                "learning_sessions": 0,
                "gpa_calculated": 0,
                "course_advisor": 0,
                "transcript_advisor": 0,
                "email_sent": 0,
                "weekly_activity": self._get_weekly_activity(events),
                "feature_distribution": self._get_feature_distribution(events),
                "learning_trends": self._get_learning_trends(events)
            }
            
            # Count events by type
            for event in events:
                action = event.get('action', '')
                if action == 'ai_chat':
                    stats["total_chats"] += 1
                elif action == 'obe_verification':
                    stats["obe_verifications"] += 1
                elif action == 'obe_rewrite':
                    stats["obe_rewrites"] += 1
                elif action == 'quiz_generated':
                    stats["quizzes_generated"] += 1
                elif action == 'study_session':
                    stats["learning_sessions"] += 1
                elif action == 'GPA_CALCULATED' or action == 'CGPA_CALCULATED':
                    stats["gpa_calculated"] += 1
                elif action == 'COURSE_ADVISOR':
                    stats["course_advisor"] += 1
                elif action == 'TRANSCRIPT_ADVISOR':
                    stats["transcript_advisor"] += 1
                elif action == 'EMAIL_SENT':
                    stats["email_sent"] += 1
                # Legacy support for old action names
                elif action == 'academic_advisor':
                    meta = event.get('meta', {})
                    if meta.get('type') == 'study_plan':
                        stats["course_advisor"] += 1
                    elif meta.get('type') == 'career_analysis':
                        stats["transcript_advisor"] += 1
                    else:
                        stats["course_advisor"] += 1
                elif action == 'email_sent':
                    stats["email_sent"] += 1
            
            return stats
        except Exception as e:
            print(f"Error calculating stats: {e}")
            return {
                "total_chats": 0,
                "obe_verifications": 0,
                "obe_rewrites": 0,
                "quizzes_generated": 0,
                "learning_sessions": 0,
                "gpa_calculated": 0,
                "course_advisor": 0,
                "transcript_advisor": 0,
                "email_sent": 0,
                "weekly_activity": [],
                "feature_distribution": [],
                "learning_trends": []
            }
    
    def _get_weekly_activity(self, events: List[Dict]) -> List[Dict]:
        """Get activity data for the last 7 days."""
        now = datetime.now()
        days = []
        
        for i in range(6, -1, -1):  # Last 7 days
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            day_events = [
                e for e in events
                if day_start.isoformat() <= e.get('timestamp', '') < day_end.isoformat()
            ]
            
            # Count by action type
            day_counts = defaultdict(int)
            for event in day_events:
                action = event.get('action', '')
                day_counts[action] += 1
            
            # Count new action types (including legacy support)
            gpa_count = day_counts.get('GPA_CALCULATED', 0) + day_counts.get('CGPA_CALCULATED', 0)
            course_advisor_count = day_counts.get('COURSE_ADVISOR', 0)
            transcript_advisor_count = day_counts.get('TRANSCRIPT_ADVISOR', 0)
            email_sent_count = day_counts.get('EMAIL_SENT', 0) + day_counts.get('email_sent', 0)
            
            # Handle legacy academic_advisor events
            academic_advisor_legacy = day_counts.get('academic_advisor', 0)
            # Split legacy events (rough estimate - can't determine type from daily count)
            if academic_advisor_legacy > 0:
                course_advisor_count += academic_advisor_legacy // 2
                transcript_advisor_count += academic_advisor_legacy - (academic_advisor_legacy // 2)
            
            days.append({
                "date": day_start.strftime("%a"),
                "full_date": day_start.strftime("%Y-%m-%d"),
                "ai_chat": day_counts.get('ai_chat', 0),
                "obe_verification": day_counts.get('obe_verification', 0),
                "obe_rewrite": day_counts.get('obe_rewrite', 0),
                "quiz_generated": day_counts.get('quiz_generated', 0),
                "study_session": day_counts.get('study_session', 0),
                "gpa_calculated": gpa_count,
                "course_advisor": course_advisor_count,
                "transcript_advisor": transcript_advisor_count,
                "email_sent": email_sent_count,
                "total": len(day_events)
            })
        
        return days
    
    def _get_feature_distribution(self, events: List[Dict]) -> List[Dict]:
        """Get feature usage distribution."""
        action_counts = defaultdict(int)
        
        for event in events:
            action = event.get('action', 'unknown')
            action_counts[action] += 1
        
        # Map to readable names
        action_names = {
            'ai_chat': 'AI Chat',
            'obe_verification': 'OBE Verification',
            'obe_rewrite': 'OBE Question Rewrite',
            'quiz_generated': 'Quiz Generated',
            'study_session': 'Study Session',
            'GPA_CALCULATED': 'GPA/CGPA Calculator',
            'CGPA_CALCULATED': 'GPA/CGPA Calculator',
            'COURSE_ADVISOR': 'Semester Course Advisor',
            'TRANSCRIPT_ADVISOR': 'Transcript Advisor',
            'EMAIL_SENT': 'Email Sent',
            'voice_assistant_call': 'Voice Assistant Call',
            # Legacy support
            'academic_advisor': 'Academic Advisor',
            'email_sent': 'Email Sent'
        }
        
        distribution = []
        for action, count in action_counts.items():
            distribution.append({
                "name": action_names.get(action, action.replace('_', ' ').title()),
                "value": count
            })
        
        # Sort by value descending
        distribution.sort(key=lambda x: x['value'], reverse=True)
        return distribution
    
    def _get_learning_trends(self, events: List[Dict]) -> List[Dict]:
        """Get learning trends (quiz + study) over the last 7 days."""
        now = datetime.now()
        days = []
        
        for i in range(6, -1, -1):  # Last 7 days
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            day_events = [
                e for e in events
                if day_start.isoformat() <= e.get('timestamp', '') < day_end.isoformat()
            ]
            
            quiz_count = sum(1 for e in day_events if e.get('action') == 'quiz_generated')
            study_count = sum(1 for e in day_events if e.get('action') == 'study_session')
            
            days.append({
                "date": day_start.strftime("%a"),
                "full_date": day_start.strftime("%Y-%m-%d"),
                "quiz": quiz_count,
                "study": study_count,
                "total": quiz_count + study_count
            })
        
        return days



