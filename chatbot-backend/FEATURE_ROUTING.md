# Feature Routing Implementation

## Overview

The AI assistant now intelligently detects when users are asking about built-in features and redirects them to use those features instead of providing generic instructions or templates.

## How It Works

### 1. Feature Detection (`app/utils/feature_router.py`)

The `FeatureRouter` class uses keyword pattern matching to detect user intent:

- **Email Features**: Detects queries about sending emails, drafting emails, email templates
- **Course Advisor**: Detects queries about transcripts, course planning, prerequisites, semester planning
- **OBE Verification**: Detects queries about Bloom's taxonomy, OBE, question verification
- **Learning Interface**: Detects queries about quizzes, study help, Socratic mode

### 2. Agent Instructions Update

The main agent (`AgentService`) now includes:
- Priority rules for feature routing
- Instructions to guide users to features instead of giving generic responses
- Clear guidance on when to redirect vs. when to answer directly

### 3. Automatic Routing

When a user asks about a feature:
1. The system detects the intent using `FeatureRouter.detect_feature()`
2. If a feature is detected (and no document context exists), feature guidance is injected
3. The agent uses this guidance to redirect the user appropriately

## Features Covered

### Email Feature
**Triggers**: "send email", "email to", "draft email", "email template", etc.
**Response**: Guides users to the **Send Email** button in the chat interface

### Course Advisor
**Triggers**: "transcript", "course planning", "prerequisites", "semester plan", etc.
**Response**: Guides users to the **Course Advisor** tab for transcript analysis

### OBE Verification
**Triggers**: "Bloom's taxonomy", "OBE", "question verification", "exam paper", etc.
**Response**: Guides users to the **OBE Verification** tab

### Learning Interface
**Triggers**: "quiz", "study help", "Socratic mode", "learning materials", etc.
**Response**: Guides users to the **Learning** tab

## Key Behaviors

1. **Document Context Priority**: If user has uploaded documents, feature routing is disabled (document analysis takes priority)

2. **Smart Detection**: Uses regex patterns to match various phrasings of the same intent

3. **Maintainable**: Easy to add new features by:
   - Adding keywords to `FeatureRouter` class
   - Adding feature guide in `FEATURE_GUIDES` dictionary
   - No changes needed to agent logic

## Example Interactions

### Before (Generic Response)
**User**: "How can I send an email to student affairs?"
**AI**: *[Provides long email template with instructions]*

### After (Feature Routing)
**User**: "How can I send an email to student affairs?"
**AI**: "I can help you send emails! IQRAi has a built-in **Send Email** feature... [guides to feature]"

## Adding New Features

To add a new feature:

1. Add keywords to `FeatureRouter`:
```python
NEW_FEATURE_KEYWORDS = [
    r'\bkeyword1\b', r'\bkeyword2\b'
]
```

2. Add feature guide:
```python
FEATURE_GUIDES = {
    'new_feature': {
        'name': 'New Feature',
        'guidance': 'Guidance message here...'
    }
}
```

3. Update detection method to include new feature

That's it! No changes needed to agent service.

## Testing

Test queries:
- "How do I send an email to student affairs?" → Should route to Email feature
- "What courses should I take next semester?" → Should route to Course Advisor
- "How do I verify my exam questions?" → Should route to OBE Verification
- "Can you help me study?" → Should route to Learning Interface

