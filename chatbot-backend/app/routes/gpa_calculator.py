"""
GPA/CGPA Calculator endpoints
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
from decimal import Decimal, ROUND_HALF_UP

router = APIRouter(prefix="/gpa", tags=["GPA Calculator"])

# Analytics service will be injected from main app
analytics_service = None

def set_analytics_service(service):
    """Set analytics service from main app."""
    global analytics_service
    analytics_service = service


class CourseGrade(BaseModel):
    """Model for a single course grade entry."""
    course: str = Field(..., description="Course name")
    credit_hours: int = Field(..., ge=1, le=4, description="Credit hours (1-4)")
    grade_point: float = Field(..., ge=0.0, le=4.0, description="Grade point (0.0-4.0)")
    
    @validator('grade_point')
    def validate_grade_point(cls, v):
        """Validate grade point is within valid range."""
        if not isinstance(v, (int, float)):
            raise ValueError("Grade point must be a number")
        if v < 0.0 or v > 4.0:
            raise ValueError("Grade point must be between 0.0 and 4.0")
        return float(v)


class GPACalculationRequest(BaseModel):
    """Request model for GPA calculation."""
    courses: List[CourseGrade] = Field(..., min_items=1, description="List of courses with grades")
    
    @validator('courses')
    def validate_courses_not_empty(cls, v):
        """Validate at least one course is provided."""
        if not v or len(v) == 0:
            raise ValueError("At least one course is required")
        return v


class CGPACalculationRequest(BaseModel):
    """Request model for CGPA calculation."""
    gpas: List[float] = Field(..., min_items=1, description="List of semester GPAs")
    
    @validator('gpas')
    def validate_gpas(cls, v):
        """Validate GPAs are within valid range."""
        if not v or len(v) == 0:
            raise ValueError("At least one semester GPA is required")
        for gpa in v:
            if not isinstance(gpa, (int, float)):
                raise ValueError("All GPAs must be numbers")
            if gpa < 0.0 or gpa > 4.0:
                raise ValueError("Each GPA must be between 0.0 and 4.0")
        return [float(gpa) for gpa in v]


@router.post("/calculate-gpa")
async def calculate_gpa(request: GPACalculationRequest):
    """
    Calculate Semester GPA from course grades.
    
    Formula: GPA = (Σ (credit_hours × grade_point)) / (Σ credit_hours)
    
    Args:
        request: List of courses with credit hours and grade points
        
    Returns:
        Calculated GPA rounded to 2 decimal places
    """
    try:
        if not request.courses or len(request.courses) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one course is required for GPA calculation"
            )
        
        total_quality_points = 0.0
        total_credit_hours = 0
        
        for course in request.courses:
            quality_points = course.credit_hours * course.grade_point
            total_quality_points += quality_points
            total_credit_hours += course.credit_hours
        
        if total_credit_hours == 0:
            raise HTTPException(
                status_code=400,
                detail="Total credit hours cannot be zero"
            )
        
        # Calculate GPA with proper rounding
        gpa = total_quality_points / total_credit_hours
        gpa_rounded = round(gpa, 2)
        
        result = {
            "success": True,
            "gpa": gpa_rounded,
            "total_credit_hours": total_credit_hours,
            "total_quality_points": round(total_quality_points, 2),
            "courses_count": len(request.courses)
        }
        
        # Track event
        if analytics_service:
            try:
                analytics_service.track_event(
                    user="anonymous",  # Can be enhanced with session_id
                    action="GPA_CALCULATED",
                    meta={"courses_count": len(request.courses), "gpa": gpa_rounded}
                )
            except Exception as e:
                print(f"Error tracking GPA calculation: {e}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate GPA: {str(e)}"
        )


@router.post("/calculate-cgpa")
async def calculate_cgpa(request: CGPACalculationRequest):
    """
    Calculate Overall CGPA from semester GPAs.
    
    Formula: CGPA = (Σ GPA) / (number_of_semesters)
    
    Args:
        request: List of semester GPAs
        
    Returns:
        Calculated CGPA rounded to 2 decimal places
    """
    try:
        if not request.gpas or len(request.gpas) == 0:
            raise HTTPException(
                status_code=400,
                detail="At least one semester GPA is required for CGPA calculation"
            )
        
        total_gpa = sum(request.gpas)
        semesters = len(request.gpas)
        
        # Calculate CGPA
        cgpa = total_gpa / semesters
        cgpa_rounded = round(cgpa, 2)
        
        result = {
            "success": True,
            "cgpa": cgpa_rounded,
            "total_gpa_sum": round(total_gpa, 2),
            "semesters_count": semesters,
            "semester_gpas": request.gpas
        }
        
        # Track event
        if analytics_service:
            try:
                analytics_service.track_event(
                    user="anonymous",  # Can be enhanced with session_id
                    action="CGPA_CALCULATED",
                    meta={"semesters_count": semesters, "cgpa": cgpa_rounded}
                )
            except Exception as e:
                print(f"Error tracking CGPA calculation: {e}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate CGPA: {str(e)}"
        )

