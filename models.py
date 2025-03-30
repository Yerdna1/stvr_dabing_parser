"""
Pydantic models for the Screenplay Parser App
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Union, Any

# Don't import from pydantic_ai directly, we'll use standard Pydantic models
# and use pydantic_ai functions separately

class Speaker(BaseModel):
    """Model for character speakers"""
    name: str = Field(..., description="The character's name, exactly as it appears in the screenplay")
    audio_notation: Optional[str] = Field(None, description="Audio notations like VO (Voice Over), MO (Monologue), OS (Off Screen), etc.")

class BaseSegment(BaseModel):
    """Base model for all segment types"""
    type: Optional[str] = Field("text", description="The type of segment (dialogue, scene_header, segment_marker, text)")
    timecode: Optional[str] = Field(None, description="Timestamp in format like '00:05:44' or segment marker like '00:05:44-----'")
    text: Optional[str] = Field("", description="The content text of this segment")
    
    @field_validator('timecode')
    def validate_timecode(cls, v):
        # If timecode is None, return it
        if v is None:
            return v
        # Make sure timecode is a string
        if not isinstance(v, str):
            raise ValueError(f"Timecode must be a string, got {type(v)}")
        return v

class DialogueSegment(BaseSegment):
    """Model for dialogue segments"""
    type: str = Field("dialogue", description="This segment contains character dialogue")
    speaker: str = Field(..., description="Name of the speaking character, including any audio notations like (VO)")
    text: str = Field(..., description="The dialogue text spoken by the character")
    
    @field_validator('speaker')
    def validate_speaker(cls, v):
        if not isinstance(v, str):
            raise ValueError(f"Speaker must be a string, got {type(v)}")
        return v

class SceneHeaderSegment(BaseSegment):
    """Model for scene header segments"""
    type: str = Field("scene_header", description="This segment indicates a new scene")
    scene_type: Optional[str] = Field(None, description="INT or EXT indicating interior or exterior scene")
    location: Optional[str] = Field(None, description="Location where the scene takes place")

class SegmentMarker(BaseSegment):
    """Model for segment markers"""
    type: str = Field("segment_marker", description="This segment marks a major division in the screenplay")
    segment_number: Optional[int] = Field(None, description="Sequential number for this segment")

class Entities(BaseModel):
    """Model for screenplay entities"""
    characters: List[str] = Field([], description="List of all unique character names in the screenplay")
    locations: List[str] = Field([], description="List of all unique locations in the screenplay")
    audio_notations: Dict[str, str] = Field({}, description="Dictionary mapping audio notation abbreviations to their meanings")

class ProcessedSegment(BaseModel):
    """Generic segment model to handle any type of segment data"""
    type: Optional[str] = Field("", description="Type of segment (dialogue, scene_header, segment_marker, text)") # Default to empty string
    timecode: Optional[str] = Field("", description="Timestamp or segment marker with dashes") # Default to empty string
    speaker: Optional[str] = Field("", description="Character speaking (for dialogue)") # Default to empty string
    text: Optional[str] = Field("", description="Content text") # Default to empty string
    segment_number: Optional[int] = Field(None, description="Number for segment markers") # Keep None for optional int
    scene_type: Optional[str] = Field("", description="INT or EXT (for scene headers)") # Default to empty string
    location: Optional[str] = Field(None, description="Location (for scene headers)")
    
    class Config:
        extra = "allow"  # Allow extra fields

class ScreenplayResult(BaseModel):
    """Model for the complete screenplay analysis result"""
    segments: List[ProcessedSegment] = Field([], description="List of all processed segments in the screenplay")
    entities: Entities = Field(default_factory=Entities, description="Entity information extracted from the screenplay")

# Removed standalone validator as defaults are handled in Fields now
