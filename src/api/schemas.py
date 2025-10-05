"""Pydantic schemas for API requests and responses"""

from pydantic import BaseModel, Field
from typing import List, Optional


class Record(BaseModel):
    """Schema for a single urban zone record with infrastructure metrics"""
    GRAPROES: Optional[float] = None
    GRAPROES_F: Optional[float] = None
    GRAPROES_M: Optional[float] = None
    RECUCALL_C: Optional[float] = None
    RAMPAS_C: Optional[float] = None
    PASOPEAT_C: Optional[float] = None
    BANQUETA_C: Optional[float] = None
    GUARNICI_C: Optional[float] = None
    CICLOVIA_C: Optional[float] = None
    CICLOCAR_C: Optional[float] = None
    ALUMPUB_C: Optional[float] = None
    LETRERO_C: Optional[float] = None
    TELPUB_C: Optional[float] = None
    ARBOLES_C: Optional[float] = None
    DRENAJEP_C: Optional[float] = None
    TRANSCOL_C: Optional[float] = None
    ACESOPER_C: Optional[float] = None
    ACESOAUT_C: Optional[float] = None
    PUESSEMI_C: Optional[float] = None
    PUESAMBU_C: Optional[float] = None


class Payload(BaseModel):
    """Request payload containing multiple zone records"""
    data: List[Record] = Field(..., description="Lista de zonas con métricas")
