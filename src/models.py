# src/models.py
from pydantic import BaseModel, Field, validator
from datetime import datetime
from fastapi import HTTPException

class FashionItem(BaseModel):
    gender: str = Field(..., description="Gender of the item")
    masterCategory: str = Field(..., description="Master category of the item")
    subCategory: str = Field(..., description="Subcategory of the item")
    articleType: str = Field(..., description="Article type of the item")
    baseColour: str = Field(..., description="Base color of the item")
    season: str = Field(..., description="Season of the item")
    year: int = Field(..., description="Year of the item")

    @validator("gender")
    def validate_gender(cls, v):
        VALID_GENDERS = ["men", "women", "boys", "girls", "unisex"]
        v_lower = v.lower()
        if v_lower not in VALID_GENDERS:
            raise ValueError(f"Gender must be one of {VALID_GENDERS}")
        return v_lower.capitalize()

    @validator("masterCategory")
    def validate_master_category(cls, v):
        VALID_MASTER_CATEGORIES = ["apparel", "accessories", "footwear", "personal care", "free items"]
        v_lower = v.lower()
        if v_lower not in VALID_MASTER_CATEGORIES:
            raise ValueError(f"MasterCategory must be one of {VALID_MASTER_CATEGORIES}")
        return v_lower.capitalize()

    @validator("subCategory")
    def validate_sub_category(cls, v):
        if not v.strip():
            raise ValueError("subCategory cannot be empty")
        return v

    @validator("articleType")
    def validate_article_type(cls, v):
        if not v.strip():
            raise ValueError("articleType cannot be empty")
        return v

    @validator("baseColour")
    def validate_base_colour(cls, v):
        if not v.strip():
            raise ValueError("baseColour cannot be empty")
        return v

    @validator("season")
    def validate_season(cls, v):
        VALID_SEASONS = ["summer", "winter", "spring", "fall"]
        v_lower = v.lower()
        if v_lower not in VALID_SEASONS:
            raise ValueError(f"Season must be one of {VALID_SEASONS}")
        return v_lower.capitalize()

    @validator("year")
    def validate_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year:
            raise ValueError(f"Year must be between 1900 and {current_year}")
        return v

    @classmethod
    def validate(cls, values):
        try:
            return super().validate(values)
        except ValueError as e:
            error = e.errors()[0]
            raise HTTPException(
                status_code=422,
                detail={
                    "type": error["type"],
                    "loc": error["loc"],
                    "msg": error["msg"],
                    "input": error["input"]
                }
            )