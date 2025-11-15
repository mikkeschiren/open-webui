from sqlalchemy import Column, String, Integer, Float, Boolean, Text, JSON, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from open_webui.internal.db import Base, get_db
import json
import uuid

class TokenUsage(Base):
    __tablename__ = "token_usage"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(BigInteger, nullable=False)  # Unix timestamp
    user_id = Column(String, index=True)
    user_email = Column(String)
    model = Column(String, nullable=False, index=True)
    provider = Column(String, index=True)

    # Token counts
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)

    # Cost information
    prompt_cost = Column(Float)
    completion_cost = Column(Float)
    total_cost = Column(Float)
    currency = Column(String, default="USD")

    # Request information
    request_id = Column(String, unique=True)
    endpoint = Column(String)
    streaming = Column(Boolean, default=False)
    token_source = Column(String, default="api")  # "api" or "estimated"

    # Additional metadata as JSON
    extra_metadata = Column(JSON)

    # For tracking conversations
    conversation_id = Column(String, index=True)
    question_text = Column(Text)
    response_text = Column(Text)

class TokenUsageTable:
    """Helper class for token usage database operations"""

    @classmethod
    def insert(cls, db, entry_data: dict) -> bool:
        """Insert a token usage entry into the database"""
        try:
            # Create the entry
            entry = TokenUsage(
                id=entry_data.get("id"),
                timestamp=entry_data.get("timestamp"),
                user_id=entry_data.get("user_id"),
                user_email=entry_data.get("user_email"),
                model=entry_data.get("model"),
                provider=entry_data.get("provider"),
                prompt_tokens=entry_data.get("prompt_tokens"),
                completion_tokens=entry_data.get("completion_tokens"),
                total_tokens=entry_data.get("total_tokens"),
                prompt_cost=entry_data.get("estimated_cost", {}).get("prompt_cost") if entry_data.get("estimated_cost") else None,
                completion_cost=entry_data.get("estimated_cost", {}).get("completion_cost") if entry_data.get("estimated_cost") else None,
                total_cost=entry_data.get("estimated_cost", {}).get("total_cost") if entry_data.get("estimated_cost") else None,
                currency=entry_data.get("estimated_cost", {}).get("currency", "USD") if entry_data.get("estimated_cost") else "USD",
                request_id=entry_data.get("request_id"),
                endpoint=entry_data.get("endpoint"),
                streaming=entry_data.get("streaming", False),
                token_source=entry_data.get("token_source", "api"),
                extra_metadata=entry_data.get("metadata"),
                conversation_id=entry_data.get("conversation_id"),
                question_text=entry_data.get("question_text")[:500] if entry_data.get("question_text") else None,
                response_text=entry_data.get("response_text")[:500] if entry_data.get("response_text") else None,
            )

            db.add(entry)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise e

    @classmethod
    def get_user_usage(cls, db, user_id: str, start_timestamp: int = None, end_timestamp: int = None):
        """Get token usage for a specific user"""
        query = db.query(TokenUsage).filter(TokenUsage.user_id == user_id)

        if start_timestamp:
            query = query.filter(TokenUsage.timestamp >= start_timestamp)
        if end_timestamp:
            query = query.filter(TokenUsage.timestamp <= end_timestamp)

        return query.all()

    @classmethod
    def get_usage_summary(cls, db, user_id: str = None, start_timestamp: int = None, end_timestamp: int = None):
        """Get aggregated usage summary"""
        from sqlalchemy import func

        query = db.query(
            TokenUsage.model,
            TokenUsage.provider,
            func.count(TokenUsage.id).label("request_count"),
            func.sum(TokenUsage.prompt_tokens).label("total_prompt_tokens"),
            func.sum(TokenUsage.completion_tokens).label("total_completion_tokens"),
            func.sum(TokenUsage.total_tokens).label("total_tokens"),
            func.sum(TokenUsage.total_cost).label("total_cost"),
        )

        if user_id:
            query = query.filter(TokenUsage.user_id == user_id)
        if start_timestamp:
            query = query.filter(TokenUsage.timestamp >= start_timestamp)
        if end_timestamp:
            query = query.filter(TokenUsage.timestamp <= end_timestamp)

        return query.group_by(TokenUsage.model, TokenUsage.provider).all()
