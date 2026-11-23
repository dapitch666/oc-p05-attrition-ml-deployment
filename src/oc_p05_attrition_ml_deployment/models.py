from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from oc_p05_attrition_ml_deployment.db import Base


class PredictionLog(Base):
    """One prediction: the raw inputs received and what the model answered."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model_version: Mapped[str]
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB)
    score: Mapped[float]
    alert: Mapped[bool]
