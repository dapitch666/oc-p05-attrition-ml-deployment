from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, SmallInteger, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from oc_p05_attrition_ml_deployment.db import Base


class EmployeeFields:
    """The raw employee fields, one typed column per EmployeeInput field."""

    age: Mapped[int] = mapped_column(SmallInteger)
    revenu_mensuel: Mapped[int]
    nombre_experiences_precedentes: Mapped[int] = mapped_column(SmallInteger)
    annees_experience_totale: Mapped[int] = mapped_column(SmallInteger)
    annees_dans_l_entreprise: Mapped[int] = mapped_column(SmallInteger)
    annees_dans_le_poste_actuel: Mapped[int] = mapped_column(SmallInteger)
    annees_depuis_la_derniere_promotion: Mapped[int] = mapped_column(SmallInteger)
    annees_sous_responsable_actuel: Mapped[int] = mapped_column(SmallInteger)
    satisfaction_employee_environnement: Mapped[int] = mapped_column(SmallInteger)
    satisfaction_employee_nature_travail: Mapped[int] = mapped_column(SmallInteger)
    satisfaction_employee_equipe: Mapped[int] = mapped_column(SmallInteger)
    satisfaction_employee_equilibre_pro_perso: Mapped[int] = mapped_column(SmallInteger)
    note_evaluation_precedente: Mapped[int] = mapped_column(SmallInteger)
    niveau_education: Mapped[int] = mapped_column(SmallInteger)
    augmentation_salaire_precedente: Mapped[int] = mapped_column(SmallInteger)
    nombre_participation_pee: Mapped[int] = mapped_column(SmallInteger)
    nb_formations_suivies: Mapped[int] = mapped_column(SmallInteger)
    distance_domicile_travail: Mapped[int] = mapped_column(SmallInteger)
    frequence_deplacement: Mapped[str]
    genre: Mapped[str]
    heures_supplementaires: Mapped[str]
    statut_marital: Mapped[str]
    poste: Mapped[str]
    domaine_etude: Mapped[str]


class ApiKey(Base):
    """A client key: only its SHA-256 hash is stored, never the key."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_name: Mapped[str]
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelInput(EmployeeFields, Base):
    """One call to the model: the raw employee fields received."""

    __tablename__ = "model_inputs"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Nullable: the rows logged before API keys existed have no key.
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"))
    output: Mapped["ModelOutput | None"] = relationship(back_populates="input")


class ModelOutput(Base):
    """What the model answered for one input; absent when the call failed after the input."""

    __tablename__ = "model_outputs"

    id: Mapped[int] = mapped_column(primary_key=True)
    input_id: Mapped[int] = mapped_column(ForeignKey("model_inputs.id"), unique=True)
    score: Mapped[float]
    alert: Mapped[bool]
    model_version: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    input: Mapped[ModelInput] = relationship(back_populates="output")


class ApiError(Base):
    """A refused call, answered with a 401 or a 422; it holds nothing the client typed."""

    __tablename__ = "api_errors"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    method: Mapped[str]
    path: Mapped[str]
    status_code: Mapped[int] = mapped_column(SmallInteger)
    # Nullable: a 401 has no key, and neither has a 422 raised before the key is checked.
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"))
    detail: Mapped[Any] = mapped_column(JSONB)


class Employee(EmployeeFields, Base):
    """One employee of the project 4 dataset, with the outcome observed."""

    __tablename__ = "employees"

    id_employee: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    departement: Mapped[str]
    niveau_hierarchique_poste: Mapped[int] = mapped_column(SmallInteger)
    note_evaluation_actuelle: Mapped[int] = mapped_column(SmallInteger)
    a_quitte_l_entreprise: Mapped[bool]
