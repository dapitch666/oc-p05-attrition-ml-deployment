from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Defined here rather than in api.py, which imports this module.
Environment = Literal["dev", "test", "prod"]


class EmployeeInput(BaseModel):
    """Raw employee fields, as collected by HR. Engineered features are computed server side."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=18, le=70, description="Age in years.", examples=[35])
    revenu_mensuel: int = Field(gt=0, le=100_000, description="Monthly income.", examples=[5000])
    nombre_experiences_precedentes: int = Field(
        ge=0, le=20, description="Number of employers before the current one.", examples=[2]
    )
    annees_experience_totale: int = Field(
        ge=0, le=50, description="Total years of professional experience.", examples=[10]
    )
    annees_dans_l_entreprise: int = Field(
        ge=0,
        le=50,
        description="Years in the company. Cannot exceed the total years of experience.",
        examples=[5],
    )
    annees_dans_le_poste_actuel: int = Field(
        ge=0,
        le=50,
        description="Years in the current role. Cannot exceed the years in the company.",
        examples=[2],
    )
    annees_depuis_la_derniere_promotion: int = Field(
        ge=0,
        le=50,
        description="Years since the last promotion. Cannot exceed the years in the company.",
        examples=[1],
    )
    annees_sous_responsable_actuel: int = Field(
        ge=0,
        le=50,
        description="Years under the current manager. Cannot exceed the years in the company.",
        examples=[2],
    )
    satisfaction_employee_environnement: int = Field(
        ge=1, le=4, description="Satisfaction with the work environment, from 1 to 4.", examples=[3]
    )
    satisfaction_employee_nature_travail: int = Field(
        ge=1,
        le=4,
        description="Satisfaction with the nature of the work, from 1 to 4.",
        examples=[3],
    )
    satisfaction_employee_equipe: int = Field(
        ge=1, le=4, description="Satisfaction with the team, from 1 to 4.", examples=[3]
    )
    satisfaction_employee_equilibre_pro_perso: int = Field(
        ge=1,
        le=4,
        description="Satisfaction with the work-life balance, from 1 to 4.",
        examples=[3],
    )
    note_evaluation_precedente: int = Field(
        ge=1, le=4, description="Previous performance review score, from 1 to 4.", examples=[3]
    )
    niveau_education: int = Field(
        ge=1, le=5, description="Education level, from 1 to 5.", examples=[3]
    )
    augmentation_salaire_precedente: int = Field(
        ge=0, le=100, description="Last salary raise, in whole percent.", examples=[12]
    )
    nombre_participation_pee: int = Field(
        ge=0,
        le=10,
        description="Number of participations in the employee savings plan.",
        examples=[1],
    )
    nb_formations_suivies: int = Field(
        ge=0, le=20, description="Number of trainings followed.", examples=[2]
    )
    distance_domicile_travail: int = Field(
        ge=0, le=200, description="Distance between home and work, in km.", examples=[10]
    )
    frequence_deplacement: Literal["Aucun", "Occasionnel", "Fréquent"] = Field(
        description="Business travel frequency.", examples=["Occasionnel"]
    )
    genre: Literal["F", "M"] = Field(description="Gender.", examples=["F"])
    heures_supplementaires: Literal["Non", "Oui"] = Field(
        description="Whether the employee works overtime.", examples=["Non"]
    )
    statut_marital: Literal["Célibataire", "Divorcé(e)", "Marié(e)"] = Field(
        description="Marital status.", examples=["Marié(e)"]
    )
    poste: Literal[
        "Assistant de Direction",
        "Cadre Commercial",
        "Consultant",
        "Directeur Technique",
        "Manager",
        "Représentant Commercial",
        "Ressources Humaines",
        "Senior Manager",
        "Tech Lead",
    ] = Field(description="Job title.", examples=["Consultant"])
    domaine_etude: Literal[
        "Autre",
        "Entrepreneuriat",
        "Infra & Cloud",
        "Marketing",
        "Ressources Humaines",
        "Transformation Digitale",
    ] = Field(description="Field of study.", examples=["Infra & Cloud"])

    @model_validator(mode="after")
    def check_years_coherence(self) -> Self:
        """Reject rows whose year counts are impossible."""
        if self.annees_dans_l_entreprise > self.annees_experience_totale:
            raise ValueError("annees_dans_l_entreprise cannot exceed annees_experience_totale")
        for name in (
            "annees_dans_le_poste_actuel",
            "annees_depuis_la_derniere_promotion",
            "annees_sous_responsable_actuel",
        ):
            if getattr(self, name) > self.annees_dans_l_entreprise:
                raise ValueError(f"{name} cannot exceed annees_dans_l_entreprise")
        return self


class PredictionOutput(BaseModel):
    """Attrition prediction for one employee."""

    score: float = Field(
        ge=0,
        le=1,
        description="Model score between 0 and 1, to be compared with the alert threshold. "
        "It is not a calibrated probability, because the class weights used in training "
        "shift the predicted values.",
    )
    alert: bool = Field(
        description="True when the score is at or above the alert threshold of the model."
    )
    model_version: str = Field(
        description="Model version from the metadata, followed by 8 hex characters of the "
        "SHA-256 of the pipeline file.",
        examples=["1.0.0-3fa9c2d1"],
    )


class HealthOutput(BaseModel):
    """State of the API."""

    status: str = Field(description="Always 'ok' while the API is running.", examples=["ok"])
    version: str = Field(
        description="Version of the application, from the package metadata.", examples=["0.6.0"]
    )
    environment: Environment = Field(
        description="Environment the API runs in, read from APP_ENV at startup.",
        examples=["prod"],
    )
