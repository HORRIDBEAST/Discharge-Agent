"""
Drug Interaction Lookup Tool — mocked.

In production this would call a clinical database (e.g., DrFirst, Multum).
Here we maintain a small hardcoded interaction table that covers common
dangerous pairs. The mock is realistic enough to demonstrate the agent's
escalation behaviour.

Interactions are severity-rated:
  - MAJOR: Must escalate, never suppress
  - MODERATE: Flag for review
  - MINOR: Informational only
"""
from __future__ import annotations
from app.tools.base import BaseTool, register_tool
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Mock interaction database — (drug_a, drug_b) -> {severity, description}
# Keys normalised to lowercase
INTERACTION_DB: list[dict] = [
    {
        "drugs": {"warfarin", "aspirin"},
        "severity": "MAJOR",
        "description": "Increased bleeding risk. Combined anticoagulant + antiplatelet effect.",
    },
    {
        "drugs": {"warfarin", "ibuprofen"},
        "severity": "MAJOR",
        "description": "Increased bleeding risk and GI haemorrhage.",
    },
    {
        "drugs": {"metformin", "contrast dye"},
        "severity": "MAJOR",
        "description": "Risk of lactic acidosis. Hold metformin 48h before contrast.",
    },
    {
        "drugs": {"lisinopril", "potassium"},
        "severity": "MODERATE",
        "description": "Risk of hyperkalaemia.",
    },
    {
        "drugs": {"metoprolol", "verapamil"},
        "severity": "MAJOR",
        "description": "Additive AV node depression — risk of bradycardia and heart block.",
    },
    {
        "drugs": {"clopidogrel", "omeprazole"},
        "severity": "MODERATE",
        "description": "CYP2C19 inhibition may reduce clopidogrel efficacy.",
    },
    {
        "drugs": {"ssri", "tramadol"},
        "severity": "MAJOR",
        "description": "Serotonin syndrome risk.",
    },
    {
        "drugs": {"amiodarone", "warfarin"},
        "severity": "MAJOR",
        "description": "Significant potentiation of anticoagulant effect.",
    },
    {
        "drugs": {"furosemide", "gentamicin"},
        "severity": "MODERATE",
        "description": "Additive ototoxicity risk.",
    },
    {
        "drugs": {"digoxin", "amiodarone"},
        "severity": "MAJOR",
        "description": "Amiodarone increases digoxin levels — risk of toxicity.",
    },
]


@register_tool
class DrugInteractionTool(BaseTool):
    name = "drug_interaction_lookup"
    description = "Check a list of medications for clinically significant drug-drug interactions."

    async def _execute(self, medications: list[str]) -> dict:
        """
        Check all pairs in the medication list against the interaction database.
        Returns list of alerts (empty if none found).
        """
        normalised = [m.lower().strip() for m in medications]
        alerts = []

        for entry in INTERACTION_DB:
            db_drugs = entry["drugs"]
            # Check if any two meds in the list match this interaction pair
            matches = [n for n in normalised if any(d in n or n in d for d in db_drugs)]
            if len(matches) >= 2:
                alerts.append({
                    "severity": entry["severity"],
                    "description": entry["description"],
                    "involved_medications": matches,
                })

        logger.info(
            "drug_interaction_check",
            medication_count=len(medications),
            alerts_found=len(alerts),
            major_alerts=sum(1 for a in alerts if a["severity"] == "MAJOR"),
        )

        return {
            "medications_checked": medications,
            "alerts": alerts,
            "has_major_interaction": any(a["severity"] == "MAJOR" for a in alerts),
            "total_alerts": len(alerts),
        }
