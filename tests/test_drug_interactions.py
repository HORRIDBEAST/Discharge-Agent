"""
Tests for the drug interaction lookup tool.
"""
import pytest
import asyncio
from app.tools.drug_interaction_tool import DrugInteractionTool


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestDrugInteractionTool:
    def setup_method(self):
        self.tool = DrugInteractionTool()

    def test_no_interaction_single_drug(self):
        result = run(self.tool._execute(medications=["metformin"]))
        assert result["total_alerts"] == 0
        assert result["has_major_interaction"] is False

    def test_major_interaction_warfarin_aspirin(self):
        result = run(self.tool._execute(medications=["warfarin", "aspirin"]))
        assert result["has_major_interaction"] is True
        assert result["total_alerts"] >= 1
        major = [a for a in result["alerts"] if a["severity"] == "MAJOR"]
        assert len(major) >= 1

    def test_major_interaction_warfarin_ibuprofen(self):
        result = run(self.tool._execute(medications=["warfarin 3mg", "ibuprofen 400mg"]))
        assert result["has_major_interaction"] is True

    def test_moderate_interaction_clopidogrel_omeprazole(self):
        result = run(self.tool._execute(medications=["clopidogrel", "omeprazole"]))
        assert result["total_alerts"] >= 1
        moderate = [a for a in result["alerts"] if a["severity"] == "MODERATE"]
        assert len(moderate) >= 1

    def test_no_interaction_safe_combination(self):
        result = run(self.tool._execute(medications=["metformin", "lisinopril", "atorvastatin"]))
        assert result["has_major_interaction"] is False

    def test_digoxin_amiodarone_major(self):
        result = run(self.tool._execute(medications=["digoxin", "amiodarone"]))
        assert result["has_major_interaction"] is True

    def test_empty_list(self):
        result = run(self.tool._execute(medications=[]))
        assert result["total_alerts"] == 0

    def test_medications_checked_returned(self):
        meds = ["aspirin", "warfarin"]
        result = run(self.tool._execute(medications=meds))
        assert result["medications_checked"] == meds
