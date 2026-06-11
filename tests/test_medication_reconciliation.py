"""
Tests for the medication reconciliation tool.
"""
import pytest
import asyncio
from app.tools.medication_reconciliation_tool import MedicationReconciliationTool
from app.models.medications import MedChangeType


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestMedicationReconciliation:
    def setup_method(self):
        self.tool = MedicationReconciliationTool()

    def _med(self, name, dose=None, freq=None, indication=None, source="test.pdf"):
        return {
            "name": name, "dose": dose, "frequency": freq,
            "route": None, "indication": indication,
            "source_document": source, "raw_text": name,
        }

    def test_unchanged_medication(self):
        admit = [self._med("Aspirin", dose="81mg", freq="daily")]
        discharge = [self._med("Aspirin", dose="81mg", freq="daily")]
        result = run(self.tool._execute(admit, discharge))
        changes = result["changes"]
        assert any(c["change_type"] == MedChangeType.UNCHANGED.value for c in changes)

    def test_discontinued_medication_flagged(self):
        admit = [self._med("Ibuprofen", dose="400mg", freq="PRN")]
        discharge = []  # Discontinued
        result = run(self.tool._execute(admit, discharge))
        changes = result["changes"]
        assert len(changes) == 1
        assert changes[0]["change_type"] == MedChangeType.DISCONTINUED.value
        assert changes[0]["flag_for_reconciliation"] is True
        assert len(result["unresolved_flags"]) == 1

    def test_added_medication_without_indication_flagged(self):
        admit = []
        discharge = [self._med("Clopidogrel", dose="75mg", freq="daily", indication=None)]
        result = run(self.tool._execute(admit, discharge))
        changes = result["changes"]
        assert changes[0]["change_type"] == MedChangeType.ADDED.value
        assert changes[0]["flag_for_reconciliation"] is True

    def test_added_medication_with_indication_not_flagged(self):
        admit = []
        discharge = [self._med("Clopidogrel", dose="75mg", freq="daily", indication="Post-PCI antiplatelet")]
        result = run(self.tool._execute(admit, discharge))
        changes = result["changes"]
        assert changes[0]["change_type"] == MedChangeType.ADDED.value
        assert changes[0]["flag_for_reconciliation"] is False

    def test_dose_change_without_reason_flagged(self):
        admit = [self._med("Atorvastatin", dose="40mg", freq="nightly")]
        discharge = [self._med("Atorvastatin", dose="80mg", freq="nightly")]
        result = run(self.tool._execute(admit, discharge))
        changes = [c for c in result["changes"] if c["medication_name"].lower() == "atorvastatin"]
        assert changes[0]["change_type"] == MedChangeType.DOSE_CHANGED.value
        assert changes[0]["flag_for_reconciliation"] is True

    def test_reconciliation_complete_when_all_documented(self):
        admit = [self._med("Metformin", dose="1000mg", freq="twice daily")]
        discharge = [self._med("Metformin", dose="1000mg", freq="twice daily")]
        result = run(self.tool._execute(admit, discharge))
        assert result["reconciliation_complete"] is True
        assert len(result["unresolved_flags"]) == 0

    def test_multiple_changes_mixed(self):
        admit = [
            self._med("Aspirin", dose="81mg", freq="daily"),
            self._med("Warfarin", dose="5mg", freq="daily"),
        ]
        discharge = [
            self._med("Aspirin", dose="81mg", freq="daily"),
            self._med("Warfarin", dose="3mg", freq="daily"),  # Dose change
            self._med("Bisoprolol", dose="2.5mg", freq="daily"),  # Added
        ]
        result = run(self.tool._execute(admit, discharge))
        change_types = {c["change_type"] for c in result["changes"]}
        assert MedChangeType.UNCHANGED.value in change_types
        assert MedChangeType.ADDED.value in change_types
        # Warfarin dose change should be flagged
        assert len(result["unresolved_flags"]) >= 1
