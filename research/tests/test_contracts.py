#!/usr/bin/env python3
"""Tests for research.src.contracts — dataclass serialization roundtrips."""

import json
import sys
import unittest
from pathlib import Path

# Allow importing research.src from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.src.contracts import (
    Constraint,
    CoverageReport,
    Entity,
    GateDecision,
    NegativeBankEntry,
    OODPrompt,
    SemanticFrame,
    Tier2Block,
    Tier3Slot,
    TypedIRExample,
)


class TestEntity(unittest.TestCase):
    def test_roundtrip(self):
        e = Entity(name="timer_duration", entity_type="duration", value="5 minutes")
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        self.assertEqual(e, e2)

    def test_json_stable(self):
        e = Entity(name="a", entity_type="b", value="c")
        s = json.dumps(e.to_dict(), sort_keys=True)
        e2 = Entity.from_dict(json.loads(s))
        self.assertEqual(e, e2)


class TestSemanticFrame(unittest.TestCase):
    def test_roundtrip(self):
        frame = SemanticFrame(
            domain="timer",
            primary_intent="set_timer",
            entities=[Entity("dur", "duration", "5m")],
            constraints=[Constraint("max_actions", {"limit": "10"})],
            estimated_complexity="simple",
        )
        d = frame.to_dict()
        frame2 = SemanticFrame.from_dict(d)
        self.assertEqual(frame.domain, frame2.domain)
        self.assertEqual(len(frame2.entities), 1)
        self.assertEqual(len(frame2.constraints), 1)


class TestTier2Block(unittest.TestCase):
    def test_roundtrip_without_ids(self):
        b = Tier2Block(
            action_index=0,
            action_name="is.workflow.actions.delay",
            tokens=["PARAM", "WFDelayTime", "5"],
        )
        d = b.to_dict()
        self.assertNotIn("token_ids", d)  # Not persisted
        b2 = Tier2Block.from_dict(d)
        self.assertEqual(b.tokens, b2.tokens)
        self.assertIsNone(b2.token_ids)

    def test_token_ids_not_serialized(self):
        b = Tier2Block(action_index=0, action_name="test", tokens=["A"], token_ids=[42])
        d = b.to_dict()
        self.assertNotIn("token_ids", d)


class TestTier3Slot(unittest.TestCase):
    def test_roundtrip(self):
        s = Tier3Slot(
            slot_id="s1",
            value_kind="string",
            value="Hello World",
            source_param="WFMessage",
        )
        d = s.to_dict()
        s2 = Tier3Slot.from_dict(d)
        self.assertEqual(s.value, s2.value)
        self.assertEqual(s.source_param, s2.source_param)

    def test_optional_source_param(self):
        s = Tier3Slot(slot_id="s1", value_kind="number", value=42)
        d = s.to_dict()
        self.assertNotIn("source_param", d)
        s2 = Tier3Slot.from_dict(d)
        self.assertIsNone(s2.source_param)


class TestTypedIRExample(unittest.TestCase):
    def _make_example(self):
        return TypedIRExample(
            shortcut_id="test-001",
            system_prompt="You are a shortcut generator.",
            prompt="Set a 5 minute timer",
            dsl='SHORTCUT "Timer"\nACTION is.workflow.actions.delay\nENDACTION\nENDSHORTCUT',
            shortcut_name="Timer",
            tier1_tokens=["SHORTCUT", "ACTION", "ENDACTION", "ENDSHORTCUT"],
            tier2_blocks=[
                Tier2Block(
                    0, "is.workflow.actions.delay", ["PARAM", "WFDelayTime", "300"]
                )
            ],
            tier3_slots=[Tier3Slot("s1", "string", "Timer")],
            metadata={"domain": "timer", "action_count": 1},
        )

    def test_roundtrip(self):
        ex = self._make_example()
        d = ex.to_dict()
        ex2 = TypedIRExample.from_dict(d)
        self.assertEqual(ex.shortcut_id, ex2.shortcut_id)
        self.assertEqual(len(ex2.tier2_blocks), 1)
        self.assertEqual(len(ex2.tier3_slots), 1)

    def test_json_roundtrip(self):
        ex = self._make_example()
        s = json.dumps(ex.to_dict())
        ex2 = TypedIRExample.from_dict(json.loads(s))
        self.assertEqual(ex.prompt, ex2.prompt)


class TestNegativeBankEntry(unittest.TestCase):
    def test_roundtrip(self):
        pos = TypedIRExample(
            shortcut_id="t1",
            system_prompt="",
            prompt="test",
            dsl="DSL",
            shortcut_name="Test",
            tier1_tokens=["A"],
            tier2_blocks=[],
            tier3_slots=[],
        )
        entry = NegativeBankEntry(
            prompt="test",
            shortcut_id="t1",
            positive=pos,
            negative=None,
            error_tags=["hallucinated_action"],
            source="distillation",
            lint_changes=[{"type": "action_repair", "confidence": 0.9}],
        )
        d = entry.to_dict()
        entry2 = NegativeBankEntry.from_dict(d)
        self.assertIsNone(entry2.negative)
        self.assertEqual(entry2.error_tags, ["hallucinated_action"])


class TestOODPrompt(unittest.TestCase):
    def test_roundtrip(self):
        p = OODPrompt("Write me a poem", "ood", "creative_writing", "synthetic")
        d = p.to_dict()
        p2 = OODPrompt.from_dict(d)
        self.assertEqual(p, p2)


class TestGateDecision(unittest.TestCase):
    def test_roundtrip(self):
        g = GateDecision(in_domain=True, confidence=0.95)
        d = g.to_dict()
        g2 = GateDecision.from_dict(d)
        self.assertEqual(g, g2)


class TestCoverageReport(unittest.TestCase):
    def test_roundtrip(self):
        r = CoverageReport(
            scope="tier1",
            dataset="eval",
            total_tokens_in_eval=500,
            covered=495,
            uncovered=["UNKNOWN_TOKEN_A", "UNKNOWN_TOKEN_B"],
            coverage_pct=99.0,
        )
        d = r.to_dict()
        r2 = CoverageReport.from_dict(d)
        self.assertEqual(r2.coverage_pct, 99.0)
        self.assertEqual(r2.uncovered, ["UNKNOWN_TOKEN_A", "UNKNOWN_TOKEN_B"])

    def test_uncovered_sorted(self):
        r = CoverageReport.from_dict(
            {
                "scope": "tier2",
                "dataset": "eval",
                "total_tokens_in_eval": 100,
                "covered": 90,
                "uncovered": ["z_token", "a_token"],
                "coverage_pct": 90.0,
            }
        )
        self.assertEqual(r.uncovered, ["a_token", "z_token"])


def run_tests():
    unittest.main(module=__name__, exit=False, verbosity=2)


if __name__ == "__main__":
    unittest.main()
