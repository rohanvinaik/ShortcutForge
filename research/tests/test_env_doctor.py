#!/usr/bin/env python3
"""Tests for research.scripts.env_doctor — environment health checks."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.scripts.env_doctor import (
    Check,
    HealthReport,
    _version_tuple,
    check_build_system,
    check_interpreter,
    check_lockfile,
    check_python_version_file,
)


class TestVersionTuple(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(_version_tuple("2.10.0"), (2, 10, 0))

    def test_two_part(self):
        self.assertEqual(_version_tuple("1.5"), (1, 5))

    def test_comparison(self):
        self.assertTrue(_version_tuple("2.10.0") > _version_tuple("2.3.1"))
        self.assertTrue(_version_tuple("5.2.2") >= _version_tuple("5.2.0"))


class TestHealthReport(unittest.TestCase):
    def test_empty_is_healthy(self):
        r = HealthReport()
        self.assertTrue(r.healthy)
        self.assertEqual(r.summary()["health"], "healthy")

    def test_warning_is_needs_attention(self):
        r = HealthReport()
        r.add(Check("test", "ok", "fine"))
        r.add(Check("test2", "warning", "hmm"))
        self.assertFalse(r.healthy)
        self.assertFalse(r.has_errors)
        self.assertTrue(r.has_warnings)
        self.assertEqual(r.summary()["health"], "needs_attention")

    def test_error_is_unhealthy(self):
        r = HealthReport()
        r.add(Check("test", "error", "bad"))
        self.assertTrue(r.has_errors)
        self.assertEqual(r.summary()["health"], "unhealthy")

    def test_counts(self):
        r = HealthReport()
        r.add(Check("a", "ok", ""))
        r.add(Check("b", "ok", ""))
        r.add(Check("c", "warning", ""))
        r.add(Check("d", "error", ""))
        s = r.summary()
        self.assertEqual(s["ok"], 2)
        self.assertEqual(s["warnings"], 1)
        self.assertEqual(s["errors"], 1)
        self.assertEqual(s["total"], 4)


class TestCheckBuildSystem(unittest.TestCase):
    def test_present(self):
        """Build system should be detected in our pyproject.toml."""
        r = HealthReport()
        check_build_system(r)
        self.assertEqual(len(r.checks), 1)
        self.assertEqual(r.checks[0].status, "ok")


if __name__ == "__main__":
    unittest.main()
