#!/usr/bin/env python3
"""
Environment health checker for Balanced Sashimi research pipeline.

Validates that the project environment is correctly configured before
any research scripts run. Designed to be the first gate in the pipeline.

Usage:
    uv run python research/scripts/env_doctor.py --strict
    uv run python research/scripts/env_doctor.py --json
    uv run python research/scripts/env_doctor.py --check-imports --check-mps
"""

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Version constraints for research dependencies ---
REQUIRED_IMPORTS = {
    "torch": "2.5.0",
    "sentence_transformers": "5.2.0",
    "numpy": "2.0.0",
    "sklearn": "1.5.0",
    "numba": "0.62.0",
    "scipy": "1.12.0",
    "accelerate": "0.20.0",
    "lark": "1.1.0",
}


@dataclass
class Check:
    name: str
    status: str  # "ok", "warning", "error"
    message: str
    suggestion: str | None = None
    evidence: dict = field(default_factory=dict)


@dataclass
class HealthReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return all(c.status == "ok" for c in self.checks)

    @property
    def has_errors(self) -> bool:
        return any(c.status == "error" for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.status == "warning" for c in self.checks)

    def summary(self) -> dict:
        ok = sum(1 for c in self.checks if c.status == "ok")
        warnings = sum(1 for c in self.checks if c.status == "warning")
        errors = sum(1 for c in self.checks if c.status == "error")
        if errors > 0:
            health = "unhealthy"
        elif warnings > 0:
            health = "needs_attention"
        else:
            health = "healthy"
        return {
            "health": health,
            "ok": ok,
            "warnings": warnings,
            "errors": errors,
            "total": len(self.checks),
        }

    def add(self, check: Check) -> None:
        self.checks.append(check)


def _version_tuple(v: str) -> tuple[int, ...]:
    """Parse version string to comparable tuple."""
    parts = []
    for p in v.split(".")[:3]:
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_interpreter(report: HealthReport) -> None:
    """Verify we're running inside the project .venv."""
    # Use unresolved path first — .venv/bin/python is a symlink to the
    # base interpreter, but we still want to detect that we're running
    # through the venv.
    exe_raw = Path(sys.executable)
    exe_resolved = exe_raw.resolve()
    venv_dir = PROJECT_ROOT / ".venv"

    in_venv = venv_dir.exists() and (
        str(exe_raw).startswith(str(venv_dir))
        or str(exe_resolved).startswith(str(venv_dir))
        # Also check sys.prefix which is set by venv activation
        or str(Path(sys.prefix).resolve()).startswith(str(venv_dir.resolve()))
    )

    if in_venv:
        report.add(
            Check(
                name="interpreter",
                status="ok",
                message="Running inside project .venv",
                evidence={"executable": str(exe_raw), "resolved": str(exe_resolved)},
            )
        )
    elif venv_dir.exists():
        report.add(
            Check(
                name="interpreter",
                status="error",
                message=f"Running outside project .venv: {exe_raw}",
                suggestion=f"Run with: uv run python {' '.join(sys.argv)}",
                evidence={"executable": str(exe_raw), "expected_venv": str(venv_dir)},
            )
        )
    else:
        report.add(
            Check(
                name="interpreter",
                status="error",
                message="No .venv found in project root",
                suggestion="Run: uv venv .venv && uv pip install -e '.[research]'",
                evidence={"project_root": str(PROJECT_ROOT)},
            )
        )


def check_python_version_file(report: HealthReport) -> None:
    """Verify .python-version exists and matches runtime."""
    pv_file = PROJECT_ROOT / ".python-version"
    if not pv_file.exists():
        report.add(
            Check(
                name="python_version_file",
                status="warning",
                message=".python-version file not found",
                suggestion="Create .python-version with your target version (e.g., '3.11.8')",
            )
        )
        return

    pinned = pv_file.read_text().strip()
    runtime = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    if runtime.startswith(pinned.rstrip(".")) or pinned == runtime:
        report.add(
            Check(
                name="python_version_file",
                status="ok",
                message=f".python-version={pinned}, runtime={runtime}",
                evidence={"pinned": pinned, "runtime": runtime},
            )
        )
    else:
        report.add(
            Check(
                name="python_version_file",
                status="warning",
                message=f".python-version={pinned} but runtime={runtime}",
                evidence={"pinned": pinned, "runtime": runtime},
            )
        )


def check_lockfile(report: HealthReport) -> None:
    """Verify uv.lock exists."""
    lock = PROJECT_ROOT / "uv.lock"
    manifest = PROJECT_ROOT / "pyproject.toml"

    if not manifest.exists():
        report.add(
            Check(
                name="lockfile",
                status="ok",
                message="No pyproject.toml found (not a Python project)",
            )
        )
        return

    if not lock.exists():
        report.add(
            Check(
                name="lockfile",
                status="error",
                message="uv.lock not found",
                suggestion="Run: uv lock",
            )
        )
        return

    # Check freshness
    manifest_mtime = manifest.stat().st_mtime
    lock_mtime = lock.stat().st_mtime

    if manifest_mtime > lock_mtime:
        staleness = manifest_mtime - lock_mtime
        report.add(
            Check(
                name="lockfile",
                status="warning",
                message=f"uv.lock is stale (pyproject.toml modified {staleness:.0f}s after lock)",
                suggestion="Run: uv lock",
                evidence={"staleness_seconds": round(staleness, 1)},
            )
        )
    else:
        report.add(
            Check(
                name="lockfile",
                status="ok",
                message="uv.lock exists and is fresh",
            )
        )


def check_lock_sync(report: HealthReport) -> None:
    """Run uv lock --check to verify lockfile is in sync."""
    lock = PROJECT_ROOT / "uv.lock"
    if not lock.exists():
        return  # Already reported by check_lockfile

    try:
        result = subprocess.run(
            ["uv", "lock", "--check"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            report.add(
                Check(
                    name="lock_sync",
                    status="ok",
                    message="uv lock --check passed",
                )
            )
        else:
            report.add(
                Check(
                    name="lock_sync",
                    status="warning",
                    message="uv lock --check failed (lockfile out of sync)",
                    suggestion="Run: uv lock",
                    evidence={"stderr": result.stderr.strip()[:200]},
                )
            )
    except FileNotFoundError:
        report.add(
            Check(
                name="lock_sync",
                status="warning",
                message="uv not found in PATH",
                suggestion="Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh",
            )
        )
    except subprocess.TimeoutExpired:
        report.add(
            Check(
                name="lock_sync",
                status="warning",
                message="uv lock --check timed out",
            )
        )


def check_imports(report: HealthReport) -> None:
    """Verify all research-critical imports succeed with minimum versions."""
    for module_name, min_version in REQUIRED_IMPORTS.items():
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")

            if version != "unknown" and _version_tuple(version) < _version_tuple(min_version):
                report.add(
                    Check(
                        name=f"import_{module_name}",
                        status="warning",
                        message=f"{module_name}=={version} (need >={min_version})",
                        evidence={"version": version, "minimum": min_version},
                    )
                )
            else:
                report.add(
                    Check(
                        name=f"import_{module_name}",
                        status="ok",
                        message=f"{module_name}=={version}",
                        evidence={"version": version},
                    )
                )
        except ImportError as e:
            report.add(
                Check(
                    name=f"import_{module_name}",
                    status="error",
                    message=f"Cannot import {module_name}: {e}",
                    suggestion="Run: uv pip install -e '.[research]'",
                )
            )
        except ValueError as e:
            # NumPy ABI mismatch raises ValueError
            report.add(
                Check(
                    name=f"import_{module_name}",
                    status="error",
                    message=f"ABI error importing {module_name}: {e}",
                    suggestion="Rebuild binary packages: uv pip install --force-reinstall scikit-learn numba",
                )
            )
        except Exception as e:
            report.add(
                Check(
                    name=f"import_{module_name}",
                    status="error",
                    message=f"Unexpected import failure for {module_name}: {type(e).__name__}: {e}",
                    suggestion="Reinstall research deps: uv pip install -e '.[research]'",
                )
            )


def check_numpy_abi(report: HealthReport) -> None:
    """Check for NumPy ABI compatibility (the specific failure we had before)."""
    try:
        import numpy as np

        # Try importing sklearn which triggers the ABI check
        from sklearn.utils import murmurhash  # noqa: F401 — import triggers ABI check

        report.add(
            Check(
                name="numpy_abi",
                status="ok",
                message="NumPy ABI compatible with scikit-learn",
                evidence={"numpy_version": np.__version__},
            )
        )
    except (ValueError, ImportError) as e:
        report.add(
            Check(
                name="numpy_abi",
                status="error",
                message=f"NumPy ABI incompatibility: {e}",
                suggestion="Run: uv pip install --force-reinstall numpy scikit-learn",
            )
        )


def check_mps(report: HealthReport) -> None:
    """Verify MPS (Metal Performance Shaders) availability."""
    try:
        import torch

        if not torch.backends.mps.is_built():
            report.add(
                Check(
                    name="mps",
                    status="error",
                    message="PyTorch not built with MPS support",
                    suggestion="Install PyTorch with MPS: pip install torch>=2.5",
                )
            )
        elif not torch.backends.mps.is_available():
            report.add(
                Check(
                    name="mps",
                    status="warning",
                    message="MPS built but not available (no Apple Silicon GPU?)",
                )
            )
        else:
            # Quick smoke test
            x = torch.randn(2, 2, device="mps")
            y = x @ x.T
            has_nan = torch.isnan(y).any().item()
            report.add(
                Check(
                    name="mps",
                    status="ok" if not has_nan else "warning",
                    message="MPS available and functional"
                    if not has_nan
                    else "MPS available but NaN in smoke test",
                    evidence={"torch_version": torch.__version__},
                )
            )
    except ImportError:
        report.add(
            Check(
                name="mps",
                status="error",
                message="Cannot import torch",
                suggestion="Run: uv pip install -e '.[research]'",
            )
        )


def check_numba_jit(report: HealthReport) -> None:
    """Verify Numba JIT compilation works."""
    try:
        import numpy as np
        from numba import njit

        @njit
        def _test_fn(x):
            return x + 1

        result = _test_fn(np.array([1.0, 2.0]))
        if np.allclose(result, [2.0, 3.0]):
            report.add(
                Check(
                    name="numba_jit",
                    status="ok",
                    message="Numba JIT compilation works",
                )
            )
        else:
            report.add(
                Check(
                    name="numba_jit",
                    status="warning",
                    message="Numba JIT produced unexpected results",
                )
            )
    except Exception as e:
        report.add(
            Check(
                name="numba_jit",
                status="error",
                message=f"Numba JIT failed: {e}",
                suggestion="Check numba/llvmlite/numpy version compatibility",
            )
        )


def check_build_system(report: HealthReport) -> None:
    """Verify pyproject.toml has [build-system]."""
    toml_path = PROJECT_ROOT / "pyproject.toml"
    if not toml_path.exists():
        return

    content = toml_path.read_text()
    if "[build-system]" in content:
        report.add(
            Check(
                name="build_system",
                status="ok",
                message="[build-system] present in pyproject.toml",
            )
        )
    else:
        report.add(
            Check(
                name="build_system",
                status="warning",
                message="No [build-system] in pyproject.toml",
                suggestion="Add [build-system] block with hatchling",
            )
        )


def run_all_checks(
    check_imports_flag: bool = True,
    check_mps_flag: bool = True,
    check_lock_flag: bool = True,
) -> HealthReport:
    """Run all environment checks and return report."""
    report = HealthReport()

    # Always run these
    check_interpreter(report)
    check_python_version_file(report)
    check_lockfile(report)
    check_build_system(report)

    if check_lock_flag:
        check_lock_sync(report)

    if check_imports_flag:
        check_imports(report)
        check_numpy_abi(report)
        check_numba_jit(report)

    if check_mps_flag:
        check_mps(report)

    return report


def print_report(report: HealthReport) -> None:
    """Pretty-print the health report."""
    summary = report.summary()

    status_icons = {"ok": "\u2705", "warning": "\u26a0\ufe0f ", "error": "\u274c"}

    print(f"\n{'=' * 60}")
    print(" Balanced Sashimi Environment Doctor")
    print(f" Health: {summary['health'].upper()}")
    print(f" ({summary['ok']} ok, {summary['warnings']} warnings, {summary['errors']} errors)")
    print(f"{'=' * 60}\n")

    for check in report.checks:
        icon = status_icons.get(check.status, "?")
        print(f"  {icon} {check.name}: {check.message}")
        if check.suggestion:
            print(f"     \u2192 {check.suggestion}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Environment health checker for Balanced Sashimi research pipeline",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any error")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    parser.add_argument(
        "--check-lock",
        action="store_true",
        default=True,
        help="Run uv lock --check (default: true)",
    )
    parser.add_argument(
        "--no-check-lock",
        dest="check_lock",
        action="store_false",
        help="Skip uv lock --check",
    )
    parser.add_argument(
        "--check-imports",
        action="store_true",
        default=True,
        help="Check research imports (default: true)",
    )
    parser.add_argument(
        "--no-check-imports",
        dest="check_imports",
        action="store_false",
        help="Skip import checks",
    )
    parser.add_argument(
        "--check-mps",
        action="store_true",
        default=True,
        help="Check MPS availability (default: true)",
    )
    parser.add_argument(
        "--no-check-mps", dest="check_mps", action="store_false", help="Skip MPS check"
    )
    args = parser.parse_args()

    report = run_all_checks(
        check_imports_flag=args.check_imports,
        check_mps_flag=args.check_mps,
        check_lock_flag=args.check_lock,
    )

    if args.json:
        output = {
            "project": str(PROJECT_ROOT),
            "summary": report.summary(),
            "checks": [asdict(c) for c in report.checks],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report)

    if args.strict and report.has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
