"""
check_setup.py — Task 1: Environment readiness check.
Run this before starting training to verify all dependencies and data files.

Usage:
    python check_setup.py
"""

import sys
import os

# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

PASS_MARK = "[PASS]"
FAIL_MARK = "[FAIL]"


def check(label: str, ok: bool, hint: str = "") -> bool:
    """Print a labelled PASS/FAIL line and return the bool result."""
    status = PASS_MARK if ok else FAIL_MARK
    suffix = f"  → {hint}" if (not ok and hint) else ""
    print(f"  {status}  {label}{suffix}")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────────────

def check_sumo_home() -> bool:
    """Check 1 — SUMO_HOME environment variable must be set."""
    if not os.environ.get("SUMO_HOME") or not os.path.exists(os.environ.get("SUMO_HOME", "")):
        # Auto-fix common path
        if os.path.exists(r"C:\Program Files (x86)\Eclipse\Sumo"):
            os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
            
    val = os.environ.get("SUMO_HOME", "")
    ok = bool(val) and os.path.exists(val)
    return check(
        "SUMO_HOME environment variable",
        ok,
        hint=(
            "Set SUMO_HOME to your SUMO installation directory.\n"
            "     Windows: [System Properties → Environment Variables]\n"
            "     Linux/macOS: export SUMO_HOME=/usr/share/sumo\n"
            "     Then re-open terminal and re-run this script."
        ) if not ok else ""
    )


def check_imports() -> bool:
    """Check 2 — All required Python packages must be importable."""
    packages = {
        "sumo_rl": "pip install sumo-rl",
        "gymnasium": "pip install gymnasium",
        "numpy": "pip install numpy",
        "pandas": "pip install pandas",
        "matplotlib": "pip install matplotlib",
        "traci": (
            "traci is bundled with SUMO.\n"
            "       Ensure SUMO is installed and SUMO_HOME/tools is on PYTHONPATH.\n"
            "       Add SUMO_HOME to sys.path inside your scripts, or:\n"
            "       export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH"
        ),
    }

    all_ok = True
    for pkg, install_hint in packages.items():
        try:
            __import__(pkg)
            check(f"  import {pkg}", True)
        except ImportError:
            check(f"  import {pkg}", False, hint=f"Install: {install_hint}")
            all_ok = False
    return all_ok


def check_data_files() -> bool:
    """Check 3 — CN+ dataset files must exist in the data/ folder."""
    required_files = [
        "data/bremen.net.xml",    # road network (copied from CN+_Dataset.net.xml)
        "data/bremen.rou.xml",    # vehicle routes (one of the 49 hourly scenario files)
        "data/bremen.sumocfg",    # SUMO simulation config
        "data/bremen.add.xml",    # additional elements: bus stops (CN+_Dataset.add.xml)
    ]

    all_ok = True
    for rel_path in required_files:
        # Resolve relative to this script's location so it works from any cwd
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, rel_path)
        exists = os.path.isfile(full_path)
        if not exists:
            all_ok = False
        check(
            f"  {rel_path}",
            exists,
            hint=(
                "Download the CN+ dataset from:\n"
                "       https://zenodo.org/records/8189767\n"
                f"       Then place {os.path.basename(rel_path)} into the data/ folder."
            ) if not exists else ""
        )
    return all_ok


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all environment checks and print a final summary."""
    print("=" * 60)
    print("  ATSC-RL Project — Environment Setup Check")
    print("=" * 60)

    results: list[bool] = []

    # ── Check 1: SUMO_HOME ────────────────────────────────────────────────────
    print("\n[1/3] SUMO_HOME")
    results.append(check_sumo_home())

    # ── Check 2: Python imports ───────────────────────────────────────────────
    print("\n[2/3] Python package imports")
    results.append(check_imports())

    # ── Check 3: Data files ───────────────────────────────────────────────────
    print("\n[3/3] CN+ Dataset files in data/")
    results.append(check_data_files())

    # ── Final summary ─────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"  Summary: {passed}/{total} checks passed")

    if passed == total:
        print("  [SUCCESS] All checks passed — ready to train!")
    else:
        print("  [WARNING] Fix the issues above, then re-run check_setup.py.")
    print("=" * 60)

    # Exit with non-zero code so CI/CD pipelines can detect failures
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
