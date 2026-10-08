"""Optional publishing helper for a user's authenticated GitHub CLI.

Creates a new private repo by default and pushes the local main branch.
Requires gh, git and an already authenticated gh session. Does not overwrite
an existing repository, force push, read credentials or publish private EEG.
"""

import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Publish this research repository using your authenticated GitHub CLI")
    parser.add_argument("--name", default="psi-qeeg-bis")
    parser.add_argument("--public", action="store_true", help="Explicitly choose public visibility (default: private)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for program in ("git", "gh"):
        if shutil.which(program) is None:
            raise SystemExit(f"Install {program} before running this publishing helper")
    subprocess.run(["gh", "auth", "status"], check=True, cwd=root)
    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-b", "main"], check=True, cwd=root)
        subprocess.run(["git", "add", "."], check=True, cwd=root)
        subprocess.run(["git", "commit", "-m", "Add PSI qEEG/BIS research pipeline and real EEG example"], check=True, cwd=root)
    status = subprocess.run(["git", "status", "--porcelain"], check=True, cwd=root, capture_output=True, text=True)
    if status.stdout.strip():
        raise SystemExit("Review and commit local changes before publishing")
    subprocess.run(["gh", "repo", "create", args.name, "--public" if args.public else "--private",
                    "--description", "PSI qEEG research with real BIS EEG, electrode coverage and reproducible provenance",
                    "--source", str(root), "--push"], check=True, cwd=root)


if __name__ == "__main__":
    main()

