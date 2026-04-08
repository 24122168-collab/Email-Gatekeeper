"""
package.py  —  Build model.tar.gz for SageMaker
================================================
Run this script once before deploying.
It bundles your code/ folder into the exact archive structure SageMaker expects.

Usage:
    cd "RL Envir"
    python sagemaker/package.py

Output:
    sagemaker/model.tar.gz   ← upload this to S3, then point SageMaker at it

What goes inside model.tar.gz:
    code/
    ├── inference.py     ← SageMaker entry point  (the 4 handlers)
    └── classifier.py    ← your rule-based classifier logic

SageMaker unpacks the archive and looks for code/inference.py automatically
when you use the SKLearn or generic Python containers.
"""

import os
import tarfile

# ── Paths — all relative to this script's location ───────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_TAR  = os.path.join(HERE, "model.tar.gz")

# Files to include — add more here if you create extra helper modules
FILES_TO_PACK = {
    "code/inference.py":  os.path.join(HERE, "inference.py"),
    "code/classifier.py": os.path.join(HERE, "classifier.py"),
}


def build():
    print("\n  Building model.tar.gz ...")
    print(f"  Output → {OUTPUT_TAR}\n")

    # Verify all source files exist before starting
    missing = [src for src in FILES_TO_PACK.values() if not os.path.exists(src)]
    if missing:
        print("  ❌ Missing files:")
        for f in missing:
            print(f"       {f}")
        raise FileNotFoundError("Fix missing files then re-run.")

    # Build the archive
    with tarfile.open(OUTPUT_TAR, "w:gz") as tar:
        for archive_name, source_path in FILES_TO_PACK.items():
            tar.add(source_path, arcname=archive_name)
            size_kb = os.path.getsize(source_path) / 1024
            print(f"  + {archive_name:<30}  ({size_kb:.1f} KB)")

    # Verify and report
    tar_size_kb = os.path.getsize(OUTPUT_TAR) / 1024
    print(f"\n  ✅ Done!  model.tar.gz = {tar_size_kb:.1f} KB")

    # Show contents as confirmation
    print("\n  Contents of model.tar.gz:")
    with tarfile.open(OUTPUT_TAR, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"     {member.name:<35}  {member.size / 1024:.1f} KB")

    print(f"\n  Next step — upload to S3:")
    print(f"  aws s3 cp {OUTPUT_TAR} s3://YOUR-BUCKET/email-gatekeeper/model.tar.gz\n")


if __name__ == "__main__":
    build()
