"""Stage 5: trim to top 100 per query and package as a Codabench submission."""
import json
import subprocess

from .paths import SUBMISSIONS_DIR


def assemble_submission(ranked_lists, config, label="submission"):
    """Trim to top 100 and save as submission."""
    submission = {}
    for qid, docs in ranked_lists.items():
        submission[qid] = docs[:100]

    out_json = SUBMISSIONS_DIR / f"{label}_data.json"
    out_zip = SUBMISSIONS_DIR / f"{label}.zip"

    # Save JSON (must be named submission_data.json inside zip)
    tmp_json = SUBMISSIONS_DIR / "submission_data.json"
    with open(tmp_json, "w") as f:
        json.dump(submission, f)

    # Create zip
    subprocess.run(["zip", "-j", str(out_zip), str(tmp_json)],
                   capture_output=True, check=True)

    # Also keep a copy with the label
    if str(out_json) != str(tmp_json):
        with open(out_json, "w") as f:
            json.dump(submission, f)

    print("\n  Submission saved:")
    print(f"    JSON: {out_json}")
    print(f"    ZIP:  {out_zip}")
    print(f"    Queries: {len(submission)}")
    print(f"    Docs per query: {len(next(iter(submission.values())))}")
    return submission
