#!/usr/bin/env python3
"""SVCSTACK Data Processing Pipeline via RLM + Modal Sandbox.

Loads Airtable exports, analyses them through the RLM pipeline
(Sonnet root + Gemini Flash sub-LLM) in a Modal sandbox, and
outputs structured results.

Usage:
    # Run a specific analysis
    python svcstack_pipeline.py --analysis budget-audit
    python svcstack_pipeline.py --analysis service-insights
    python svcstack_pipeline.py --analysis geo-clusters
    python svcstack_pipeline.py --analysis full

    # Dry run (show what would be loaded, no LLM calls)
    python svcstack_pipeline.py --analysis budget-audit --dry-run

    # Custom output directory
    python svcstack_pipeline.py --analysis full --output ./results

Prerequisites:
    - .env with ANTHROPIC_API_KEY and GEMINI_API_KEY
    - Modal authenticated (modal setup)
    - Airtable exports at EXPORTS_DIR
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent
load_dotenv(REPO_ROOT / ".env")

from rlm import RLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPORTS_DIR = Path(os.environ.get(
    "SVCSTACK_EXPORTS_DIR",
    "/home/pmcconville/dev/svcstack/data/airtable/exports/appLkPdvIBdPPvFaz",
))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "svcstack"

# Analysis definitions: which tables to load for each analysis type
ANALYSES = {
    "budget-audit": {
        "description": "Audit campaign budgets: allocation, utilisation, and anomalies",
        "tables": ["Campaigns", "Ad_Sets", "Budget_Constraints", "Budget_Allocations", "Budget_Audit"],
        "query": (
            "Analyse this SVCSTACK campaign budget data. For each campaign:\n"
            "1. Calculate budget utilisation rate (allocated_to_adsets / total_budget)\n"
            "2. Flag campaigns with >80% or <20% utilisation as anomalies\n"
            "3. Identify budget constraints that may be limiting performance\n"
            "4. Recommend reallocation opportunities\n"
            "Return a structured summary with per-campaign findings and overall recommendations."
        ),
    },
    "service-insights": {
        "description": "Analyse service categories: job values, objectives, and market positioning",
        "tables": ["Service_Categories", "Service_Lines", "Client_Details"],
        "query": (
            "Analyse the SVCSTACK service categories and client data:\n"
            "1. Rank service categories by revenue potential (typical_job_value_max)\n"
            "2. Group services by objective type (Calls vs Leads) and analyse the split\n"
            "3. Identify which services are best suited for each client tier\n"
            "4. Suggest service bundling opportunities based on complementary categories\n"
            "Return a structured analysis with ranked services and actionable recommendations."
        ),
    },
    "geo-clusters": {
        "description": "Analyse geographic clusters: budget distribution, priority segments, and coverage gaps",
        "tables": ["Geo_Clusters", "Clusters_Zones", "Regions"],
        "query": (
            "Analyse the SVCSTACK geographic cluster data:\n"
            "1. Summarise cluster distribution by segment_type and commercial_priority\n"
            "2. Calculate total and average budget per cluster\n"
            "3. Identify clusters with available_budget > 50% of total (underutilised)\n"
            "4. Identify high-priority clusters that may need budget increases\n"
            "5. Flag any data quality issues (missing fields, inconsistent values)\n"
            "Return a structured geographic analysis with cluster rankings and recommendations."
        ),
    },
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_tables(table_names: list[str]) -> dict[str, list[dict]]:
    """Load Airtable export JSON files. Skip missing tables with a warning."""
    data = {}
    for name in table_names:
        path = EXPORTS_DIR / f"{name}.json"
        if not path.exists():
            print(f"  WARN: {path} not found, skipping")
            continue
        with open(path, "r") as f:
            records = json.load(f)
        data[name] = records
        print(f"  Loaded {name}: {len(records)} records ({path.stat().st_size:,} bytes)")
    return data


def build_context(data: dict[str, list[dict]]) -> str:
    """Convert loaded tables into a single text context for the RLM."""
    parts = []
    for table_name, records in data.items():
        parts.append(f"## Table: {table_name}")
        parts.append(f"Records: {len(records)}")
        parts.append("")
        # Include full JSON for small tables, summarised for large ones
        table_json = json.dumps(records, indent=2, default=str)
        if len(table_json) > 100_000:
            # For large tables, include first 10 and last 5 records + schema
            fields = list(records[0].get("fields", {}).keys()) if records else []
            parts.append(f"Fields: {fields}")
            parts.append(f"First 10 records:")
            parts.append(json.dumps(records[:10], indent=2, default=str))
            parts.append(f"Last 5 records:")
            parts.append(json.dumps(records[-5:], indent=2, default=str))
        else:
            parts.append(table_json)
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# RLM Execution
# ---------------------------------------------------------------------------

def create_rlm() -> RLM:
    """Create configured RLM instance: Sonnet root + Gemini Flash sub-LLM + Modal."""
    return RLM(
        backend="anthropic",
        backend_kwargs={
            "api_key": os.environ["ANTHROPIC_API_KEY"],
            "model_name": "claude-sonnet-4-20250514",
            "max_tokens": 4096,
        },
        other_backends=["gemini"],
        other_backend_kwargs=[{
            "api_key": os.environ["GEMINI_API_KEY"],
            "model_name": "gemini-2.5-flash",
        }],
        environment="modal",
        environment_kwargs={
            "app_name": "rlm-svcstack-pipeline",
            "timeout": 600,
        },
        max_iterations=5,
        verbose=True,
    )


def run_analysis(analysis_name: str, output_dir: Path, dry_run: bool = False) -> dict:
    """Run a single analysis and return results."""
    config = ANALYSES[analysis_name]
    print(f"\n{'='*60}")
    print(f"Analysis: {analysis_name}")
    print(f"Description: {config['description']}")
    print(f"Tables: {config['tables']}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading tables...")
    data = load_tables(config["tables"])
    if not data:
        print("ERROR: No tables loaded. Check exports directory.")
        return {"error": "No tables loaded"}

    # Build context
    context = build_context(data)
    print(f"\nContext built: {len(context):,} chars")

    if dry_run:
        print("\n[DRY RUN] Would send this context to RLM. Skipping LLM calls.")
        print(f"Query: {config['query'][:200]}...")
        return {"dry_run": True, "context_chars": len(context), "tables": list(data.keys())}

    # Run RLM
    print("\nStarting RLM pipeline...")
    start = time.perf_counter()
    rlm = create_rlm()
    result = rlm.completion(context, root_prompt=config["query"])
    elapsed = time.perf_counter() - start

    # Build output
    output = {
        "analysis": analysis_name,
        "timestamp": datetime.now().isoformat(),
        "tables_loaded": {name: len(records) for name, records in data.items()},
        "context_chars": len(context),
        "execution_time_seconds": round(elapsed, 1),
        "usage": result.usage_summary.to_dict(),
        "result": result.response,
    }

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Also save a readable text version
    text_path = output_path.with_suffix(".txt")
    with open(text_path, "w") as f:
        f.write(f"SVCSTACK Analysis: {analysis_name}\n")
        f.write(f"Date: {output['timestamp']}\n")
        f.write(f"Time: {output['execution_time_seconds']}s\n")
        f.write(f"Tables: {output['tables_loaded']}\n")
        f.write(f"\n{'='*60}\n\n")
        f.write(result.response)

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results saved to:")
    print(f"  JSON: {output_path}")
    print(f"  Text: {text_path}")
    print(f"\nUsage: {json.dumps(result.usage_summary.to_dict(), indent=2)}")

    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SVCSTACK Data Processing Pipeline via RLM + Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available analyses: " + ", ".join(ANALYSES.keys()) + ", full",
    )
    parser.add_argument(
        "--analysis",
        required=True,
        choices=list(ANALYSES.keys()) + ["full"],
        help="Which analysis to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and show context size without making LLM calls",
    )
    args = parser.parse_args()

    # Validate env
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if missing:
        print(f"ERROR: Missing environment variables: {missing}")
        print("Set them in .env file at the repo root.")
        sys.exit(1)

    if not EXPORTS_DIR.exists():
        print(f"ERROR: Exports directory not found: {EXPORTS_DIR}")
        sys.exit(1)

    # Run
    if args.analysis == "full":
        results = {}
        for name in ANALYSES:
            results[name] = run_analysis(name, args.output, args.dry_run)
        print(f"\n{'='*60}")
        print("All analyses complete.")
        for name, result in results.items():
            status = "DRY RUN" if result.get("dry_run") else f"{result.get('execution_time_seconds', '?')}s"
            print(f"  {name}: {status}")
    else:
        run_analysis(args.analysis, args.output, args.dry_run)


if __name__ == "__main__":
    main()
