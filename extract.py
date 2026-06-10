#!/usr/bin/env python3
"""
Shannon Transcripts Knowledge Extractor

Processes Brian Shannon's trading transcripts using a local LLM (Ollama)
to extract and consolidate technical analysis techniques.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import ollama
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def load_prompt_template(template_path: str) -> str:
    """Load the extraction prompt template."""
    path = Path(__file__).parent / template_path
    with open(path, "r") as f:
        return f.read()


def get_transcript_files(transcripts_dir: str) -> list[Path]:
    """Get all transcript files from directory."""
    path = Path(transcripts_dir).expanduser()
    if not path.exists():
        console.print(f"[red]Error: Transcripts directory not found: {path}[/red]")
        sys.exit(1)

    # Support common transcript formats
    files = []
    for ext in ["*.txt", "*.md", "*.vtt", "*.srt"]:
        files.extend(path.glob(ext))

    return sorted(files)


def clean_transcript(text: str) -> str:
    """Clean transcript text, removing timestamps and formatting."""
    # Remove VTT/SRT timestamps (e.g., "00:00:00.000 --> 00:00:05.000")
    text = re.sub(r'\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}', '', text)
    # Remove simple timestamps (e.g., "[00:00:00]" or "0:00")
    text = re.sub(r'\[?\d{1,2}:\d{2}(?::\d{2})?\]?', '', text)
    # Remove VTT header
    text = re.sub(r'^WEBVTT\s*\n', '', text)
    # Remove sequence numbers (common in SRT)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Clean up multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def chunk_transcript(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split long transcript into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near the chunk boundary
            for i in range(min(200, end - start)):
                if text[end - i] in '.!?':
                    end = end - i + 1
                    break

        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def extract_from_transcript(
    transcript: str,
    prompt_template: str,
    model: str,
    temperature: float
) -> dict | None:
    """Send transcript to Ollama and extract techniques."""
    prompt = prompt_template.format(transcript=transcript)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )

        content = response["message"]["content"]

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON directly
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                json_str = json_match.group(0)
            else:
                console.print("[yellow]Warning: Could not find JSON in response[/yellow]")
                return None

        return json.loads(json_str)

    except json.JSONDecodeError as e:
        console.print(f"[yellow]Warning: Invalid JSON response: {e}[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error calling Ollama: {e}[/red]")
        return None


def merge_extractions(extractions: list[dict]) -> dict:
    """Merge multiple extraction results (from chunks) into one."""
    merged = {
        "features": [],
        "tickers": [],
        "key_quotes": [],
        "feature_combinations": [],
        "macro_events": []
    }

    seen_features = set()
    seen_tickers = set()
    seen_quotes = set()

    for ext in extractions:
        if not ext:
            continue

        for feature in ext.get("features", []):
            key = (feature.get("name") or "").lower()
            if key and key not in seen_features:
                seen_features.add(key)
                merged["features"].append(feature)

        for ticker in ext.get("tickers", []):
            symbol = (ticker.get("symbol") or "").upper()
            if symbol and symbol not in seen_tickers:
                seen_tickers.add(symbol)
                merged["tickers"].append(ticker)

        for quote in ext.get("key_quotes", []):
            if quote and quote not in seen_quotes:
                seen_quotes.add(quote)
                merged["key_quotes"].append(quote)

        merged["feature_combinations"].extend(ext.get("feature_combinations", []))
        merged["macro_events"].extend(ext.get("macro_events", []))

    return merged


def save_raw_extraction(extraction: dict, transcript_name: str, output_dir: Path):
    """Save raw extraction to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{transcript_name}.json"
    with open(output_file, "w") as f:
        json.dump(extraction, f, indent=2)


def load_all_raw_extractions(raw_dir: Path) -> list[dict]:
    """Load all raw extraction files."""
    extractions = []
    for json_file in raw_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            data["_source_file"] = json_file.stem
            extractions.append(data)
    return extractions


def consolidate_extractions(extractions: list[dict]) -> dict:
    """Consolidate all extractions into master database with frequency analysis."""
    feature_counts = defaultdict(lambda: {"count": 0, "usages": [], "category": ""})
    ticker_counts = defaultdict(lambda: {"count": 0, "contexts": []})
    all_quotes = []
    combo_counts = defaultdict(lambda: {"count": 0, "usages": []})
    macro_counts = defaultdict(lambda: {"count": 0, "contexts": []})

    # Co-occurrence tracking: which features appear together
    feature_cooccurrence = defaultdict(lambda: defaultdict(int))

    total_transcripts = len(extractions)

    for ext in extractions:
        # Track features in this transcript for co-occurrence
        transcript_features = set()

        for feature in ext.get("features", []):
            name = feature.get("name", "").strip()
            if not name:
                continue
            name_lower = name.lower()
            transcript_features.add(name_lower)
            feature_counts[name_lower]["count"] += 1
            feature_counts[name_lower]["usages"].append(feature.get("usage", ""))
            if feature.get("category"):
                feature_counts[name_lower]["category"] = feature["category"]

        # Update co-occurrence matrix
        for f1 in transcript_features:
            for f2 in transcript_features:
                if f1 < f2:  # Only count each pair once
                    feature_cooccurrence[f1][f2] += 1

        for ticker in ext.get("tickers", []):
            symbol = (ticker.get("symbol") or "").upper()
            if symbol:
                ticker_counts[symbol]["count"] += 1
                if ticker.get("context"):
                    ticker_counts[symbol]["contexts"].append(ticker["context"])

        for quote in ext.get("key_quotes", []):
            if quote:
                all_quotes.append(quote)

        for combo in ext.get("feature_combinations", []):
            features = tuple(sorted(f.lower() for f in combo.get("features", [])))
            if features:
                combo_counts[features]["count"] += 1
                if combo.get("usage"):
                    combo_counts[features]["usages"].append(combo["usage"])

        for macro in ext.get("macro_events", []):
            event = macro.get("event", "").strip()
            if event:
                macro_counts[event]["count"] += 1
                if macro.get("context"):
                    macro_counts[event]["contexts"].append(macro["context"])

    return {
        "total_transcripts": total_transcripts,
        "features": dict(feature_counts),
        "tickers": dict(ticker_counts),
        "quotes": all_quotes,
        "feature_combinations": {str(k): v for k, v in combo_counts.items()},
        "macro_events": dict(macro_counts),
        "feature_cooccurrence": {k: dict(v) for k, v in feature_cooccurrence.items()}
    }


def generate_reference_markdown(consolidated: dict, output_path: Path):
    """Generate the final reference.md document."""
    total = consolidated["total_transcripts"]

    # Sort features by frequency
    sorted_features = sorted(
        consolidated["features"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )

    # Sort tickers by frequency
    sorted_tickers = sorted(
        consolidated["tickers"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )

    # Sort macro events by frequency
    sorted_macro = sorted(
        consolidated["macro_events"].items(),
        key=lambda x: x[1]["count"],
        reverse=True
    )

    # Find top co-occurring feature pairs
    cooccurrence = consolidated.get("feature_cooccurrence", {})
    top_pairs = []
    for f1, pairs in cooccurrence.items():
        for f2, count in pairs.items():
            if count >= 5:  # Only include pairs that appear together at least 5 times
                top_pairs.append((f1, f2, count))
    top_pairs.sort(key=lambda x: x[2], reverse=True)

    md = []
    md.append("# Brian Shannon Analysis Toolkit")
    md.append(f"*Extracted from {total} transcripts*")
    md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    md.append("")
    md.append("---")
    md.append("")

    # Features section
    md.append("## Technical Analysis Features (Ranked by Frequency)")
    md.append("")

    for rank, (name, data) in enumerate(sorted_features[:50], 1):
        count = data["count"]
        pct = (count / total) * 100
        category = data.get("category", "Uncategorized")

        md.append(f"### {rank}. {name.title()}")
        md.append(f"**Mentioned in:** {count} transcripts ({pct:.1f}%)")
        md.append(f"**Category:** {category}")
        md.append("")

        # Get unique usage descriptions
        usages = list(set(u for u in data["usages"] if u))[:5]
        if usages:
            md.append("**How Shannon uses it:**")
            for usage in usages:
                md.append(f"- {usage}")
            md.append("")
        md.append("")

    # Tickers section
    md.append("---")
    md.append("")
    md.append("## Tickers Referenced")
    md.append("")
    md.append("| Ticker | Mentions | Top Contexts |")
    md.append("|--------|----------|--------------|")

    for symbol, data in sorted_tickers[:30]:
        count = data["count"]
        contexts = list(set(data["contexts"]))[:2]
        context_str = "; ".join(contexts) if contexts else "-"
        md.append(f"| {symbol} | {count} | {context_str} |")

    md.append("")

    # Macro Events section
    md.append("---")
    md.append("")
    md.append("## Macro Events & Catalysts")
    md.append("")

    for event, data in sorted_macro[:20]:
        count = data["count"]
        md.append(f"### {event}")
        md.append(f"*Mentioned {count} times*")
        contexts = list(set(data["contexts"]))[:3]
        if contexts:
            for ctx in contexts:
                md.append(f"- {ctx}")
        md.append("")

    # Co-occurrence / Common Setups section
    md.append("---")
    md.append("")
    md.append("## Common Feature Combinations (Co-occurrence Analysis)")
    md.append("")
    md.append("Features that frequently appear together in the same transcript:")
    md.append("")

    for f1, f2, count in top_pairs[:20]:
        pct = (count / total) * 100
        md.append(f"- **{f1.title()}** + **{f2.title()}**: {count} transcripts ({pct:.1f}%)")

    md.append("")

    # Sample quotes section
    md.append("---")
    md.append("")
    md.append("## Sample Quotes")
    md.append("")

    # Take a sample of quotes
    quotes = consolidated.get("quotes", [])
    sample_quotes = random.sample(quotes, min(20, len(quotes))) if quotes else []

    for quote in sample_quotes:
        # Truncate long quotes
        if len(quote) > 300:
            quote = quote[:300] + "..."
        md.append(f"> \"{quote}\"")
        md.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(md))


def process_transcripts(
    transcript_files: list[Path],
    config: dict,
    prompt_template: str,
    output_dir: Path,
    mode: str,
    verbose: bool = False
):
    """Process transcripts and extract techniques."""
    model = config["model"]["name"]
    temperature = config["model"]["temperature"]
    chunk_size = config["processing"]["chunk_size"]
    chunk_overlap = config["processing"]["chunk_overlap"]

    raw_dir = output_dir / "raw"
    total = len(transcript_files)

    for i, transcript_file in enumerate(transcript_files, 1):
        # Skip if already processed (for incremental mode)
        existing = raw_dir / f"{transcript_file.stem}.json"
        if mode == "incremental" and existing.exists():
            if verbose:
                console.print(f"[dim][{i}/{total}] Skipping (exists): {transcript_file.name}[/dim]")
            continue

        console.print(f"[cyan][{i}/{total}][/cyan] {transcript_file.name}")

        # Load and clean transcript
        try:
            with open(transcript_file, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            console.print(f"  [yellow]Warning: Could not read: {e}[/yellow]")
            continue

        cleaned = clean_transcript(text)

        if not cleaned or len(cleaned) < 100:
            console.print(f"  [dim]Skipped (too short)[/dim]")
            continue

        # Chunk if needed
        chunks = chunk_transcript(cleaned, chunk_size, chunk_overlap)
        if verbose or len(chunks) > 1:
            console.print(f"  [dim]Chunks: {len(chunks)}[/dim]")

        # Extract from each chunk
        chunk_extractions = []
        for j, chunk in enumerate(chunks, 1):
            extraction = extract_from_transcript(
                chunk, prompt_template, model, temperature
            )
            if extraction:
                chunk_extractions.append(extraction)
                feat_count = len(extraction.get("features", []))
                if verbose:
                    console.print(f"  [dim]Chunk {j}: {feat_count} features[/dim]")

        # Merge chunk results
        if chunk_extractions:
            merged = merge_extractions(chunk_extractions)
            save_raw_extraction(merged, transcript_file.stem, raw_dir)
            feat_count = len(merged.get("features", []))
            ticker_count = len(merged.get("tickers", []))
            console.print(f"  [green]Extracted: {feat_count} features, {ticker_count} tickers[/green]")
        else:
            console.print(f"  [yellow]No extraction results[/yellow]")

    console.print(f"\n[bold green]Processed {total} transcripts[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Extract trading techniques from Brian Shannon transcripts"
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "batch", "incremental"],
        default="sample",
        help="Processing mode: sample (50 random), batch (all), incremental (chronological)"
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        help="Path to transcripts directory (overrides config)"
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Skip extraction, only consolidate existing raw files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output for each transcript"
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Resolve paths
    project_dir = Path(__file__).parent
    transcripts_dir = args.transcripts or config["paths"]["transcripts"]
    output_dir = project_dir / config["paths"]["output_dir"]

    console.print(Panel.fit(
        "[bold cyan]Shannon Transcripts Knowledge Extractor[/bold cyan]",
        border_style="cyan"
    ))

    if not args.consolidate_only:
        # Load prompt template
        prompt_template = load_prompt_template(config["paths"]["prompt_template"])

        # Get transcript files
        transcript_files = get_transcript_files(transcripts_dir)

        if not transcript_files:
            console.print("[red]No transcript files found![/red]")
            sys.exit(1)

        console.print(f"Found [bold]{len(transcript_files)}[/bold] transcript files")

        # Select files based on mode
        if args.mode == "sample":
            sample_size = config["processing"]["sample_size"]
            transcript_files = random.sample(
                transcript_files,
                min(sample_size, len(transcript_files))
            )
            console.print(f"[yellow]Sample mode: processing {len(transcript_files)} random transcripts[/yellow]")
        elif args.mode == "incremental":
            # Sort by name (usually chronological)
            transcript_files = sorted(transcript_files)
            console.print("[yellow]Incremental mode: processing chronologically, skipping existing[/yellow]")
        else:
            console.print("[yellow]Batch mode: processing all transcripts[/yellow]")

        # Process transcripts
        process_transcripts(
            transcript_files,
            config,
            prompt_template,
            output_dir,
            args.mode,
            verbose=args.verbose
        )

    # Consolidate
    console.print("\n[cyan]Consolidating extractions...[/cyan]")
    raw_dir = output_dir / "raw"

    if not raw_dir.exists() or not list(raw_dir.glob("*.json")):
        console.print("[red]No raw extractions found to consolidate![/red]")
        sys.exit(1)

    extractions = load_all_raw_extractions(raw_dir)
    console.print(f"Loaded [bold]{len(extractions)}[/bold] raw extractions")

    consolidated = consolidate_extractions(extractions)

    # Save consolidated data
    consolidated_dir = output_dir / "consolidated"
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    with open(consolidated_dir / "database.json", "w") as f:
        json.dump(consolidated, f, indent=2)

    # Generate markdown reference
    reference_path = project_dir / config["paths"]["reference_file"]
    generate_reference_markdown(consolidated, reference_path)

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Raw extractions: {raw_dir}")
    console.print(f"  Consolidated DB: {consolidated_dir / 'database.json'}")
    console.print(f"  Reference doc:   {reference_path}")

    # Show summary stats
    table = Table(title="Extraction Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Transcripts processed", str(consolidated["total_transcripts"]))
    table.add_row("Unique features", str(len(consolidated["features"])))
    table.add_row("Unique tickers", str(len(consolidated["tickers"])))
    table.add_row("Macro events", str(len(consolidated["macro_events"])))
    table.add_row("Total quotes", str(len(consolidated["quotes"])))

    console.print(table)


if __name__ == "__main__":
    main()
