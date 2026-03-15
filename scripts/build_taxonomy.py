#!/usr/bin/env python3
"""
Build genome → taxonomy mapping from FASTA headers.

Parses all .fna files in the genomes directory to extract:
  - Species (binomial name)
  - Genus (first word of species)
  - Approximate phylum (from known genus→phylum lookup)

Output: CSV with columns [genome, species, genus, phylum]

Usage:
    python scripts/build_taxonomy.py \
        --genomes-dir data/genomes \
        --output data/processed/genome_taxonomy.csv
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ── Known genus → phylum mapping (covers major bacterial phyla) ──────────
# This is a best-effort lookup for visualization. Not exhaustive.
# Genera not found here get phylum="Unknown".
GENUS_TO_PHYLUM = {
    # Pseudomonadota (formerly Proteobacteria)
    "Escherichia": "Pseudomonadota", "Salmonella": "Pseudomonadota",
    "Klebsiella": "Pseudomonadota", "Pseudomonas": "Pseudomonadota",
    "Vibrio": "Pseudomonadota", "Helicobacter": "Pseudomonadota",
    "Campylobacter": "Pseudomonadota", "Neisseria": "Pseudomonadota",
    "Bordetella": "Pseudomonadota", "Legionella": "Pseudomonadota",
    "Brucella": "Pseudomonadota", "Rickettsia": "Pseudomonadota",
    "Acinetobacter": "Pseudomonadota", "Burkholderia": "Pseudomonadota",
    "Rhizobium": "Pseudomonadota", "Bradyrhizobium": "Pseudomonadota",
    "Agrobacterium": "Pseudomonadota", "Caulobacter": "Pseudomonadota",
    "Sinorhizobium": "Pseudomonadota", "Xanthomonas": "Pseudomonadota",
    "Stenotrophomonas": "Pseudomonadota", "Ralstonia": "Pseudomonadota",
    "Wolbachia": "Pseudomonadota", "Bartonella": "Pseudomonadota",
    "Sphingomonas": "Pseudomonadota", "Erwinia": "Pseudomonadota",
    "Serratia": "Pseudomonadota", "Proteus": "Pseudomonadota",
    "Enterobacter": "Pseudomonadota", "Citrobacter": "Pseudomonadota",
    "Shigella": "Pseudomonadota", "Yersinia": "Pseudomonadota",
    "Haemophilus": "Pseudomonadota", "Pasteurella": "Pseudomonadota",
    "Moraxella": "Pseudomonadota", "Azospirillum": "Pseudomonadota",
    "Methylobacterium": "Pseudomonadota", "Acidovorax": "Pseudomonadota",
    # Bacillota (formerly Firmicutes)
    "Bacillus": "Bacillota", "Staphylococcus": "Bacillota",
    "Streptococcus": "Bacillota", "Enterococcus": "Bacillota",
    "Clostridium": "Bacillota", "Clostridioides": "Bacillota",
    "Listeria": "Bacillota", "Lactobacillus": "Bacillota",
    "Lactococcus": "Bacillota", "Leuconostoc": "Bacillota",
    "Mycoplasma": "Bacillota", "Ureaplasma": "Bacillota",
    "Anaerococcus": "Bacillota", "Peptostreptococcus": "Bacillota",
    "Paenibacillus": "Bacillota", "Lacticaseibacillus": "Bacillota",
    "Limosilactobacillus": "Bacillota", "Ligilactobacillus": "Bacillota",
    "Lactiplantibacillus": "Bacillota", "Weissella": "Bacillota",
    # Actinomycetota (formerly Actinobacteria)
    "Mycobacterium": "Actinomycetota", "Corynebacterium": "Actinomycetota",
    "Streptomyces": "Actinomycetota", "Bifidobacterium": "Actinomycetota",
    "Propionibacterium": "Actinomycetota", "Cutibacterium": "Actinomycetota",
    "Nocardia": "Actinomycetota", "Rhodococcus": "Actinomycetota",
    "Gordonia": "Actinomycetota", "Frankia": "Actinomycetota",
    "Micrococcus": "Actinomycetota", "Arthrobacter": "Actinomycetota",
    "Brevibacterium": "Actinomycetota", "Actinomyces": "Actinomycetota",
    # Bacteroidota (formerly Bacteroidetes)
    "Bacteroides": "Bacteroidota", "Prevotella": "Bacteroidota",
    "Porphyromonas": "Bacteroidota", "Flavobacterium": "Bacteroidota",
    "Chryseobacterium": "Bacteroidota", "Sphingobacterium": "Bacteroidota",
    "Cytophaga": "Bacteroidota", "Elizabethkingia": "Bacteroidota",
    # Cyanobacteriota
    "Synechocystis": "Cyanobacteriota", "Synechococcus": "Cyanobacteriota",
    "Prochlorococcus": "Cyanobacteriota", "Nostoc": "Cyanobacteriota",
    "Anabaena": "Cyanobacteriota", "Microcystis": "Cyanobacteriota",
    # Spirochaetota
    "Treponema": "Spirochaetota", "Borrelia": "Spirochaetota",
    "Leptospira": "Spirochaetota", "Brachyspira": "Spirochaetota",
    # Deinococcota
    "Deinococcus": "Deinococcota", "Thermus": "Deinococcota",
    # Fusobacteriota
    "Fusobacterium": "Fusobacteriota",
    # Chlamydiota
    "Chlamydia": "Chlamydiota", "Chlamydophila": "Chlamydiota",
    # Verrucomicrobiota
    "Akkermansia": "Verrucomicrobiota",
    # Aquificota
    "Aquifex": "Aquificota",
    # Thermotogota
    "Thermotoga": "Thermotogota",
}


def parse_species_from_header(header: str) -> str:
    """Extract species binomial from FASTA header line."""
    # Remove the > and accession
    # Format: >NZ_CP120703.1 Bradyrhizobium centrosematis strain LMG 29515 ...
    header = header.strip().lstrip(">")

    # Split on first space to remove accession
    parts = header.split(None, 1)
    if len(parts) < 2:
        return "Unknown"

    desc = parts[1]

    # Remove common suffixes
    for suffix in [
        " strain ", " str. ", " chromosome", " complete genome",
        " plasmid", " contig", " scaffold", " whole genome",
        " shotgun", " draft", ", complete", " genomic",
        " assembly", " isolate "
    ]:
        idx = desc.lower().find(suffix.lower())
        if idx > 0:
            desc = desc[:idx]

    # Clean up
    desc = desc.strip().rstrip(",").strip()

    # Take first two words as species binomial
    words = desc.split()
    if len(words) >= 2:
        # Handle "Candidatus" prefix
        if words[0] == "Candidatus" and len(words) >= 3:
            return f"Candidatus {words[1]} {words[2]}"
        return f"{words[0]} {words[1]}"
    elif len(words) == 1:
        return words[0]
    return "Unknown"


def extract_genus(species: str) -> str:
    """Extract genus from species binomial."""
    if species.startswith("Candidatus "):
        parts = species.split()
        return f"Candidatus {parts[1]}" if len(parts) >= 2 else "Unknown"
    return species.split()[0] if species else "Unknown"


def lookup_phylum(genus: str) -> str:
    """Best-effort phylum lookup from genus name."""
    # Strip "Candidatus " prefix for lookup
    clean = genus.replace("Candidatus ", "")
    return GENUS_TO_PHYLUM.get(clean, "Unknown")


def main():
    parser = argparse.ArgumentParser(description="Build genome → taxonomy mapping")
    parser.add_argument("--genomes-dir", required=True, help="Directory containing .fna files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    genomes_dir = Path(args.genomes_dir)
    fna_files = sorted(genomes_dir.glob("*.fna"))

    if not fna_files:
        print(f"ERROR: No .fna files found in {genomes_dir}")
        sys.exit(1)

    print(f"Processing {len(fna_files)} genome files...")

    rows = []
    genus_counts = Counter()
    phylum_counts = Counter()

    for i, fna in enumerate(fna_files):
        # Genome ID = filename without .fna (matches CSV 'genome' column)
        genome_id = fna.stem

        # Read first line for species info
        with open(fna, "r") as f:
            header = f.readline().strip()

        species = parse_species_from_header(header)
        genus = extract_genus(species)
        phylum = lookup_phylum(genus)

        rows.append({
            "genome": genome_id,
            "species": species,
            "genus": genus,
            "phylum": phylum,
        })

        genus_counts[genus] += 1
        phylum_counts[phylum] += 1

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(fna_files)}...")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["genome", "species", "genus", "phylum"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"  Output: {args.output}")
    print(f"  Genomes:  {len(rows)}")
    print(f"  Species:  {len(set(r['species'] for r in rows))}")
    print(f"  Genera:   {len(genus_counts)}")
    print(f"  Phyla:    {len(phylum_counts)}")
    print(f"{'='*60}")

    print(f"\nTop 20 genera:")
    for genus, count in genus_counts.most_common(20):
        print(f"  {genus:<30s} {count:>5d} genomes")

    print(f"\nPhylum distribution:")
    for phylum, count in phylum_counts.most_common():
        print(f"  {phylum:<30s} {count:>5d} genomes ({100*count/len(rows):.1f}%)")

    # Flag genera with unknown phylum
    unknown_genera = [g for g, c in genus_counts.items()
                      if lookup_phylum(g) == "Unknown"]
    if unknown_genera:
        print(f"\n  {len(unknown_genera)} genera with unknown phylum "
              f"({sum(genus_counts[g] for g in unknown_genera)} genomes)")
        print(f"  Top unmapped: {', '.join(sorted(unknown_genera, key=lambda g: -genus_counts[g])[:10])}")


if __name__ == "__main__":
    main()
