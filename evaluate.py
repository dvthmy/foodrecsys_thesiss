#!/usr/bin/env python3
"""Evaluation Runner Script.

Runs the comprehensive evaluation of the Food Recommendation System
and generates a detailed report covering all evaluation dimensions.

Usage:
    python evaluate.py                    # Run all evaluations
    python evaluate.py --quick            # Run quick evaluation (fewer samples)
    python evaluate.py --export report    # Export report to JSON file
    python evaluate.py --section safety   # Run specific section only
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.evaluation_service import (
    EvaluationService,
    EvaluationReport,
    run_evaluation,
)
from src.services.neo4j_service import Neo4jService


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the evaluation script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def export_report(report: EvaluationReport, output_path: str) -> None:
    """Export evaluation report to JSON file.
    
    Args:
        report: The evaluation report to export.
        output_path: Path to the output JSON file.
    """
    # Convert dataclasses to dict
    report_dict = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "canonicalization": asdict(report.canonicalization),
        "graph_topology": asdict(report.graph_topology),
        "jaccard_matrix": asdict(report.jaccard_matrix),
        "retrieval": asdict(report.retrieval),
        "ranking": asdict(report.ranking),
        "ablation": asdict(report.ablation),
        "safety": asdict(report.safety),
        "behavioral": asdict(report.behavioral),
        "latency": asdict(report.latency),
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"\n📄 Report exported to: {output_file.absolute()}")


def run_section(service: EvaluationService, section: str) -> None:
    """Run a specific evaluation section.
    
    Args:
        service: The evaluation service instance.
        section: Name of the section to run.
    """
    section_map = {
        "canonicalization": service.evaluate_canonicalization,
        "topology": service.evaluate_graph_topology,
        "jaccard": service.evaluate_jaccard_matrix,
        "retrieval": service.evaluate_image_retrieval,
        "ranking": service.evaluate_ranking_quality,
        "ablation": service.evaluate_ablation_study,
        "safety": service.evaluate_safety,
        "behavioral": service.evaluate_behavioral_metrics,
        "latency": service.evaluate_latency,
    }
    
    if section not in section_map:
        print(f"Unknown section: {section}")
        print(f"Available sections: {', '.join(section_map.keys())}")
        return
    
    print(f"\n🔬 Running {section} evaluation...")
    section_map[section]()
    service.print_report()


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive evaluation of the Food Recommendation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                      Run full evaluation
  python evaluate.py --quick              Quick evaluation with fewer samples
  python evaluate.py --export results     Export report to results.json
  python evaluate.py --section safety     Run only safety evaluation
  python evaluate.py -v                   Verbose logging
        """,
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation with fewer samples",
    )
    
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export report to JSON file",
    )
    
    parser.add_argument(
        "--section",
        type=str,
        choices=[
            "canonicalization",
            "topology",
            "jaccard",
            "retrieval",
            "ranking",
            "ablation",
            "safety",
            "behavioral",
            "latency",
        ],
        help="Run only a specific evaluation section",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    print("\n" + "=" * 70)
    print("   🍽️  FOOD RECOMMENDATION SYSTEM - EVALUATION SUITE")
    print("   Multimodal Graph-Based Recommendation Evaluation")
    print("=" * 70)
    
    try:
        # Verify database connection
        print("\n📊 Connecting to Neo4j database...")
        neo4j = Neo4jService()
        neo4j.verify_connectivity()
        print("✓ Database connection successful")
        
        # Create evaluation service
        service = EvaluationService(neo4j_service=neo4j)
        
        # Run evaluation
        if args.section:
            # Load data first
            service._load_data()
            run_section(service, args.section)
            report = service._report
        else:
            print("\n🚀 Starting full evaluation...")
            report = service.run_full_evaluation()
            service.print_report()
        
        # Export if requested
        if args.export:
            export_report(report, args.export)
        
        # Cleanup
        neo4j.close()
        
        print("\n✅ Evaluation completed successfully!\n")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
        return 1
        
    except Exception as e:
        logging.exception("Evaluation failed with error")
        print(f"\n❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
