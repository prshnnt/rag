
# ============================================================================
# CLI INTERFACE
# ============================================================================

from pathlib import Path
import sys

from loguru import logger
from config.legal_guardrails import SUPPORTED_LAWS
from config.settings import Settings
from ingestion.build_indices_pdf import IndexBuilder


def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build indices for Indian Legal RAG System"
    )
    parser.add_argument(
        "--laws",
        nargs="+",
        choices=list(SUPPORTED_LAWS.keys()),
        help="Specific laws to process (default: all active laws)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild indices even if they exist"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = Settings()
    
    # Check if indices already exist
    index_dir = Path(settings.index_dir)
    if (index_dir / "faiss.index").exists() and not args.rebuild:
        logger.warning("Indices already exist. Use --rebuild to force rebuild.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Aborted.")
            return
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        laws = args.laws or [code for code, info in SUPPORTED_LAWS.items() if info.get("active")]
        logger.info(f"Would process: {', '.join(laws)}")
        return
    
    # Build indices
    builder = IndexBuilder(settings)
    
    try:
        builder.build_all(laws=args.laws)
        logger.success("\n✓ Index build completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("\n\nBuild interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n\n✗ Build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

# ============================================================================
# USAGE EXAMPLES
# ============================================================================
"""
# Basic usage
python -m src.ingestion.build_indices

# Build specific laws only
python -m src.ingestion.build_indices --laws constitution bns

# Rebuild existing indices
python -m src.ingestion.build_indices --rebuild

# Dry run (see what would be done)
python -m src.ingestion.build_indices --dry-run

# Check build stats
cat data/indices/build_stats.json
"""