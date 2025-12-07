"""
CLI tool for running model matching

Usage:
    python run_matching.py [--force]
"""
import sys
import argparse
from ai_cost_manager.database import get_session, close_session, init_db
from ai_cost_manager.model_matching_service import ModelMatchingService


def main():
    parser = argparse.ArgumentParser(description='Run model matching across all providers')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-matching even if matches already exist')
    
    args = parser.parse_args()
    
    print("🤖 AI Cost Manager - Model Matching Tool")
    print("=" * 50)
    print()
    
    # Initialize database
    print("Initializing database...")
    init_db()
    
    session = get_session()
    
    try:
        service = ModelMatchingService(session)
        
        print(f"Running model matching (force_refresh={args.force})...")
        print()
        
        result = service.match_all_models(force_refresh=args.force)
        
        print("=" * 50)
        print("Results:")
        print("=" * 50)
        
        if result['status'] == 'already_matched' and not args.force:
            print("⚠️  Models already matched!")
            print(f"   Existing matches: {result['existing_matches']}")
            print("   Use --force to re-match")
        elif result['status'] == 'no_models':
            print("⚠️  No models found to match")
            print("   Add models from sources first")
        elif result['status'] == 'success':
            print("✅ Model matching complete!")
            print()
            print("📊 Statistics:")
            print(f"   • Canonical models created: {result['canonical_models_created']}")
            print(f"   • Model matches created: {result['model_matches_created']}")
            print(f"   • Total models processed: {result['total_models_processed']}")
            print(f"   • Match groups: {result['match_groups']}")
            print()
            print("💡 View results at: http://localhost:5000/canonical-models")
        else:
            print(f"⚠️  Unknown status: {result['status']}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        close_session()


if __name__ == '__main__':
    main()
