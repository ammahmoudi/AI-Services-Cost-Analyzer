#!/usr/bin/env python
"""
Management CLI for AI Cost Manager

Provides commands to manage API sources and extract model data.
"""
import sys
import argparse
from datetime import datetime
from ai_cost_manager.database import init_db, get_session, close_session
from ai_cost_manager.models import APISource, AIModel, LLMConfiguration
from extractors import get_extractor, list_extractors


def cmd_init_db(args):
    """Initialize the database"""
    print("Initializing database...")
    init_db()
    print("✓ Database initialized successfully!")


def cmd_add_source(args):
    """Add a new API source"""
    session = get_session()
    
    try:
        # Check if source already exists
        existing = session.query(APISource).filter_by(name=args.name).first()
        if existing:
            print(f"✗ Source '{args.name}' already exists!")
            if not args.force:
                return
            print("  Updating existing source...")
            source = existing
        else:
            source = APISource()
        
        # Update source fields
        source.name = args.name
        source.url = args.url
        source.extractor_name = args.extractor
        source.is_active = True
        
        if not existing:
            session.add(source)
        
        session.commit()
        print(f"✓ Source '{args.name}' added successfully!")
        print(f"  URL: {args.url}")
        print(f"  Extractor: {args.extractor}")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error adding source: {e}")
    finally:
        close_session()


def cmd_list_sources(args):
    """List all API sources"""
    session = get_session()
    
    try:
        sources = session.query(APISource).all()
        
        if not sources:
            print("No sources configured yet.")
            print("\nAdd a source with:")
            print("  python manage.py add-source --name 'Source Name' --url 'URL' --extractor 'extractor_name'")
            return
        
        print(f"\n{'ID':<5} {'Name':<20} {'Extractor':<15} {'Models':<10} {'Active':<8} {'Last Extracted'}")
        print("-" * 90)
        
        for source in sources:
            model_count = session.query(AIModel).filter_by(source_id=source.id).count()
            last_extracted = source.last_extracted.strftime('%Y-%m-%d %H:%M') if source.last_extracted else 'Never'
            active = '✓' if source.is_active else '✗'
            
            print(f"{source.id:<5} {source.name:<20} {source.extractor_name:<15} {model_count:<10} {active:<8} {last_extracted}")
        
    finally:
        close_session()


def cmd_extract(args):
    """Extract data from API sources"""
    session = get_session()
    
    try:
        # Check if LLM config exists
        llm_config = session.query(LLMConfiguration).filter_by(is_active=True).first()
        
        # If --use-llm is not specified and LLM config exists, ask the user
        if not args.use_llm and llm_config:
            print("\n" + "="*60)
            print("🤖 LLM Configuration Detected")
            print("="*60)
            print(f"Active LLM: {llm_config.model_name}")
            print("\nLLM can extract detailed pricing information from model data.")
            print("This includes pricing type, formulas, cost units, and variables.")
            response = input("\nUse LLM for pricing extraction? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                args.use_llm = True
                print("✓ LLM extraction enabled\n")
            else:
                print("✓ Skipping LLM extraction\n")
        
        # Get sources to extract
        if args.source_id:
            sources = [session.query(APISource).filter_by(id=args.source_id).first()]
            if not sources[0]:
                print(f"✗ Source with ID {args.source_id} not found!")
                return
        else:
            sources = session.query(APISource).filter_by(is_active=True).all()
        
        if not sources:
            print("No active sources to extract from!")
            return
        
        total_models = 0
        
        for source in sources:
            print(f"\n{'='*60}")
            print(f"Extracting from: {source.name}")
            print(f"{'='*60}")
            
            try:
                # Get extractor
                extractor = get_extractor(source.extractor_name, source.url)
                
                # Enable force refresh if requested
                if args.force_refresh and hasattr(extractor, 'force_refresh'):
                    extractor.force_refresh = True
                    print(f"  🔄 Force refresh enabled - bypassing cache for {source.name}")
                
                # Enable schema fetching if requested (only for fal extractor)
                if args.fetch_schemas and hasattr(extractor, 'fetch_schemas'):
                    extractor.fetch_schemas = True
                    print(f"  📋 Schema fetching enabled for {source.name}")
                
                # Enable LLM extraction if requested
                if args.use_llm and hasattr(extractor, 'use_llm'):
                    extractor.use_llm = True
                    print(f"  🤖 LLM pricing extraction enabled for {source.name}")
                
                # Extract models
                models_data = extractor.extract()
                
                if not models_data:
                    print(f"  ⚠ No models extracted from {source.name}")
                    continue
                
                # Save models to database
                new_count = 0
                updated_count = 0
                
                for model_data in models_data:
                    existing_model = session.query(AIModel).filter_by(
                        source_id=source.id,
                        model_id=model_data['model_id']
                    ).first()
                    
                    if existing_model:
                        # Update existing model
                        for key, value in model_data.items():
                            setattr(existing_model, key, value)
                        updated_count += 1
                    else:
                        # Create new model
                        model = AIModel(source_id=source.id, **model_data)
                        session.add(model)
                        new_count += 1
                
                # Update last_extracted timestamp
                source.last_extracted = datetime.utcnow()
                
                session.commit()
                
                print(f"  ✓ Extracted {len(models_data)} models")
                print(f"    - New: {new_count}")
                print(f"    - Updated: {updated_count}")
                
                total_models += len(models_data)
                
            except Exception as e:
                print(f"  ✗ Error extracting from {source.name}: {e}")
                session.rollback()
        
        print(f"\n{'='*60}")
        print(f"✓ Total models extracted: {total_models}")
        print(f"{'='*60}")
        
    finally:
        close_session()


def cmd_list_models(args):
    """List all AI models"""
    session = get_session()
    
    try:
        query = session.query(AIModel)
        
        # Filter by source if specified
        if args.source_id:
            query = query.filter_by(source_id=args.source_id)
        
        # Filter by type if specified
        if args.type:
            query = query.filter_by(model_type=args.type)
        
        models = query.all()
        
        if not models:
            print("No models found.")
            return
        
        print(f"\n{'ID':<6} {'Name':<30} {'Type':<20} {'Cost/Call':<12} {'Source'}")
        print("-" * 95)
        
        for model in models:
            source_name = model.source.name if model.source else 'Unknown'
            cost_str = f"${model.cost_per_call:.4f}" if model.cost_per_call else "N/A"
            
            print(f"{model.id:<6} {model.name[:28]:<30} {model.model_type[:18]:<20} {cost_str:<12} {source_name}")
        
        print(f"\nTotal: {len(models)} models")
        
    finally:
        close_session()


def cmd_list_extractors(args):
    """List available extractors"""
    from extractors import PRIMARY_EXTRACTORS
    
    print("\nAvailable extractors:")
    for name, description in PRIMARY_EXTRACTORS.items():
        print(f"  • {description}")
    print()


def cmd_config_llm(args):
    """Configure LLM for pricing extraction"""
    from ai_cost_manager.openrouter_client import fetch_openrouter_models, get_recommended_models, format_model_for_display
    
    # Handle --list-models flag
    if args.list_models:
        print("\n" + "="*60)
        print("Fetching available OpenRouter models...")
        print("="*60)
        
        api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else None
        models = fetch_openrouter_models(api_key, sort_by_free=True)
        
        if not models:
            print("⚠ Could not fetch models. Check your connection.")
            return
        
        # Separate free and paid models
        free_models = [m for m in models if m.get('is_free')]
        paid_models = [m for m in models if not m.get('is_free')]
        recommended = get_recommended_models()
        
        print(f"\n✓ Found {len(models)} models ({len(free_models)} free, {len(paid_models)} paid)\n")
        
        # Show free models first
        if free_models:
            print("🆓 FREE MODELS:")
            print("-" * 60)
            for i, model in enumerate(free_models, 1):
                print(f"  {i}. {format_model_for_display(model)}")
            print()
        
        print("⭐ RECOMMENDED PAID MODELS FOR PRICING EXTRACTION:")
        print("-" * 60)
        
        for model in models:
            if model['id'] in recommended and not model.get('is_free'):
                print(f"  ⭐ {format_model_for_display(model)}")
        
        print(f"\n\nALL PAID MODELS: ({len(paid_models)} total)")
        print("-" * 60)
        
        for i, model in enumerate(paid_models[:20], 1):  # Show first 20 paid
            if model['id'] not in recommended:
                print(f"  {i}. {format_model_for_display(model)}")
        
        if len(paid_models) > 20:
            print(f"\n  ... and {len(paid_models) - 20} more paid models")
        
        print("\nTo configure, run:")
        print("  python manage.py config-llm --api-key YOUR_KEY --model MODEL_ID")
        return
    
    # Require API key for configuration
    if not args.api_key:
        print("Error: --api-key is required for configuration")
        print("Tip: Use --list-models to see available models first")
        return
    
    session = get_session()
    
    try:
        # Deactivate existing configs if setting as active
        if not args.inactive:
            session.query(LLMConfiguration).update({'is_active': False})
        
        config = LLMConfiguration(
            api_key=args.api_key,
            model_name=args.model or 'openai/gpt-4o-mini',
            base_url=args.base_url or 'https://openrouter.ai/api/v1',
            is_active=not args.inactive
        )
        
        session.add(config)
        session.commit()
        
        print(f"✓ LLM configuration saved!")
        print(f"  Model: {config.model_name}")
        print(f"  Active: {config.is_active}")
        
    except Exception as e:
        session.rollback()
        print(f"✗ Error saving LLM config: {e}")
    finally:
        close_session()


def cmd_list_llm_configs(args):
    """List LLM configurations"""
    session = get_session()
    
    try:
        configs = session.query(LLMConfiguration).all()
        
        if not configs:
            print("\nNo LLM configurations found.")
            print("\nAdd one with:")
            print('  python manage.py config-llm --api-key "your-openrouter-key"')
            return
        
        print(f"\n{'ID':<5} {'Model':<35} {'Active':<8} {'Created'}")
        print("-" * 70)
        
        for config in configs:
            active = '✓' if config.is_active else '✗'
            created = config.created_at.strftime('%Y-%m-%d')
            model = config.model_name[:33]
            
            print(f"{config.id:<5} {model:<35} {active:<8} {created}")
        
        print()
        
    finally:
        close_session()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='AI Cost Manager - Manage AI model pricing data'
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # init-db command
    parser_init = subparsers.add_parser('init-db', help='Initialize the database')
    parser_init.set_defaults(func=cmd_init_db)
    
    # add-source command
    parser_add = subparsers.add_parser('add-source', help='Add a new API source')
    parser_add.add_argument('--name', required=True, help='Source name')
    parser_add.add_argument('--url', required=True, help='Source API URL')
    parser_add.add_argument('--extractor', required=True, help='Extractor name (e.g., fal, openai)')
    parser_add.add_argument('--force', action='store_true', help='Force update if exists')
    parser_add.set_defaults(func=cmd_add_source)
    
    # list-sources command
    parser_list_src = subparsers.add_parser('list-sources', help='List all API sources')
    parser_list_src.set_defaults(func=cmd_list_sources)
    
    # extract command
    parser_extract = subparsers.add_parser('extract', help='Extract data from sources')
    parser_extract.add_argument('--source-id', type=int, help='Extract from specific source ID only')
    parser_extract.add_argument('--fetch-schemas', action='store_true', help='Fetch OpenAPI schemas for models (slower but more complete)')
    parser_extract.add_argument('--use-llm', action='store_true', help='Use LLM to extract detailed pricing information (requires LLM config)')
    parser_extract.add_argument('--force-refresh', action='store_true', help='Force refresh data, bypassing cache')
    parser_extract.set_defaults(func=cmd_extract)
    
    # list-models command
    parser_list_mod = subparsers.add_parser('list-models', help='List all AI models')
    parser_list_mod.add_argument('--source-id', type=int, help='Filter by source ID')
    parser_list_mod.add_argument('--type', help='Filter by model type')
    parser_list_mod.set_defaults(func=cmd_list_models)
    
    # list-extractors command
    parser_list_ext = subparsers.add_parser('list-extractors', help='List available extractors')
    parser_list_ext.set_defaults(func=cmd_list_extractors)
    
    # config-llm command
    parser_config_llm = subparsers.add_parser('config-llm', help='Configure LLM for pricing extraction')
    parser_config_llm.add_argument('--api-key', help='OpenRouter API key')
    parser_config_llm.add_argument('--model', help='Model to use (default: openai/gpt-4o-mini)')
    parser_config_llm.add_argument('--base-url', help='Base URL (default: https://openrouter.ai/api/v1)')
    parser_config_llm.add_argument('--inactive', action='store_true', help='Set as inactive')
    parser_config_llm.add_argument('--list-models', action='store_true', help='List available OpenRouter models')
    parser_config_llm.set_defaults(func=cmd_config_llm)
    
    # list-llm-configs command
    parser_list_llm = subparsers.add_parser('list-llm-configs', help='List LLM configurations')
    parser_list_llm.set_defaults(func=cmd_list_llm_configs)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
