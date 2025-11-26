"""
AI Cost Manager - Web Application

Simple Flask web interface for viewing and managing AI model costs.
"""
import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from ai_cost_manager.database import get_session, close_session, init_db
from ai_cost_manager.models import APISource, AIModel, LLMConfiguration
from extractors import get_extractor, list_extractors
from datetime import datetime

# Load environment
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize database on startup
with app.app_context():
    init_db()


@app.route('/')
def index():
    """Home page - show dashboard"""
    session = get_session()
    
    try:
        sources = session.query(APISource).all()
        models = session.query(AIModel).filter_by(is_active=True).all()
        
        # Calculate statistics
        total_models = len(models)
        total_sources = len(sources)
        
        # Group models by type
        model_types = {}
        for model in models:
            model_type = model.model_type or 'other'
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        # Recent models
        recent_models = session.query(AIModel).order_by(AIModel.updated_at.desc()).limit(10).all()
        
        return render_template('index.html',
                             sources=sources,
                             total_models=total_models,
                             total_sources=total_sources,
                             model_types=model_types,
                             recent_models=recent_models)
    finally:
        close_session()


@app.route('/sources')
def sources():
    """List all API sources"""
    session = get_session()
    
    try:
        sources = session.query(APISource).all()
        # Get primary extractors only (no aliases)
        from extractors import PRIMARY_EXTRACTORS
        available_extractors = list(PRIMARY_EXTRACTORS.keys())
        
        return render_template('sources.html',
                             sources=sources,
                             extractors=available_extractors)
    finally:
        close_session()


@app.route('/sources/add', methods=['POST'])
def add_source():
    """Add a new API source"""
    session = get_session()
    
    try:
        name = request.form.get('name')
        url = request.form.get('url')
        extractor = request.form.get('extractor')
        
        if not all([name, url, extractor]):
            flash('All fields are required!', 'error')
            return redirect(url_for('sources'))
        
        # Check if exists
        existing = session.query(APISource).filter_by(name=name).first()
        if existing:
            flash(f'Source "{name}" already exists!', 'error')
            return redirect(url_for('sources'))
        
        source = APISource(
            name=name,
            url=url,
            extractor_name=extractor,
            is_active=True
        )
        
        session.add(source)
        session.commit()
        
        flash(f'Source "{name}" added successfully!', 'success')
        
    except Exception as e:
        session.rollback()
        flash(f'Error adding source: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('sources'))


@app.route('/sources/<int:source_id>/delete', methods=['POST'])
def delete_source(source_id):
    """Delete an API source"""
    session = get_session()
    
    try:
        source = session.query(APISource).filter_by(id=source_id).first()
        if source:
            session.delete(source)
            session.commit()
            flash(f'Source "{source.name}" deleted!', 'success')
        else:
            flash('Source not found!', 'error')
    except Exception as e:
        session.rollback()
        flash(f'Error deleting source: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('sources'))


@app.route('/sources/<int:source_id>/extract', methods=['POST'])
def extract_source(source_id):
    """Extract data from a specific source"""
    session = get_session()
    
    try:
        source = session.query(APISource).filter_by(id=source_id).first()
        if not source:
            return jsonify({'error': 'Source not found'}), 404
        
        # Get options from request
        use_llm = request.json.get('use_llm', False) if request.is_json else False
        fetch_schemas = request.json.get('fetch_schemas', False) if request.is_json else False
        force_refresh = request.json.get('force_refresh', False) if request.is_json else False
        
        # Get extractor
        extractor = get_extractor(source.extractor_name, source.url)
        
        # Enable options if requested
        if fetch_schemas and hasattr(extractor, 'fetch_schemas'):
            extractor.fetch_schemas = True
        
        if use_llm and hasattr(extractor, 'use_llm'):
            extractor.use_llm = True
        
        if force_refresh and hasattr(extractor, 'force_refresh'):
            extractor.force_refresh = True
        
        # Extract models
        models_data = extractor.extract()
        
        if not models_data:
            return jsonify({'error': 'No models extracted'}), 400
        
        # Save models
        new_count = 0
        updated_count = 0
        
        for model_data in models_data:
            # Remove keys that aren't columns in the model
            filtered_data = {
                k: v for k, v in model_data.items()
                if hasattr(AIModel, k)
            }
            
            existing_model = session.query(AIModel).filter_by(
                source_id=source.id,
                model_id=model_data['model_id']
            ).first()
            
            if existing_model:
                for key, value in filtered_data.items():
                    if key != 'id':  # Don't update the primary key
                        setattr(existing_model, key, value)
                updated_count += 1
            else:
                model = AIModel(source_id=source.id, **filtered_data)
                session.add(model)
                new_count += 1
        
        source.last_extracted = datetime.utcnow()
        session.commit()
        
        flash(f'Extracted {len(models_data)} models ({new_count} new, {updated_count} updated)', 'success')
        
        return jsonify({
            'success': True,
            'total': len(models_data),
            'new': new_count,
            'updated': updated_count
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/models')
def models():
    """List all AI models"""
    session = get_session()
    
    try:
        # Get filters
        source_id = request.args.get('source_id', type=int)
        model_type = request.args.get('type')
        search = request.args.get('search', '')
        missing_data = request.args.get('missing_data')  # 'schema', 'playground', 'llm'
        
        # Build query
        query = session.query(AIModel)
        
        if source_id:
            query = query.filter_by(source_id=source_id)
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        if search:
            query = query.filter(
                (AIModel.name.ilike(f'%{search}%')) |
                (AIModel.description.ilike(f'%{search}%'))
            )
        
        # Filter by missing data
        if missing_data == 'schema':
            query = query.filter(AIModel.last_schema_fetched.is_(None))
        elif missing_data == 'playground':
            query = query.filter(AIModel.last_playground_fetched.is_(None))
        elif missing_data == 'llm':
            query = query.filter(AIModel.last_llm_fetched.is_(None))
        
        models = query.order_by(AIModel.name).all()
        sources = session.query(APISource).all()
        
        # Get unique model types
        all_types = session.query(AIModel.model_type).distinct().all()
        model_types = [t[0] for t in all_types if t[0]]
        
        # Count missing data statistics
        total_models = session.query(AIModel).count()
        missing_schema = session.query(AIModel).filter(AIModel.last_schema_fetched.is_(None)).count()
        missing_playground = session.query(AIModel).filter(AIModel.last_playground_fetched.is_(None)).count()
        missing_llm = session.query(AIModel).filter(AIModel.last_llm_fetched.is_(None)).count()
        
        return render_template('models.html',
                             models=models,
                             sources=sources,
                             model_types=model_types,
                             current_source=source_id,
                             current_type=model_type,
                             search=search,
                             missing_data=missing_data,
                             stats={
                                 'total': total_models,
                                 'missing_schema': missing_schema,
                                 'missing_playground': missing_playground,
                                 'missing_llm': missing_llm
                             })
    finally:
        close_session()


@app.route('/models/<int:model_id>')
def model_detail(model_id):
    """Show detailed information about a model"""
    session = get_session()
    
    try:
        model = session.query(AIModel).filter_by(id=model_id).first()
        if not model:
            flash('Model not found!', 'error')
            return redirect(url_for('models'))
        
        return render_template('model_detail.html', model=model)
    finally:
        close_session()


@app.route('/llm-config')
def llm_config():
    """LLM configuration page"""
    from ai_cost_manager.openrouter_client import fetch_openrouter_models, get_recommended_models
    
    session = get_session()
    
    try:
        configs = session.query(LLMConfiguration).all()
        active_config = session.query(LLMConfiguration).filter_by(is_active=True).first()
        
        # Fetch available models (without requiring API key) and sort with free models first
        available_models = fetch_openrouter_models(sort_by_free=True)
        recommended = get_recommended_models()
        
        return render_template('llm_config.html',
                             configs=configs,
                             active_config=active_config,
                             available_models=available_models,
                             recommended_models=recommended)
    finally:
        close_session()


@app.route('/llm-config/save', methods=['POST'])
def save_llm_config():
    """Save LLM configuration"""
    session = get_session()
    
    try:
        api_key = request.form.get('api_key')
        model_name = request.form.get('model_name', 'openai/gpt-4o-mini')
        base_url = request.form.get('base_url', 'https://openrouter.ai/api/v1')
        
        if not api_key:
            flash('API key is required!', 'error')
            return redirect(url_for('llm_config'))
        
        # Deactivate all existing configs
        session.query(LLMConfiguration).update({'is_active': False})
        
        # Create new config
        config = LLMConfiguration(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            is_active=True
        )
        
        session.add(config)
        session.commit()
        
        flash('LLM configuration saved successfully!', 'success')
        
    except Exception as e:
        session.rollback()
        flash(f'Error saving configuration: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('llm_config'))


@app.route('/llm-config/<int:config_id>/activate', methods=['POST'])
def activate_llm_config(config_id):
    """Activate an LLM configuration"""
    session = get_session()
    
    try:
        # Deactivate all
        session.query(LLMConfiguration).update({'is_active': False})
        
        # Activate selected
        config = session.query(LLMConfiguration).filter_by(id=config_id).first()
        if config:
            config.is_active = True
            session.commit()
            flash(f'Activated configuration with model {config.model_name}', 'success')
        else:
            flash('Configuration not found!', 'error')
            
    except Exception as e:
        session.rollback()
        flash(f'Error activating configuration: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('llm_config'))


@app.route('/llm-config/<int:config_id>/delete', methods=['POST'])
def delete_llm_config(config_id):
    """Delete an LLM configuration"""
    session = get_session()
    
    try:
        config = session.query(LLMConfiguration).filter_by(id=config_id).first()
        if config:
            session.delete(config)
            session.commit()
            flash('Configuration deleted!', 'success')
        else:
            flash('Configuration not found!', 'error')
            
    except Exception as e:
        session.rollback()
        flash(f'Error deleting configuration: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('llm_config'))


@app.route('/models/<int:model_id>/extract-pricing', methods=['POST'])
def extract_model_pricing(model_id):
    """Extract pricing for a specific model using LLM"""
    from ai_cost_manager.llm_extractor import extract_pricing_with_llm
    
    session = get_session()
    
    try:
        model = session.query(AIModel).filter_by(id=model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Check if LLM config exists
        llm_config = session.query(LLMConfiguration).filter_by(is_active=True).first()
        if not llm_config:
            return jsonify({'error': 'No active LLM configuration. Please configure LLM first.'}), 400
        
        # Extract pricing using LLM
        pricing_details = extract_pricing_with_llm({
            'name': model.name,
            'pricing_info': model.pricing_info or '',
            'creditsRequired': model.credits_required,
            'model_type': model.model_type,
        })
        
        if pricing_details:
            # Update model with extracted pricing
            model.pricing_type = pricing_details.get('pricing_type')
            model.pricing_formula = pricing_details.get('pricing_formula')
            model.pricing_variables = pricing_details.get('pricing_variables')
            model.input_cost_per_unit = pricing_details.get('input_cost_per_unit')
            model.output_cost_per_unit = pricing_details.get('output_cost_per_unit')
            model.cost_unit = pricing_details.get('cost_unit')
            model.llm_extracted = True
            
            session.commit()
            
            return jsonify({
                'success': True,
                'pricing': pricing_details
            })
        else:
            return jsonify({'error': 'Failed to extract pricing information'}), 500
    
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/cache')
def cache_management():
    """Cache management page"""
    from ai_cost_manager.cache import cache_manager
    
    stats = cache_manager.get_cache_stats()
    
    return render_template('cache.html', cache_stats=stats)


@app.route('/cache/clear/<source_name>', methods=['POST'])
def clear_source_cache(source_name):
    """Clear cache for a specific source"""
    try:
        from ai_cost_manager.cache import cache_manager
        cache_manager.clear_cache(source_name)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/clear-all', methods=['POST'])
def clear_all_cache():
    """Clear all cache"""
    try:
        from ai_cost_manager.cache import cache_manager
        cache_manager.clear_all_cache()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/clear-llm', methods=['POST'])
def clear_llm_cache():
    """Clear LLM extraction cache"""
    try:
        from ai_cost_manager.cache import cache_manager
        cache_manager.clear_llm_cache()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/cache/clear-schemas', methods=['POST'])
def clear_schema_cache():
    """Clear schema cache"""
    try:
        from ai_cost_manager.cache import cache_manager
        cache_manager.clear_schema_cache()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
