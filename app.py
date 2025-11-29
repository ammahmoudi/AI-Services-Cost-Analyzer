"""
AI Cost Manager - Web Application

Simple Flask web interface for viewing and managing AI model costs.
"""
import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from ai_cost_manager.database import get_session, close_session, init_db
from ai_cost_manager.models import APISource, AIModel, LLMConfiguration, AuthSettings, ExtractorAPIKey
from extractors import get_extractor, list_extractors
from datetime import datetime

# Load environment
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Add Jinja2 filters
@app.template_filter('from_json')
def from_json_filter(s):
    """Convert JSON string to Python object"""
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}

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
        # Expire all cached data to ensure fresh database reads
        session.expire_all()
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


@app.route('/api/sources/<int:source_id>/features', methods=['GET'])
def get_source_features(source_id):
    """Get extractor feature support for a specific source"""
    from extractors import get_extractor_features
    
    session = get_session()
    
    try:
        source = session.query(APISource).filter_by(id=source_id).first()
        if not source:
            return jsonify({'error': 'Source not found'}), 404
        
        features = get_extractor_features(source.extractor_name)
        return jsonify(features)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/sources/<int:source_id>/extract', methods=['POST'])
def extract_source(source_id):
    """Extract data from a specific source"""
    from ai_cost_manager.progress_tracker import ProgressTracker
    
    session = get_session()
    
    try:
        source = session.query(APISource).filter_by(id=source_id).first()
        if not source:
            return jsonify({'error': 'Source not found'}), 404
        
        # Store source attributes we need before potentially detaching
        source_name = source.name
        source_url = source.url
        source_extractor_name = source.extractor_name
        source_db_id = source.id
        
        # Get options from request
        use_llm = request.json.get('use_llm', False) if request.is_json else False
        fetch_schemas = request.json.get('fetch_schemas', False) if request.is_json else False
        force_refresh = request.json.get('force_refresh', False) if request.is_json else False
        
        # Create progress tracker
        progress_tracker = ProgressTracker(source_name, source_db_id)
        
        # Get extractor class and instantiate it
        extractor_class = get_extractor(source_extractor_name)
        extractor = extractor_class(source_url=source_url, fetch_schemas=fetch_schemas, use_llm=use_llm)
        
        # Enable force refresh if requested
        if force_refresh and hasattr(extractor, 'force_refresh'):
            extractor.force_refresh = True
        
        if use_llm and hasattr(extractor, 'use_llm'):
            extractor.use_llm = True
        
        if force_refresh and hasattr(extractor, 'force_refresh'):
            extractor.force_refresh = True
        
        # Extract models with progress tracking
        models_data = extractor.extract(progress_tracker=progress_tracker)
        
        if not models_data:
            if progress_tracker:
                progress_tracker.error('No models extracted - API may be unavailable or returned empty response')
            return jsonify({
                'error': 'No models extracted. The API may be temporarily unavailable or returned no data. Please try again later.',
                'success': False
            }), 400
        
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
                source_id=source_db_id,
                model_id=model_data['model_id']
            ).first()
            
            if existing_model:
                for key, value in filtered_data.items():
                    if key != 'id':  # Don't update the primary key
                        setattr(existing_model, key, value)
                updated_count += 1
                progress_tracker.increment_updated()
            else:
                model = AIModel(source_id=source_db_id, **filtered_data)
                session.add(model)
                new_count += 1
                progress_tracker.increment_new()
        
        # Re-query source again to update last extracted timestamp
        source = session.query(APISource).filter_by(id=source_db_id).first()
        if source:
            source.last_extracted = datetime.utcnow()
            session.commit()
            # Refresh the source to ensure the timestamp is persisted
            session.refresh(source)
        else:
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
        if 'progress_tracker' in locals():
            progress_tracker.error(str(e))
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
        category = request.args.get('category')
        tags_filter = request.args.getlist('tags')  # Multi-select support
        search = request.args.get('search', '')
        missing_data = request.args.get('missing_data')  # 'cost', 'schema', 'llm', 'raw'
        
        # Build query
        query = session.query(AIModel)
        
        if source_id:
            query = query.filter_by(source_id=source_id)
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        if category:
            query = query.filter_by(category=category)
        
        # Filter by tags (any of the selected tags)
        if tags_filter:
            from sqlalchemy import cast, String, or_
            # Filter models that have ANY of the selected tags
            tag_filters = [cast(AIModel.tags, String).ilike(f'%{tag}%') for tag in tags_filter]
            query = query.filter(or_(*tag_filters))
        
        if search:
            query = query.filter(
                (AIModel.name.ilike(f'%{search}%')) |
                (AIModel.description.ilike(f'%{search}%'))
            )
        
        # Filter by missing data
        if missing_data == 'cost':
            query = query.filter((AIModel.cost_per_call == 0.0) | (AIModel.cost_per_call.is_(None)))
        elif missing_data == 'schema':
            query = query.filter(
                (AIModel.input_schema.is_(None)) & (AIModel.output_schema.is_(None))
            )
        elif missing_data == 'llm':
            query = query.filter(AIModel.llm_extracted.is_(None))
        elif missing_data == 'raw':
            query = query.filter(AIModel.raw_metadata.is_(None))
        
        models = query.order_by(AIModel.name).all()
        sources = session.query(APISource).all()
        
        # Get unique model types
        all_types = session.query(AIModel.model_type).distinct().all()
        model_types = sorted([t[0] for t in all_types if t[0]])
        
        # Get unique categories
        all_categories = session.query(AIModel.category).distinct().all()
        categories = sorted([c[0] for c in all_categories if c[0]])
        
        # Get unique tags (flatten all tag arrays)
        all_models = session.query(AIModel.tags).filter(AIModel.tags.isnot(None)).all()
        unique_tags = set()
        for model_tags in all_models:
            if model_tags[0] and isinstance(model_tags[0], list):
                unique_tags.update(model_tags[0])
        all_tags = sorted(list(unique_tags))
        
        # Count missing data statistics
        base_query = session.query(AIModel)
        if source_id:
            base_query = base_query.filter_by(source_id=source_id)
        
        total_models = base_query.count()
        missing_cost = base_query.filter((AIModel.cost_per_call == 0.0) | (AIModel.cost_per_call.is_(None))).count()
        missing_schema = base_query.filter(
            (AIModel.input_schema.is_(None)) & (AIModel.output_schema.is_(None))
        ).count()
        missing_llm = base_query.filter(AIModel.llm_extracted.is_(None)).count()
        missing_raw = base_query.filter(AIModel.raw_metadata.is_(None)).count()
        
        return render_template('models.html',
                             models=models,
                             sources=sources,
                             model_types=model_types,
                             categories=categories,
                             all_tags=all_tags,
                             current_source=source_id,
                             current_type=model_type,
                             current_category=category,
                             current_tags=tags_filter,
                             search=search,
                             missing_data=missing_data,
                             stats={
                                 'total': total_models,
                                 'missing_cost': missing_cost,
                                 'missing_schema': missing_schema,
                                 'missing_llm': missing_llm,
                                 'missing_raw': missing_raw,
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


@app.route('/models/<int:model_id>/delete', methods=['POST'])
def delete_model(model_id):
    """Delete a single model"""
    session = get_session()
    
    try:
        model = session.query(AIModel).filter_by(id=model_id).first()
        if not model:
            flash('Model not found!', 'error')
            return redirect(url_for('models'))
        
        model_name = model.name
        session.delete(model)
        session.commit()
        
        flash(f'Model "{model_name}" deleted successfully!', 'success')
        return redirect(url_for('models'))
        
    except Exception as e:
        session.rollback()
        flash(f'Error deleting model: {e}', 'error')
        return redirect(url_for('models'))
    finally:
        close_session()


@app.route('/sources/<int:source_id>/models/delete-all', methods=['POST'])
def delete_all_models(source_id):
    """Delete all models from a specific source"""
    session = get_session()
    
    try:
        source = session.query(APISource).filter_by(id=source_id).first()
        if not source:
            return jsonify({'error': 'Source not found'}), 404
        
        # Count models before deletion
        model_count = session.query(AIModel).filter_by(source_id=source_id).count()
        
        # Delete all models from this source
        session.query(AIModel).filter_by(source_id=source_id).delete()
        session.commit()
        
        return jsonify({
            'success': True,
            'deleted_count': model_count,
            'message': f'Deleted {model_count} models from {source.name}'
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/settings')
def settings():
    """Settings page with tabs for LLM and extractor API keys"""
    session = get_session()
    
    try:
        # Get LLM configuration
        llm_config = session.query(LLMConfiguration).filter_by(is_active=True).first()
        
        # Get extractor API keys
        together_key = session.query(ExtractorAPIKey).filter_by(extractor_name='together').first()
        
        # Get fal.ai authentication
        fal_auth = session.query(AuthSettings).filter_by(source_name='fal.ai').first()
        
        return render_template('settings.html',
                             llm_config=llm_config,
                             together_key=together_key,
                             fal_auth=fal_auth)
    finally:
        close_session()


@app.route('/settings/llm', methods=['POST'])
def save_llm_settings():
    """Save LLM configuration from settings page"""
    session = get_session()
    
    try:
        config_id = request.form.get('config_id')
        api_key = request.form.get('api_key')
        model_name = request.form.get('model_name', 'openai/gpt-4o-mini')
        base_url = request.form.get('base_url', 'https://openrouter.ai/api/v1')
        is_active = 'is_active' in request.form
        
        if not api_key:
            flash('API key is required!', 'error')
            return redirect(url_for('settings'))
        
        if config_id:
            # Update existing
            config = session.query(LLMConfiguration).filter_by(id=int(config_id)).first()
            if config:
                config.api_key = api_key
                config.model_name = model_name
                config.base_url = base_url
                config.is_active = is_active
                config.updated_at = datetime.utcnow()
        else:
            # Create new and deactivate others
            if is_active:
                session.query(LLMConfiguration).update({'is_active': False})
            
            config = LLMConfiguration(
                api_key=api_key,
                model_name=model_name,
                base_url=base_url,
                is_active=is_active
            )
            session.add(config)
        
        session.commit()
        flash('LLM configuration saved successfully!', 'success')
        
    except Exception as e:
        session.rollback()
        flash(f'Error saving LLM configuration: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('settings'))


@app.route('/settings/extractors', methods=['POST'])
def save_extractor_keys():
    """Save extractor API keys and authentication from settings page"""
    session = get_session()
    
    try:
        # Handle Together AI API key
        together_api_key = request.form.get('together_api_key', '').strip()
        together_active = 'together_active' in request.form
        
        if together_api_key:
            # Get or create Together AI key
            together_key = session.query(ExtractorAPIKey).filter_by(extractor_name='together').first()
            
            if together_key:
                together_key.api_key = together_api_key
                together_key.is_active = together_active
                together_key.updated_at = datetime.utcnow()
            else:
                together_key = ExtractorAPIKey(
                    extractor_name='together',
                    api_key=together_api_key,
                    is_active=together_active
                )
                session.add(together_key)
        else:
            # Delete if empty
            together_key = session.query(ExtractorAPIKey).filter_by(extractor_name='together').first()
            if together_key:
                session.delete(together_key)
        
        # Handle fal.ai authentication
        wos_session = request.form.get('wos_session', '').strip()
        fal_active = 'fal_active' in request.form
        
        if wos_session:
            # Get or create fal.ai auth
            fal_auth = session.query(AuthSettings).filter_by(source_name='fal.ai').first()
            
            cookies_dict = {'wos-session': wos_session}
            cookies_json = json.dumps(cookies_dict)
            
            if fal_auth:
                fal_auth.cookies = cookies_json
                fal_auth.is_active = fal_active
                fal_auth.updated_at = datetime.utcnow()
            else:
                fal_auth = AuthSettings(
                    source_name='fal.ai',
                    cookies=cookies_json,
                    is_active=fal_active
                )
                session.add(fal_auth)
        else:
            # Delete if empty
            fal_auth = session.query(AuthSettings).filter_by(source_name='fal.ai').first()
            if fal_auth:
                session.delete(fal_auth)
        
        session.commit()
        flash('Extractor settings saved successfully!', 'success')
        
    except Exception as e:
        session.rollback()
        flash(f'Error saving extractor settings: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('settings'))


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


@app.route('/auth-settings/save', methods=['POST'])
def save_auth_settings():
    """Add or update authentication settings"""
    import json
    session = get_session()
    
    try:
        source_name = request.form.get('source_name', 'fal.ai')
        notes = request.form.get('notes', '').strip()
        is_active = 'is_active' in request.form
        
        # Build cookies JSON from individual fields
        cookies_dict = {}
        
        wos_session = request.form.get('wos_session', '').strip()
        if wos_session:
            cookies_dict['wos-session'] = wos_session
        
        cf_clearance = request.form.get('cf_clearance', '').strip()
        if cf_clearance:
            cookies_dict['cf_clearance'] = cf_clearance
        
        unify_session_id = request.form.get('unify_session_id', '').strip()
        if unify_session_id:
            cookies_dict['unify_session_id'] = unify_session_id
        
        if not cookies_dict:
            flash('At least one cookie is required (wos-session recommended)', 'error')
            return redirect(url_for('auth_settings'))
        
        cookies_json = json.dumps(cookies_dict)
        
        # Check if already exists
        auth_config = session.query(AuthSettings).filter_by(source_name=source_name).first()
        
        if auth_config:
            # Update existing
            auth_config.cookies = cookies_json
            auth_config.headers = None  # Not used in simplified version
            auth_config.session_data = None
            auth_config.notes = notes or None
            auth_config.is_active = is_active
            auth_config.updated_at = datetime.utcnow()
            flash(f'Authentication for {source_name} updated!', 'success')
        else:
            # Create new
            auth_config = AuthSettings(
                source_name=source_name,
                cookies=cookies_json,
                headers=None,
                session_data=None,
                notes=notes or None,
                is_active=is_active
            )
            session.add(auth_config)
            flash(f'Authentication for {source_name} added!', 'success')
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        flash(f'Error saving authentication: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('settings'))


@app.route('/auth-settings/<int:auth_id>/toggle', methods=['POST'])
def toggle_auth_settings(auth_id):
    """Toggle authentication settings active status"""
    session = get_session()
    try:
        auth_config = session.query(AuthSettings).get(auth_id)
        if auth_config:
            auth_config.is_active = not auth_config.is_active
            auth_config.updated_at = datetime.utcnow()
            session.commit()
            status = 'enabled' if auth_config.is_active else 'disabled'
            flash(f'Authentication for {auth_config.source_name} {status}!', 'success')
        else:
            flash('Authentication settings not found', 'error')
    except Exception as e:
        session.rollback()
        flash(f'Error toggling authentication: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('settings'))


@app.route('/auth-settings/<int:auth_id>/delete', methods=['POST'])
def delete_auth_settings(auth_id):
    """Delete authentication settings"""
    session = get_session()
    try:
        auth_config = session.query(AuthSettings).get(auth_id)
        if auth_config:
            session.delete(auth_config)
            session.commit()
            flash('Authentication settings deleted!', 'success')
        else:
            flash('Authentication settings not found', 'error')
    except Exception as e:
        session.rollback()
        flash(f'Error deleting authentication: {e}', 'error')
    finally:
        close_session()
    
    return redirect(url_for('settings'))


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
            'tags': model.tags,
            'raw_metadata': model.raw_metadata,
            'input_schema': model.input_schema,
            'output_schema': model.output_schema,
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
    from ai_cost_manager.cache import cache_manager, CACHE_BACKEND, CACHE_DIR
    
    stats = cache_manager.get_cache_stats()
    
    return render_template('cache.html', 
                         cache_stats=stats,
                         cache_backend=CACHE_BACKEND,
                         cache_dir=CACHE_DIR)


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


@app.route('/api/extraction-progress/<int:source_id>', methods=['GET'])
def get_extraction_progress(source_id):
    """Get real-time extraction progress for a source"""
    try:
        from ai_cost_manager.progress_tracker import ProgressTracker
        
        progress = ProgressTracker.load(source_id)
        
        if not progress:
            return jsonify({'error': 'No active extraction'}), 404
        
        return jsonify(progress)
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


@app.route('/sources/<int:source_id>/re-extract', methods=['POST'])
def re_extract_missing(source_id):
    """Re-extract missing data for models from a source"""
    from threading import Thread
    from ai_cost_manager.progress_tracker import ProgressTracker
    
    session = get_session()
    
    try:
        source = session.query(APISource).filter_by(id=source_id).first()
        if not source:
            return jsonify({'error': 'Source not found'}), 404
        
        missing_type = request.args.get('missing_type', 'cost')
        
        # Get models with missing data
        query = session.query(AIModel).filter_by(source_id=source_id)
        
        if missing_type == 'cost':
            query = query.filter((AIModel.cost_per_call == 0.0) | (AIModel.cost_per_call.is_(None)))
        elif missing_type == 'schema':
            query = query.filter(
                (AIModel.input_schema.is_(None)) & (AIModel.output_schema.is_(None))
            )
        elif missing_type == 'llm':
            query = query.filter(AIModel.llm_extracted.is_(None))
        elif missing_type == 'raw':
            query = query.filter(AIModel.raw_metadata.is_(None))
        
        models_to_update = query.all()
        
        if not models_to_update:
            return jsonify({
                'success': True,
                'message': 'No models need re-extraction',
                'total': 0
            })
        
        # Create progress tracker
        progress_tracker = ProgressTracker(
            source_id=source_id,
            total_models=len(models_to_update),
            source_name=source.name
        )
        progress_tracker.save()
        
        # Start re-extraction in background thread
        def run_re_extraction():
            from ai_cost_manager.database import get_session, close_session
            thread_session = get_session()
            
            try:
                # Get extractor
                extractor_class = get_extractor(source.extractor_name)
                if not extractor_class:
                    progress_tracker.error(f'Unknown extractor: {source.extractor_name}')
                    progress_tracker.save()
                    return
                
                # Initialize extractor with re-extraction mode
                extractor = extractor_class(
                    force_refresh=True,  # Force refresh to re-fetch data
                    use_llm=True  # Enable LLM extraction
                )
                
                updated_count = 0
                
                for model in models_to_update:
                    try:
                        progress_tracker.set_current_model(model.name)
                        progress_tracker.save()
                        
                        # Re-extract model data
                        model_data = extractor.extract_model(model.model_id)
                        
                        if model_data:
                            # Update model with new data
                            for key, value in model_data.items():
                                if hasattr(AIModel, key) and key != 'id':
                                    setattr(model, key, value)
                            
                            updated_count += 1
                            progress_tracker.increment_updated()
                        
                        progress_tracker.increment_processed()
                        progress_tracker.save()
                        
                    except Exception as e:
                        print(f"Error re-extracting {model.name}: {e}")
                        progress_tracker.increment_processed()
                        progress_tracker.save()
                        continue
                
                thread_session.commit()
                progress_tracker.complete(
                    success=True,
                    message=f'Re-extracted {updated_count} models'
                )
                progress_tracker.save()
                
            except Exception as e:
                thread_session.rollback()
                progress_tracker.error(str(e))
                progress_tracker.save()
            finally:
                close_session()
        
        # Start background thread
        thread = Thread(target=run_re_extraction)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'total': len(models_to_update),
            'missing_type': missing_type
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/re-extract-progress/<int:source_id>')
def re_extract_progress(source_id):
    """Get re-extraction progress"""
    try:
        from ai_cost_manager.progress_tracker import ProgressTracker
        
        progress = ProgressTracker.load(source_id)
        
        if not progress:
            return jsonify({'error': 'No active re-extraction'}), 404
        
        return jsonify(progress)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/models/<int:model_id>/refetch', methods=['POST'])
def refetch_model(model_id):
    """Re-fetch all data for a single model"""
    session = get_session()
    
    try:
        model = session.query(AIModel).filter_by(id=model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        source = model.source
        
        # Clear cache for this specific model to force fresh fetch
        from ai_cost_manager.cache import cache_manager
        cache_manager.clear_model_cache(source.name, model.model_id)
        
        extractor_class = get_extractor(source.extractor_name)
        
        if not extractor_class:
            return jsonify({'error': f'Extractor not found: {source.extractor_name}'}), 400
        
        # Initialize extractor with schema fetching enabled for this single model
        extractor = extractor_class(
            source_url=source.url,
            fetch_schemas=True,  # Enable schema fetching - only fetches for THIS model
            use_llm=True
        )
        extractor.force_refresh = True
        
        # Use efficient extract_model() method that only fetches data for ONE model
        # This fetches the model list (fast), finds the target, then fetches schema only for it
        updated_model_data = extractor.extract_model(model.model_id)
        
        if not updated_model_data:
            return jsonify({'error': 'Model not found in source'}), 404
        
        # Ensure model is in the session (needed because of scoped_session)
        if model not in session:
            model = session.merge(model)
        
        # Update model fields
        model.name = updated_model_data.get('name', model.name)
        model.description = updated_model_data.get('description', model.description)
        model.model_type = updated_model_data.get('model_type', model.model_type)
        model.cost_per_call = updated_model_data.get('cost_per_call', 0.0)
        model.credits_required = updated_model_data.get('credits_required')
        model.pricing_info = updated_model_data.get('pricing_info', '')
        model.thumbnail_url = updated_model_data.get('thumbnail_url', '')
        model.tags = updated_model_data.get('tags', [])
        model.category = updated_model_data.get('category')
        model.input_schema = updated_model_data.get('input_schema')
        model.output_schema = updated_model_data.get('output_schema')
        model.raw_metadata = updated_model_data.get('raw_metadata')
        model.pricing_type = updated_model_data.get('pricing_type')
        model.pricing_formula = updated_model_data.get('pricing_formula')
        model.pricing_variables = updated_model_data.get('pricing_variables')
        model.input_cost_per_unit = updated_model_data.get('input_cost_per_unit')
        model.output_cost_per_unit = updated_model_data.get('output_cost_per_unit')
        model.cost_unit = updated_model_data.get('cost_unit')
        model.llm_extracted = updated_model_data.get('llm_extracted')
        model.last_raw_fetched = updated_model_data.get('last_raw_fetched')
        model.last_schema_fetched = updated_model_data.get('last_schema_fetched')
        model.last_llm_fetched = updated_model_data.get('last_llm_fetched')
        model.updated_at = datetime.utcnow()
        
        session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/models/refetch-filtered', methods=['POST'])
def refetch_filtered():
    """Re-fetch data for all models matching current filters"""
    session = get_session()
    
    try:
        # Get filter parameters
        search = request.args.get('search', '').strip()
        source_id = request.args.get('source_id', type=int)
        model_type = request.args.get('type', '').strip()
        missing_data = request.args.get('missing_data', '').strip()
        
        # Build query with same logic as models() route
        query = session.query(AIModel).filter_by(is_active=True)
        
        if search:
            query = query.filter(
                (AIModel.name.ilike(f'%{search}%')) |
                (AIModel.model_id.ilike(f'%{search}%')) |
                (AIModel.description.ilike(f'%{search}%'))
            )
        
        if source_id:
            query = query.filter_by(source_id=source_id)
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        if missing_data:
            if missing_data == 'cost':
                query = query.filter((AIModel.cost_per_call == None) | (AIModel.cost_per_call == 0))
            elif missing_data == 'llm':
                query = query.filter(AIModel.llm_extracted == None)
            elif missing_data == 'schema':
                query = query.filter((AIModel.input_schema == None) & (AIModel.output_schema == None))
            elif missing_data == 'raw':
                query = query.filter(AIModel.raw_metadata == None)
        
        models = query.all()
        
        if not models:
            return jsonify({'error': 'No models match filters'}), 400
        
        # Group models by source for efficient extraction
        models_by_source = {}
        for model in models:
            source_id = model.source_id
            if source_id not in models_by_source:
                models_by_source[source_id] = []
            models_by_source[source_id].append(model)
        
        updated_count = 0
        
        # Re-fetch each source's models
        for source_id, source_models in models_by_source.items():
            source = session.query(APISource).filter_by(id=source_id).first()
            if not source:
                continue
            
            extractor_class = get_extractor(source.extractor_name)
            if not extractor_class:
                continue
            
            try:
                extractor = extractor_class(
                    source_url=source.url,
                    fetch_schemas=True,
                    use_llm=True
                )
                extractor.force_refresh = True
                
                all_models = extractor.extract()
                model_data_map = {m['model_id']: m for m in all_models}
                
                # Update each model
                for model in source_models:
                    updated_data = model_data_map.get(model.model_id)
                    if not updated_data:
                        continue
                    
                    # Update fields
                    model.name = updated_data.get('name', model.name)
                    model.description = updated_data.get('description', model.description)
                    model.model_type = updated_data.get('model_type', model.model_type)
                    model.cost_per_call = updated_data.get('cost_per_call', 0.0)
                    model.credits_required = updated_data.get('credits_required')
                    model.pricing_info = updated_data.get('pricing_info', '')
                    model.thumbnail_url = updated_data.get('thumbnail_url', '')
                    model.tags = updated_data.get('tags', [])
                    model.category = updated_data.get('category')
                    model.input_schema = updated_data.get('input_schema')
                    model.output_schema = updated_data.get('output_schema')
                    model.raw_metadata = updated_data.get('raw_metadata')
                    model.pricing_type = updated_data.get('pricing_type')
                    model.pricing_formula = updated_data.get('pricing_formula')
                    model.pricing_variables = updated_data.get('pricing_variables')
                    model.input_cost_per_unit = updated_data.get('input_cost_per_unit')
                    model.output_cost_per_unit = updated_data.get('output_cost_per_unit')
                    model.cost_unit = updated_data.get('cost_unit')
                    model.llm_extracted = updated_data.get('llm_extracted')
                    model.last_raw_fetched = updated_data.get('last_raw_fetched')
                    model.last_schema_fetched = updated_data.get('last_schema_fetched')
                    model.last_llm_fetched = updated_data.get('last_llm_fetched')
                    model.updated_at = datetime.utcnow()
                    
                    updated_count += 1
                
            except Exception as e:
                print(f"Error re-fetching from {source.name}: {e}")
                continue
        
        session.commit()
        
        return jsonify({
            'success': True,
            'count': updated_count
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
