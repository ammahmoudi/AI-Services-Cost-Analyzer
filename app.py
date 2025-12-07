"""
AI Cost Manager - Web Application

Simple Flask web interface for viewing and managing AI model costs.
"""
import os
import json
import time
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv
from ai_cost_manager.database import get_session, close_session, init_db
from ai_cost_manager.models import APISource, AIModel, LLMConfiguration, AuthSettings, ExtractorAPIKey, ExtractionTask
from ai_cost_manager.openrouter_client import fetch_openrouter_models, get_recommended_models
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


def apply_llm_extraction_to_model(model, llm_result, session, source_name=None):
    """
    Apply LLM extraction results to a model instance.
    This is the single source of truth for updating models from LLM extraction.
    
    Args:
        model: AIModel instance
        llm_result: Dictionary with extraction results
        session: SQLAlchemy session
        source_name: Optional source name for cache (if None, will try to get from model.source)
        
    Returns:
        bool: True if any changes were made
    """
    from ai_cost_manager.model_types import normalize_model_type
    from sqlalchemy.orm import attributes as sa_attributes
    from ai_cost_manager.cache import cache_manager
    
    changes_made = False
    original_type = model.model_type
    
    # Update model_type
    extracted_type = llm_result.get('model_type')
    if extracted_type:
        normalized_type = normalize_model_type(extracted_type)
        
        if normalized_type != original_type:
            model.model_type = normalized_type
            sa_attributes.flag_modified(model, 'model_type')
            print(f"  Type: {original_type} → {normalized_type}")
            changes_made = True
    
    # Update category
    extracted_category = llm_result.get('category')
    if extracted_category and str(extracted_category).lower() != 'none':
        if extracted_category != model.category:
            model.category = extracted_category
            print(f"  Category: {model.category} → {extracted_category}")
            changes_made = True
    
    # Update tags
    extracted_tags = llm_result.get('tags')
    if extracted_tags and extracted_tags != model.tags:
        model.tags = extracted_tags
        print(f"  Tags: {model.tags} → {extracted_tags}")
        changes_made = True
    
    # Update description
    extracted_description = llm_result.get('description')
    if extracted_description and extracted_description != model.description:
        model.description = extracted_description
        changes_made = True
    
    # Update pricing fields
    extracted_cost = llm_result.get('cost_per_call')
    if extracted_cost is not None and extracted_cost != 0:
        model.cost_per_call = extracted_cost
        changes_made = True
    
    model.pricing_type = llm_result.get('pricing_type')
    model.pricing_formula = llm_result.get('pricing_formula')
    model.pricing_variables = llm_result.get('pricing_variables')
    model.input_cost_per_unit = llm_result.get('input_cost_per_unit')
    model.output_cost_per_unit = llm_result.get('output_cost_per_unit')
    model.cost_unit = llm_result.get('cost_unit')
    
    # Store full LLM extraction result
    model.llm_extracted = llm_result
    model.last_llm_fetched = datetime.utcnow()
    
    # Save to cache if source_name provided
    if source_name:
        cache_manager.save_llm_extraction(source_name, model.model_id, llm_result)
        print(f"  💾 Saved LLM extraction to cache ({source_name})")
    
    return changes_made


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
    """Start a background extraction task for a specific source"""
    from threading import Thread
    from ai_cost_manager.models import ExtractionTask
    from ai_cost_manager.progress_tracker import ProgressTracker
    
    session = get_session()
    
    try:
        source = session.query(APISource).filter_by(id=source_id).first()
        if not source:
            return jsonify({'error': 'Source not found'}), 404
        
        # Get options from request
        use_llm = request.json.get('use_llm', False) if request.is_json else False
        fetch_schemas = request.json.get('fetch_schemas', False) if request.is_json else False
        force_refresh = request.json.get('force_refresh', False) if request.is_json else False
        
        # Check if there's already a running task for this source
        existing_task = session.query(ExtractionTask).filter(
            ExtractionTask.source_id == source_id,
            ExtractionTask.status.in_(['pending', 'running'])
        ).first()
        
        if existing_task:
            return jsonify({
                'task_id': existing_task.id,
                'status': existing_task.status,
                'message': 'Extraction already in progress',
                'already_running': True
            })
        
        # Create new extraction task
        task = ExtractionTask(
            source_id=source_id,
            status='pending',
            use_llm=use_llm,
            fetch_schemas=fetch_schemas,
            force_refresh=force_refresh,
            started_at=datetime.utcnow()
        )
        session.add(task)
        session.commit()
        task_id = task.id
        
        # Start extraction in background thread
        def run_extraction():
            from ai_cost_manager.database import get_session, close_session
            thread_session = get_session()
            
            try:
                # Get task and source in this thread's session
                task = thread_session.query(ExtractionTask).filter_by(id=task_id).first()
                source = thread_session.query(APISource).filter_by(id=source_id).first()
                
                if not task or not source:
                    return
                
                # Update task status to running
                task.status = 'running'
                task.progress = 0
                thread_session.commit()
                
                # Store source attributes we need
                source_name = source.name
                source_url = source.url
                source_extractor_name = source.extractor_name
                source_db_id = source.id
                
                # Create progress tracker
                progress_tracker = ProgressTracker(source_name, source_db_id)
                
                # Get extractor class and instantiate it
                extractor_class = get_extractor(source_extractor_name)
                extractor = extractor_class(
                    source_url=source_url,
                    fetch_schemas=fetch_schemas,
                    use_llm=use_llm
                )
                
                if force_refresh and hasattr(extractor, 'force_refresh'):
                    extractor.force_refresh = True
                
                # Extract models with progress tracking
                models_data = extractor.extract(progress_tracker=progress_tracker)
                
                if not models_data:
                    task.status = 'failed'
                    task.error_message = 'No models extracted - API may be unavailable'
                    task.completed_at = datetime.utcnow()
                    thread_session.commit()
                    return
                
                task.total_models = len(models_data)
                task.status = 'running'  # Mark task as running
                thread_session.commit()
                
                # Save models incrementally
                new_count = 0
                updated_count = 0
                batch_size = 10
                batch_count = 0
                
                for idx, model_data in enumerate(models_data):
                    try:
                        # Re-query task to check cancellation (avoid detached instance issues)
                        current_task = thread_session.query(ExtractionTask).filter_by(id=task_id).first()
                        if not current_task or current_task.status == 'cancelled':
                            if current_task:
                                current_task.error_message = 'Extraction cancelled by user'
                                current_task.completed_at = datetime.utcnow()
                                thread_session.commit()
                            return
                        
                        filtered_data = {
                            k: v for k, v in model_data.items()
                            if hasattr(AIModel, k)
                        }
                        
                        existing_model = thread_session.query(AIModel).filter_by(
                            source_id=source_db_id,
                            model_id=model_data['model_id']
                        ).first()
                        
                        if existing_model:
                            for key, value in filtered_data.items():
                                if key != 'id':
                                    setattr(existing_model, key, value)
                            updated_count += 1
                        else:
                            model = AIModel(source_id=source_db_id, **filtered_data)
                            thread_session.add(model)
                            new_count += 1
                        
                        batch_count += 1
                        
                        # Update task progress (use current_task to avoid detached issues)
                        current_task.processed_models = idx + 1
                        current_task.new_models = new_count
                        current_task.updated_models = updated_count
                        current_task.current_model = model_data.get('name', 'Unknown')
                        current_task.progress = int((idx + 1) / len(models_data) * 100)
                        
                        # Commit progress every model (lightweight update)
                        thread_session.commit()
                        
                        # Reset batch counter when we hit batch size
                        if batch_count >= batch_size:
                            batch_count = 0
                            
                    except Exception as model_error:
                        print(f"Warning: Error saving model {model_data.get('model_id', 'unknown')}: {str(model_error)}")
                        thread_session.rollback()
                        batch_count = 0
                        # Re-query both task and current_task after rollback
                        task = thread_session.query(ExtractionTask).filter_by(id=task_id).first()
                        current_task = task
                        continue
                
                # Final commit (in case last model wasn't committed yet)
                thread_session.commit()
                
                # Re-query task and source for final update
                task = thread_session.query(ExtractionTask).filter_by(id=task_id).first()
                source = thread_session.query(APISource).filter_by(id=source_db_id).first()
                
                if source:
                    source.last_extracted = datetime.utcnow()
                
                # Mark task as completed
                if task:
                    task.status = 'completed'
                    task.progress = 100
                    task.completed_at = datetime.utcnow()
                    thread_session.commit()
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"❌ Extraction error: {error_trace}")
                
                # Re-query task for error update
                try:
                    task = thread_session.query(ExtractionTask).filter_by(id=task_id).first()
                    if task:
                        task.status = 'failed'
                        task.error_message = str(e)
                        task.completed_at = datetime.utcnow()
                        thread_session.commit()
                except Exception as inner_error:
                    print(f"❌ Failed to update task error: {inner_error}")
            finally:
                close_session()
        
        # Start background thread
        thread = Thread(target=run_extraction, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'status': 'pending',
            'message': 'Extraction started in background'
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/models')
def models():
    """List all AI models"""
    from ai_cost_manager.model_types import VALID_MODEL_TYPES
    
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
        
        # Use VALID_MODEL_TYPES for filter dropdown (shows all available types)
        model_types = VALID_MODEL_TYPES
        
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


@app.route('/canonical-models')
def canonical_models_page():
    """Show unified view of models grouped by canonical name"""
    return render_template('canonical_models.html')


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
        
        # Get Runware authentication
        runware_auth = session.query(AuthSettings).filter_by(source_name='runware').first()
        
        # Fetch available models with free models first
        available_models = fetch_openrouter_models(sort_by_free=True)
        recommended = get_recommended_models()
        
        return render_template('settings.html',
                             llm_config=llm_config,
                             together_key=together_key,
                             fal_auth=fal_auth,
                             runware_auth=runware_auth,
                             available_models=available_models,
                             recommended_models=recommended)
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
        
        # Handle Runware authentication
        runware_username = request.form.get('runware_username', '').strip()
        runware_password = request.form.get('runware_password', '').strip()
        runware_active = 'runware_active' in request.form
        
        if runware_username and runware_password:
            # Get or create Runware auth
            runware_auth = session.query(AuthSettings).filter_by(source_name='runware').first()
            
            if runware_auth:
                runware_auth.username = runware_username
                runware_auth.password = runware_password
                runware_auth.is_active = runware_active
                runware_auth.updated_at = datetime.utcnow()
            else:
                runware_auth = AuthSettings(
                    source_name='runware',
                    username=runware_username,
                    password=runware_password,
                    is_active=runware_active
                )
                session.add(runware_auth)
        elif runware_username or runware_password:
            # Partial credentials - show warning
            flash('Both Runware email and password are required', 'warning')
        else:
            # Delete if both empty
            runware_auth = session.query(AuthSettings).filter_by(source_name='runware').first()
            if runware_auth:
                session.delete(runware_auth)
        
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
        
        # Check if this is credential-based auth (Runware) or cookie-based (fal.ai)
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        cookies_json = None
        
        if username and password:
            # Credential-based authentication (e.g., Runware)
            print(f"Saving credential-based auth for {source_name}: {username}")
        else:
            # Cookie-based authentication (e.g., fal.ai)
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
                flash('Either credentials or cookies are required', 'error')
                return redirect(url_for('auth_settings'))
            
            cookies_json = json.dumps(cookies_dict)
        
        # Check if already exists
        auth_config = session.query(AuthSettings).filter_by(source_name=source_name).first()
        
        if auth_config:
            # Update existing
            if username and password:
                auth_config.username = username
                auth_config.password = password
                auth_config.cookies = None
            else:
                auth_config.cookies = cookies_json
                auth_config.username = None
                auth_config.password = None
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
                username=username or None,
                password=password or None,
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
        
        # Get source name while model is still bound to session
        source_name = model.source.name if model.source else 'unknown'
        
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
            # Use shared function to apply LLM extraction
            print(f"Applying LLM extraction to {model.name}...")
            changes_made = apply_llm_extraction_to_model(model, pricing_details, session, source_name)
            
            if changes_made:
                print(f"  ✅ Updated {model.name}")
            
            session.flush()
            session.commit()
            print(f"💾 Committed changes for model ID {model.id}")
            
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


@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the status of an extraction task"""
    from ai_cost_manager.models import ExtractionTask
    
    session = get_session()
    try:
        task = session.query(ExtractionTask).filter_by(id=task_id).first()
        
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify({
            'task_id': task.id,
            'source_id': task.source_id,
            'source_name': task.source.name if task.source else 'Unknown',
            'status': task.status,
            'progress': task.progress,
            'total_models': task.total_models,
            'processed_models': task.processed_models,
            'new_models': task.new_models,
            'updated_models': task.updated_models,
            'current_model': task.current_model,
            'error_message': task.error_message,
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/sources/<int:source_id>/active-task', methods=['GET'])
def get_source_active_task(source_id):
    """Get any active extraction task for a source"""
    from ai_cost_manager.models import ExtractionTask
    
    session = get_session()
    try:
        # Look for any pending or running task for this source
        task = session.query(ExtractionTask).filter(
            ExtractionTask.source_id == source_id,
            ExtractionTask.status.in_(['pending', 'running'])
        ).order_by(ExtractionTask.started_at.desc()).first()
        
        if not task:
            return jsonify({'active': False})
        
        return jsonify({
            'active': True,
            'task_id': task.id,
            'status': task.status,
            'progress': task.progress,
            'started_at': task.started_at.isoformat() if task.started_at else None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/tasks/<int:task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running extraction task"""
    from ai_cost_manager.models import ExtractionTask
    
    session = get_session()
    try:
        task = session.query(ExtractionTask).filter_by(id=task_id).first()
        
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        if task.status in ['completed', 'failed', 'cancelled']:
            return jsonify({
                'success': False,
                'message': f'Task already {task.status}'
            })
        
        # Mark task as cancelled (the background thread will check this)
        task.status = 'cancelled'
        task.completed_at = datetime.utcnow()
        session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Task cancelled successfully'
        })
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


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
                batch_size = 10  # Commit every 10 models
                batch_count = 0
                
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
                        
                        batch_count += 1
                        
                        # Commit in batches to prevent data loss
                        if batch_count >= batch_size:
                            thread_session.commit()
                            batch_count = 0
                        
                    except Exception as e:
                        print(f"Error re-extracting {model.name}: {e}")
                        progress_tracker.increment_processed()
                        progress_tracker.save()
                        thread_session.rollback()
                        batch_count = 0
                        continue
                
                # Commit any remaining models
                if batch_count > 0:
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


@app.route('/api/run-llm-extraction', methods=['POST'])
def run_llm_extraction():
    """Run LLM extraction on models to fix types, tags, and metadata"""
    from threading import Thread
    from ai_cost_manager.llm_extractor import LLMPricingExtractor
    from ai_cost_manager.progress_tracker import ProgressTracker
    
    session = get_session()
    
    try:
        data = request.get_json() or {}
        source_id = data.get('source_id')
        model_type = data.get('model_type')
        missing_data = data.get('missing_data')  # 'llm', 'all', etc.
        
        # Build query to count models
        query = session.query(AIModel).filter(AIModel.is_active == True)
        
        if source_id:
            query = query.filter_by(source_id=source_id)
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        if missing_data == 'llm':
            query = query.filter(AIModel.llm_extracted.is_(None))
        
        model_count = query.count()
        
        if model_count == 0:
            return jsonify({
                'success': True,
                'message': 'No models match filters',
                'total': 0
            })
        
        # Store filter parameters to rebuild query in thread
        filter_params = {
            'source_id': source_id,
            'model_type': model_type,
            'missing_data': missing_data
        }
        
        # Use source_id 9999 as a special ID for LLM extraction progress
        progress_source_id = 9999
        
        # Create progress tracker
        source_name = "All Sources" if not source_id else session.query(APISource).get(source_id).name
        progress_tracker = ProgressTracker(
            source_name=f"LLM Extraction - {source_name}",
            source_id=progress_source_id
        )
        progress_tracker.start(total_models=model_count, options={'use_llm': True})
        
        # Start LLM extraction in background thread
        def run_llm_thread():
            from ai_cost_manager.database import get_session, close_session
            thread_session = get_session()
            
            try:
                # Rebuild query in thread session to avoid detached instances
                thread_query = thread_session.query(AIModel).filter(AIModel.is_active == True)
                
                if filter_params['source_id']:
                    thread_query = thread_query.filter_by(source_id=filter_params['source_id'])
                
                if filter_params['model_type']:
                    thread_query = thread_query.filter_by(model_type=filter_params['model_type'])
                
                if filter_params['missing_data'] == 'llm':
                    thread_query = thread_query.filter(AIModel.llm_extracted.is_(None))
                
                models = thread_query.all()
                model_ids = [m.id for m in models]  # Store only IDs
                
                llm_extractor = LLMPricingExtractor()
                
                updated_count = 0
                
                for i, model_id in enumerate(model_ids, 1):
                    try:
                        # Re-query model fresh in this thread's session EVERY iteration
                        # This ensures SQLAlchemy properly tracks changes
                        model = thread_session.query(AIModel).filter_by(id=model_id).first()
                        if not model:
                            continue
                        
                        # Get source name while model is bound to session
                        source_name = model.source.name if model.source else 'unknown'
                        
                        # Store original values before extraction
                        original_type = model.model_type
                        
                        # Update progress
                        progress_tracker.update(
                            processed=i,
                            current_model_id=str(model.model_id),
                            current_model_name=model.name
                        )
                        
                        # Prepare model data for LLM extraction
                        model_data = {
                            'name': model.name,
                            'model_id': model.model_id,
                            'description': model.description,
                            'pricing_info': model.pricing_formula or '',
                            'creditsRequired': model.credits_required or 0,
                            'model_type': model.model_type,
                            'category': model.category,
                            'tags': model.tags or [],
                            'raw_metadata': model.raw_metadata or {},
                            'input_schema': model.input_schema or {},
                            'output_schema': model.output_schema or {},
                        }
                        
                        # Run LLM extraction with error handling
                        try:
                            llm_result = llm_extractor.extract_pricing(model_data)
                        except Exception as llm_error:
                            print(f"  LLM extraction failed for {model.name}: {llm_error}")
                            llm_result = None
                        
                        # Brief delay to avoid rate limiting
                        time.sleep(0.5)
                        
                        # Store name before session operations that might detach the model
                        model_name = model.name
                        model_db_id = model.id
                        
                        if llm_result:
                            # Prepare extraction info for UI
                            from ai_cost_manager.model_types import normalize_model_type
                            extracted_type = llm_result.get('model_type')
                            normalized_type = normalize_model_type(extracted_type) if extracted_type else 'other'
                            
                            # Update progress with extraction details
                            progress_tracker.update(
                                processed=i,
                                current_model_id=str(model.model_id),
                                current_model_name=model_name,
                                current_extraction={
                                    'original_type': original_type,
                                    'extracted_type': normalized_type,
                                    'extracted_category': llm_result.get('category'),
                                    'extracted_tags': llm_result.get('tags', [])
                                }
                            )
                            
                            # Use shared function to apply LLM extraction
                            print(f"Processing {model_name}...")
                            changes_made = apply_llm_extraction_to_model(model, llm_result, thread_session, source_name)
                            
                            if changes_made:
                                updated_count += 1
                                progress_tracker.increment_updated()
                                print(f"  ✅ Updated {model_name}")
                        
                        # Commit IMMEDIATELY after each model to ensure persistence
                        thread_session.flush()
                        thread_session.commit()
                        print(f"    💾 Committed model ID {model_db_id}")
                        
                    except Exception as e:
                        print(f"Error processing model ID {model_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        progress_tracker.update(
                            processed=i,
                            current_model_id=str(model_id),
                            current_model_name="unknown",
                            has_error=True,
                            error_message=str(e)
                        )
                        thread_session.rollback()
                        # Continue with next model instead of stopping
                
                # Final summary
                print("\n" + "="*60)
                progress_tracker.complete()
                
                print(f"✅ LLM extraction completed: {updated_count}/{len(model_ids)} models updated")
                
            except Exception as e:
                print(f"❌ LLM extraction failed: {e}")
                progress_tracker.error(str(e))
                thread_session.rollback()
            finally:
                close_session()
        
        # Start thread
        thread = Thread(target=run_llm_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Started LLM extraction for {model_count} models',
            'total': model_count,
            'progress_id': progress_source_id
        })
        
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/llm-extraction-progress')
def llm_extraction_progress():
    """Get LLM extraction progress"""
    try:
        from ai_cost_manager.progress_tracker import ProgressTracker
        
        progress = ProgressTracker.load(9999)  # Special ID for LLM extraction
        
        if not progress:
            return jsonify({'error': 'No active LLM extraction'}), 404
        
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


# ==============================================================================
# Model Matching API Endpoints
# ==============================================================================

@app.route('/api/match-models', methods=['POST'])
def match_models():
    """Run model matching across all providers"""
    from ai_cost_manager.model_matching_service import ModelMatchingService
    
    session = get_session()
    try:
        force_refresh = request.json.get('force_refresh', False) if request.is_json else False
        
        service = ModelMatchingService(session)
        result = service.match_all_models(force_refresh=force_refresh)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/models/<int:model_id>/alternatives', methods=['GET'])
def get_model_alternatives(model_id):
    """Get alternative providers for a specific model"""
    from ai_cost_manager.model_matching_service import ModelMatchingService
    
    session = get_session()
    try:
        service = ModelMatchingService(session)
        alternatives = service.get_alternatives(model_id)
        
        return jsonify({
            'model_id': model_id,
            'alternatives': alternatives
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/models/<int:model_id>/with-alternatives', methods=['GET'])
def get_model_with_alternatives(model_id):
    """Get a model with all its alternatives and pricing comparison"""
    from ai_cost_manager.model_matching_service import ModelMatchingService
    
    session = get_session()
    try:
        service = ModelMatchingService(session)
        result = service.get_model_with_alternatives(model_id)
        
        if not result:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/canonical-models', methods=['GET'])
def get_canonical_models():
    """Get all canonical models with provider details (paginated)"""
    try:
        from ai_cost_manager.model_matching_service import ModelMatchingService
    except ImportError as e:
        return jsonify({
            'error': f'Model matching feature not available: {str(e)}. Please install required dependencies: pip install openai',
            'models': []
        }), 500
    
    session = get_session()
    all_models = []
    
    try:
        model_type = request.args.get('model_type')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        service = ModelMatchingService(session)
        # Get all models and extract data before session closes
        all_models = service.get_canonical_models(model_type=model_type)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_canonical_models: {error_details}")
        return jsonify({'error': str(e), 'models': []}), 500
    finally:
        close_session()
    
    # Paginate after session is closed (data is already serialized)
    total = len(all_models)
    start = (page - 1) * per_page
    end = start + per_page
    models = all_models[start:end]
    
    return jsonify({
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': (total + per_page - 1) // per_page,
        'models': models
    })


@app.route('/api/canonical-models/<int:canonical_id>', methods=['GET'])
def get_canonical_model_detail(canonical_id):
    """Get details of a specific canonical model"""
    session = get_session()
    try:
        from ai_cost_manager.models import CanonicalModel, ModelMatch
        
        canonical = session.query(CanonicalModel).filter(
            CanonicalModel.id == canonical_id
        ).first()
        
        if not canonical:
            return jsonify({'error': 'Canonical model not found'}), 404
        
        # Get all matches
        matches = session.query(ModelMatch).filter(
            ModelMatch.canonical_model_id == canonical_id
        ).all()
        
        providers = []
        for match in matches:
            if match.ai_model and match.ai_model.is_active:
                providers.append({
                    'model_id': match.ai_model.id,
                    'name': match.ai_model.name,
                    'provider': match.ai_model.source.name if match.ai_model.source else 'unknown',
                    'cost_per_call': match.ai_model.cost_per_call,
                    'pricing_formula': match.ai_model.pricing_formula,
                    'description': match.ai_model.description,
                    'confidence': match.confidence,
                })
        
        # Sort by price
        providers.sort(key=lambda x: x['cost_per_call'] if x['cost_per_call'] else float('inf'))
        
        return jsonify({
            'canonical_id': canonical.id,
            'canonical_name': canonical.canonical_name,
            'display_name': canonical.display_name,
            'description': canonical.description,
            'model_type': canonical.model_type,
            'tags': canonical.tags,
            'provider_count': len(providers),
            'providers': providers,
            'best_price': providers[0] if providers else None,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


# ============================================================================
# CRUD API Endpoints for AI Models
# ============================================================================

@app.route('/api/ai-models', methods=['GET'])
def api_get_ai_models():
    """Get all AI models with filtering and pagination"""
    session = get_session()
    try:
        from ai_cost_manager.models import AIModel, APISource
        
        # Get query parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        search = request.args.get('search', '')
        source_id = request.args.get('source_id', type=int)
        model_type = request.args.get('type', '')
        category = request.args.get('category', '')
        
        # Build query
        query = session.query(AIModel).filter(AIModel.is_active == True)
        
        if search:
            query = query.filter(
                (AIModel.name.ilike(f'%{search}%')) | 
                (AIModel.model_id.ilike(f'%{search}%'))
            )
        if source_id:
            query = query.filter(AIModel.source_id == source_id)
        if model_type:
            query = query.filter(AIModel.model_type == model_type)
        if category:
            query = query.filter(AIModel.category == category)
        
        # Paginate
        total = query.count()
        models = query.order_by(AIModel.name).offset((page - 1) * per_page).limit(per_page).all()
        
        # Format response
        models_data = []
        for model in models:
            models_data.append({
                'id': model.id,
                'model_id': model.model_id,
                'name': model.name,
                'description': model.description,
                'model_type': model.model_type,
                'category': model.category,
                'cost_per_call': model.cost_per_call,
                'cost_per_1k_tokens': model.cost_per_1k_tokens,
                'credits_required': model.credits_required,
                'pricing_formula': model.pricing_formula,
                'tags': model.tags or [],
                'source': {
                    'id': model.source.id,
                    'name': model.source.name
                } if model.source else None,
                'has_llm_data': model.llm_extracted,
                'has_schema': bool(model.input_schema or model.output_schema),
                'has_raw_data': bool(model.raw_metadata),
            })
        
        return jsonify({
            'models': models_data,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/ai-models', methods=['POST'])
def api_create_ai_model():
    """Create a new AI model"""
    session = get_session()
    try:
        from ai_cost_manager.models import AIModel, APISource
        from ai_cost_manager.model_types import VALID_MODEL_TYPES
        
        data = request.get_json()
        
        # Validate required fields
        if not data.get('source_id'):
            return jsonify({'error': 'source_id is required'}), 400
        if not data.get('model_id'):
            return jsonify({'error': 'model_id is required'}), 400
        if not data.get('name'):
            return jsonify({'error': 'name is required'}), 400
        if not data.get('model_type'):
            return jsonify({'error': 'model_type is required'}), 400
        
        # Validate source exists
        source = session.query(APISource).filter(APISource.id == data['source_id']).first()
        if not source:
            return jsonify({'error': 'Source not found'}), 404
        
        # Validate model_type
        if data['model_type'] not in VALID_MODEL_TYPES:
            return jsonify({'error': f"Invalid model_type. Must be one of: {', '.join(VALID_MODEL_TYPES)}"}), 400
        
        # Check if model_id already exists for this source
        existing = session.query(AIModel).filter(
            AIModel.source_id == data['source_id'],
            AIModel.model_id == data['model_id']
        ).first()
        if existing:
            return jsonify({'error': 'A model with this model_id already exists for this source'}), 409
        
        # Parse tags
        tags = data.get('tags', [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        
        # Create new model
        model = AIModel(
            source_id=data['source_id'],
            model_id=data['model_id'],
            name=data['name'],
            description=data.get('description', ''),
            model_type=data['model_type'],
            category=data.get('category'),
            cost_per_call=float(data['cost_per_call']) if data.get('cost_per_call') else 0.0,
            cost_per_1k_tokens=float(data['cost_per_1k_tokens']) if data.get('cost_per_1k_tokens') else None,
            credits_required=float(data['credits_required']) if data.get('credits_required') else None,
            pricing_formula=data.get('pricing_formula', ''),
            tags=tags,
            is_active=True
        )
        
        session.add(model)
        session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Model created successfully',
            'id': model.id
        }), 201
    
    except Exception as e:
        session.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/ai-models/<int:model_id>', methods=['GET'])
def api_get_ai_model(model_id):
    """Get a specific AI model"""
    session = get_session()
    try:
        from ai_cost_manager.models import AIModel
        
        model = session.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({
            'id': model.id,
            'model_id': model.model_id,
            'name': model.name,
            'description': model.description,
            'model_type': model.model_type,
            'category': model.category,
            'cost_per_call': model.cost_per_call,
            'cost_per_1k_tokens': model.cost_per_1k_tokens,
            'credits_required': model.credits_required,
            'pricing_type': model.pricing_type,
            'pricing_formula': model.pricing_formula,
            'pricing_info': model.pricing_info,
            'tags': model.tags or [],
            'thumbnail_url': model.thumbnail_url,
            'source': {
                'id': model.source.id,
                'name': model.source.name
            } if model.source else None,
            'input_schema': model.input_schema,
            'output_schema': model.output_schema,
            'raw_metadata': model.raw_metadata,
            'llm_extracted': model.llm_extracted,
            'created_at': model.created_at.isoformat() if model.created_at else None,
            'updated_at': model.updated_at.isoformat() if model.updated_at else None,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/ai-models/<int:model_id>', methods=['PUT'])
def api_update_ai_model(model_id):
    """Update an AI model"""
    session = get_session()
    try:
        from ai_cost_manager.models import AIModel
        from ai_cost_manager.model_types import VALID_MODEL_TYPES
        
        model = session.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        data = request.get_json()
        
        # Update fields
        if 'name' in data:
            model.name = data['name']
        if 'description' in data:
            model.description = data['description']
        if 'model_type' in data:
            if data['model_type'] not in VALID_MODEL_TYPES:
                return jsonify({'error': f"Invalid model_type. Must be one of: {', '.join(VALID_MODEL_TYPES)}"}), 400
            model.model_type = data['model_type']
        if 'category' in data:
            model.category = data['category']
        if 'cost_per_call' in data:
            model.cost_per_call = float(data['cost_per_call'])
        if 'cost_per_1k_tokens' in data:
            model.cost_per_1k_tokens = float(data['cost_per_1k_tokens']) if data['cost_per_1k_tokens'] else None
        if 'credits_required' in data:
            model.credits_required = float(data['credits_required']) if data['credits_required'] else None
        if 'pricing_formula' in data:
            model.pricing_formula = data['pricing_formula']
        if 'tags' in data:
            model.tags = data['tags'] if isinstance(data['tags'], list) else []
        
        model.updated_at = datetime.utcnow()
        session.commit()
        
        return jsonify({'success': True, 'message': 'Model updated successfully'})
    
    except Exception as e:
        session.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/ai-models/<int:model_id>', methods=['DELETE'])
def api_delete_ai_model(model_id):
    """Delete an AI model"""
    session = get_session()
    try:
        from ai_cost_manager.models import AIModel
        
        model = session.query(AIModel).filter(AIModel.id == model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        model_name = model.name
        session.delete(model)
        session.commit()
        
        return jsonify({'success': True, 'message': f'Model "{model_name}" deleted successfully'})
    
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


# ============================================================================
# CRUD API Endpoints for Canonical Models
# ============================================================================

@app.route('/api/canonical-models', methods=['POST'])
def api_create_canonical_model():
    """Create a new canonical model"""
    session = get_session()
    try:
        from ai_cost_manager.models import CanonicalModel
        from ai_cost_manager.model_types import VALID_MODEL_TYPES
        
        data = request.get_json()
        
        # Validate required fields
        if not data.get('canonical_name'):
            return jsonify({'error': 'canonical_name is required'}), 400
        if not data.get('display_name'):
            return jsonify({'error': 'display_name is required'}), 400
        
        # Validate model_type
        model_type = data.get('model_type')
        if model_type and model_type not in VALID_MODEL_TYPES:
            return jsonify({'error': f"Invalid model_type. Must be one of: {', '.join(VALID_MODEL_TYPES)}"}), 400
        
        # Check if canonical_name already exists
        existing = session.query(CanonicalModel).filter(
            CanonicalModel.canonical_name == data['canonical_name']
        ).first()
        if existing:
            return jsonify({'error': 'A canonical model with this name already exists'}), 409
        
        # Create new canonical model
        canonical = CanonicalModel(
            canonical_name=data['canonical_name'],
            display_name=data['display_name'],
            description=data.get('description', ''),
            model_type=model_type,
            tags=data.get('tags', [])
        )
        
        session.add(canonical)
        session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Canonical model created successfully',
            'id': canonical.id
        }), 201
    
    except Exception as e:
        session.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/canonical-models/<int:canonical_id>', methods=['PUT'])
def api_update_canonical_model(canonical_id):
    """Update a canonical model"""
    session = get_session()
    try:
        from ai_cost_manager.models import CanonicalModel
        from ai_cost_manager.model_types import VALID_MODEL_TYPES
        
        canonical = session.query(CanonicalModel).filter(
            CanonicalModel.id == canonical_id
        ).first()
        
        if not canonical:
            return jsonify({'error': 'Canonical model not found'}), 404
        
        data = request.get_json()
        
        # Update fields
        if 'canonical_name' in data:
            # Check if new name conflicts with existing
            existing = session.query(CanonicalModel).filter(
                CanonicalModel.canonical_name == data['canonical_name'],
                CanonicalModel.id != canonical_id
            ).first()
            if existing:
                return jsonify({'error': 'A canonical model with this name already exists'}), 409
            canonical.canonical_name = data['canonical_name']
        
        if 'display_name' in data:
            canonical.display_name = data['display_name']
        if 'description' in data:
            canonical.description = data['description']
        if 'model_type' in data:
            if data['model_type'] and data['model_type'] not in VALID_MODEL_TYPES:
                return jsonify({'error': f"Invalid model_type. Must be one of: {', '.join(VALID_MODEL_TYPES)}"}), 400
            canonical.model_type = data['model_type']
        if 'tags' in data:
            canonical.tags = data['tags'] if isinstance(data['tags'], list) else []
        
        canonical.updated_at = datetime.utcnow()
        session.commit()
        
        return jsonify({'success': True, 'message': 'Canonical model updated successfully'})
    
    except Exception as e:
        session.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/canonical-models/<int:canonical_id>', methods=['DELETE'])
def api_delete_canonical_model(canonical_id):
    """Delete a canonical model and its matches"""
    session = get_session()
    try:
        from ai_cost_manager.models import CanonicalModel
        
        canonical = session.query(CanonicalModel).filter(
            CanonicalModel.id == canonical_id
        ).first()
        
        if not canonical:
            return jsonify({'error': 'Canonical model not found'}), 404
        
        canonical_name = canonical.canonical_name
        session.delete(canonical)  # Cascade will delete model matches
        session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Canonical model "{canonical_name}" deleted successfully'
        })
    
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/canonical-models/<int:canonical_id>/models', methods=['POST'])
def api_add_model_to_canonical(canonical_id):
    """Add an AI model to a canonical model"""
    session = get_session()
    try:
        from ai_cost_manager.models import CanonicalModel, ModelMatch, AIModel
        
        canonical = session.query(CanonicalModel).filter(
            CanonicalModel.id == canonical_id
        ).first()
        
        if not canonical:
            return jsonify({'error': 'Canonical model not found'}), 404
        
        data = request.get_json()
        ai_model_id = data.get('ai_model_id')
        
        if not ai_model_id:
            return jsonify({'error': 'ai_model_id is required'}), 400
        
        # Check if AI model exists
        ai_model = session.query(AIModel).filter(AIModel.id == ai_model_id).first()
        if not ai_model:
            return jsonify({'error': 'AI model not found'}), 404
        
        # Check if match already exists
        existing_match = session.query(ModelMatch).filter(
            ModelMatch.canonical_model_id == canonical_id,
            ModelMatch.ai_model_id == ai_model_id
        ).first()
        
        if existing_match:
            return jsonify({'error': 'This model is already linked to this canonical model'}), 409
        
        # Create match
        match = ModelMatch(
            canonical_model_id=canonical_id,
            ai_model_id=ai_model_id,
            confidence=data.get('confidence', 1.0),
            matched_by=data.get('matched_by', 'manual')
        )
        
        session.add(match)
        session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Model "{ai_model.name}" added to canonical model'
        }), 201
    
    except Exception as e:
        session.rollback()
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


@app.route('/api/canonical-models/<int:canonical_id>/models/<int:ai_model_id>', methods=['DELETE'])
def api_remove_model_from_canonical(canonical_id, ai_model_id):
    """Remove an AI model from a canonical model"""
    session = get_session()
    try:
        from ai_cost_manager.models import ModelMatch
        
        match = session.query(ModelMatch).filter(
            ModelMatch.canonical_model_id == canonical_id,
            ModelMatch.ai_model_id == ai_model_id
        ).first()
        
        if not match:
            return jsonify({'error': 'Model match not found'}), 404
        
        session.delete(match)
        session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Model removed from canonical model'
        })
    
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        close_session()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
