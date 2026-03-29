"""
app/app.py
=============================================================================
Main Application Entry Point.
Initializes the Flask server, hydrates the ML registry, 
and sets up global error handling and modular routing.
"""

import os
import sys
import logging
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

# Setup Path so we can import from `app` and `src`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.models import registry
from app.routes import bp as main_blueprint

# Global Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    """
    Flask Application Factory.
    Creates and configures the app, initializes memory singletons, 
    and registers blueprints.
    """
    app = Flask(__name__)
    CORS(app) # Enable cross-origin resource sharing for decoupled frontends
    
    # 1. Hydrate Model Registry (Singleton) on startup
    try:
        registry.load_artifacts()
    except Exception as e:
        logger.error(f"Inference Engine could not be hydrated internally: {e}")
        # Note: In production, we might want to halt completely here
    
    # 2. Register Blueprints
    app.register_blueprint(main_blueprint)
    
    # 3. Global Error Handling
    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({"error": "Resource not found", "success": False}), 404
        
    @app.errorhandler(500)
    def handle_server_error(e):
        return jsonify({"error": "Internal Server Error", "success": False}), 500
        
    return app

# Development Server Entry
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
