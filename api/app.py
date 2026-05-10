"""
Flask REST API cho Surface Defect Detection.
"""

import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from loguru import logger

from api.routes.detection import detection_bp


def create_app() -> Flask:
    """Application factory pattern."""
    load_dotenv()

    app = Flask(__name__)
    CORS(app)

    # Config từ env
    app.config["MODEL_PATH"] = os.getenv("MODEL_PATH", "models/onnx/defect_detector.onnx")
    app.config["CONFIDENCE_THRESHOLD"] = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    app.config["IOU_THRESHOLD"] = float(os.getenv("IOU_THRESHOLD", "0.45"))
    app.config["MAX_IMAGE_SIZE_MB"] = float(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_IMAGE_SIZE_MB"] * 1024 * 1024)

    # Register blueprints
    app.register_blueprint(detection_bp, url_prefix="/api/v1")

    # Health check
    @app.route("/health")
    def health():
        return {"status": "healthy", "version": "1.0.0"}, 200

    logger.info(f"App initialized. Model: {app.config['MODEL_PATH']}")
    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
