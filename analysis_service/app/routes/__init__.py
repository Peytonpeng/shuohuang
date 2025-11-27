# app/routes/__init__.py
from analysis_service.hello_routes import hello_blueprint
from .auth import auth_bp
from .train_select import train_select_bp
from .preprocess import preprocess_bp
from .feature import feature_bp
from .train import train_bp
from .apply import apply_bp


def register_blueprints(app):
    app.register_blueprint(hello_blueprint)  # 你原来的 hello
    app.register_blueprint(auth_bp,        url_prefix="/api/analysis")
    app.register_blueprint(train_select_bp, url_prefix="/api/analysis/train")
    app.register_blueprint(preprocess_bp,   url_prefix="/api/analysis/train")
    app.register_blueprint(feature_bp,      url_prefix="/api/analysis/train")
    app.register_blueprint(train_bp,        url_prefix="/api/analysis/train")
    app.register_blueprint(apply_bp,        url_prefix="/api/analysis/apply")
