import os
import logging
from flask import Flask
from flask_cors import CORS
from models import db

# Create logger
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Enable CORS
CORS(app)

# Configure app
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max request size
app.config["SCRAPER_TIMEOUT"] = 30  # 30 seconds timeout for scrapers
app.config["RATE_LIMIT_WINDOW"] = 60  # 60 seconds window for rate limiting
app.config["RATE_LIMIT_MAX_REQUESTS"] = 10  # 10 requests per window
app.config["LLM_MODEL"] = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.

# Configure database (SQLite)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///horoscope.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database
db.init_app(app)

# Create all tables
with app.app_context():
    db.create_all()

logger.info("Flask app initialized")
