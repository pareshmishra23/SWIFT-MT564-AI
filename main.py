import logging
from app import app
from api.routes import register_routes
from api.mt564_routes import register_mt564_routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Register API routes
register_routes(app)
register_mt564_routes(app)

if __name__ == "__main__":
    logger.info("Starting the web scraper API server")
    app.run(host="0.0.0.0", port=5001, debug=True)
