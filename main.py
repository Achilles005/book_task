from build_app import AppBuilder
import uvicorn

def main():
    # Initialize AppBuilder and load all necessary components
    builder = AppBuilder(config_path="config.yaml")
    components = builder.build()

    # Extract components for use
    logger = components["logger"]
    app = components["app"]

    # Log the initialization status
    logger.info("Application components initialized successfully.")

    # Start the FastAPI server with Uvicorn
    logger.info("Starting FastAPI app with Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
