import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up.")
