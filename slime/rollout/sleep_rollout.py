import logging
import time

logger = logging.getLogger(__name__)


def sleep(args, rollout_id, data_source, evaluation=False):
    count = 0
    while True:
        time.sleep(3600)
        count += 1
        logger.info(f"rollout sleep for {count} hours")
