import __set_base_path__  # noqa

import common.settings
from common.logger import set_logger

LOGGER_IS_ACTIVE_STREAM = common.settings.logger_is_active_stream

logger = set_logger(__name__, is_active_stream=LOGGER_IS_ACTIVE_STREAM)

logger.info("__set_base_path__ sample")
