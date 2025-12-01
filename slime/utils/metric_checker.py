import logging

logger = logging.getLogger(__name__)


class MetricChecker:
    @staticmethod
    def maybe_create(args):
        if args.ci_test and (args.ci_metric_checker_key is not None):
            return MetricChecker(args)
        return None

    def __init__(self, args):
        self.args = args
        self._exists_check_success = False

    def on_eval(self, metrics: dict[str, float]):
        actual_value = metrics.get(self.args.ci_metric_checker_key)
        assert actual_value is not None, f"{metrics=} {self.args.ci_metric_checker_key=}"

        check_success = actual_value >= self.args.ci_metric_checker_threshold
        logger.info(f"[MetricChecker] {check_success=} {actual_value=} {self.args.ci_metric_checker_threshold=}")

        self._exists_check_success |= check_success

    def dispose(self):
        assert self._exists_check_success, "[MetricChecker] accuracy check failed"
        logger.info("[MetricChecker] pass dispose check")
