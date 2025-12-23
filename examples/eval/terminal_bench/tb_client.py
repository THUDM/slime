import logging
import time
from typing import Any

import requests
from examples.eval.eval_delegate import EvalClient, EvalDelegateError
from examples.eval.terminal_bench.tb_config import TerminalBenchConfig

logger = logging.getLogger(__name__)


class TerminalBenchClient(EvalClient):
    """HTTP client that proxies evaluation requests to the Terminal Bench server."""

    def __init__(self, config: TerminalBenchConfig, router_url: str):
        super().__init__(config.name or "terminal_bench")
        self._config = config
        endpoint = (config.url or "").rstrip("/")
        if endpoint.endswith("/evaluate"):
            base_endpoint = endpoint[: -len("/evaluate")]
        else:
            base_endpoint = endpoint
        self._endpoint = f"{base_endpoint}/evaluate" if base_endpoint else ""
        self._status_endpoint = f"{base_endpoint}/status" if base_endpoint else ""
        self._timeout_secs = float(config.timeout_secs)
        self._max_retries = max(1, int(config.max_retries))
        self._headers = dict(config.headers or {})
        self._session = requests.Session()

    @classmethod
    def from_config(cls, config: TerminalBenchConfig, router_url: str):
        if not config.url:
            return None
        return cls(config, router_url)

    def evaluate(self, args, rollout_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = self._build_payload(args, rollout_id)
        response = self._request(payload)
        metrics = response.get("raw_metrics", {})
        return metrics, response

    def _build_payload(self, args, rollout_id: int) -> dict[str, Any]:
        payload = {
            "model_name": self._config.model_name,
            "api_base": self._config.api_base,
            "n_tasks": self._config.n_tasks,
            "n_concurrent": self._config.n_concurrent,
        }
        if self._config.dataset_path:
            payload["dataset_path"] = self._config.dataset_path
        if self._config.task_ids:
            payload["task_ids"] = list(self._config.task_ids)
        return payload

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout_secs,
                    headers=self._headers,
                )
                response.raise_for_status()
                if not response.content:
                    return {}
                body = response.json()
                if body.get("status") == "completed":
                    return body
                job_id = body.get("job_id")
                if not job_id:
                    return body
                return self._poll_status(job_id)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Terminal Bench delegate request failed (attempt %s/%s): %s", attempt, self._max_retries, exc
                )
                if attempt < self._max_retries:
                    time.sleep(min(2**attempt, 30))
        raise EvalDelegateError("Terminal Bench evaluation request failed") from last_error

    def _poll_status(self, job_id: str) -> dict[str, Any]:
        status_url = f"{self._status_endpoint}/{job_id}"
        deadline = time.time() + self._timeout_secs
        while time.time() < deadline:
            response = self._session.get(status_url, timeout=min(self._timeout_secs, 30), headers=self._headers)
            response.raise_for_status()
            if not response.content:
                time.sleep(2)
                continue
            body = response.json()
            status = body.get("status")
            if status == "completed":
                return body
            if status == "failed":
                error = body.get("error") or "Terminal Bench job failed"
                raise EvalDelegateError(error)
            time.sleep(2)
        raise EvalDelegateError("Terminal Bench evaluation timed out")
