from __future__ import annotations

import contextlib
import gc
import logging
import threading
import time
from typing import Any, Generator, Iterable, Literal

logger = logging.getLogger("laas.concurrency")

HeavyResourceType = Literal["llm", "image", "image_edit"]


class ConcurrencyCoordinator:
    """Coordinates GPU VRAM occupancy and serialization of heavy model execution."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.active_jobs: dict[HeavyResourceType, int] = {
            "llm": 0,
            "image": 0,
            "image_edit": 0,
        }
        self.active_resource: HeavyResourceType | None = None
        self.managers: dict[HeavyResourceType, Any] = {}

    def register_manager(self, resource: HeavyResourceType, manager: Any) -> None:
        """Registers a ModelManager, ImageManager, or ImageEditManager."""
        with self.lock:
            self.managers[resource] = manager

    def acquire(self, resource: HeavyResourceType) -> None:
        """Acquires execution rights for a resource, loading it and unloading others if needed."""
        with self.lock:
            self._wait_for_idle_locked(waiter=f"Resource '{resource}'")

            # No other resource has active jobs. We can safely unload other loaded resources.
            self._unload_other_resources_locked(resource)

            # Ensure the target resource is loaded
            mgr = self.managers.get(resource)
            if mgr is not None:
                try:
                    if not getattr(mgr, "is_loaded", False):
                        logger.info(f"Loading resource '{resource}' under serialization lock")
                        download_if_missing = True
                        if hasattr(mgr, "settings"):
                            if resource == "llm":
                                download_if_missing = getattr(mgr.settings, "auto_download", True)
                            elif resource == "image":
                                download_if_missing = getattr(mgr.settings, "image_auto_download", True)
                            elif resource == "image_edit":
                                download_if_missing = getattr(mgr.settings, "image_edit_auto_download", True)
                        mgr.load(download_if_missing=download_if_missing)
                except Exception:
                    # Notify others if we crashed during load, so they don't wait indefinitely
                    self.condition.notify_all()
                    raise

            # Increment active job counter
            self.active_jobs[resource] += 1
            self.active_resource = resource
            logger.debug(f"Acquired resource '{resource}' (active jobs: {self.active_jobs[resource]})")

    def release(self, resource: HeavyResourceType) -> None:
        """Releases execution rights for a resource."""
        with self.lock:
            self.active_jobs[resource] = max(0, self.active_jobs[resource] - 1)
            logger.debug(f"Released resource '{resource}' (active jobs: {self.active_jobs[resource]})")
            if self.active_jobs[resource] == 0:
                if self.active_resource == resource:
                    self.active_resource = None
            self.condition.notify_all()

    @contextlib.contextmanager
    def execute(self, resource: HeavyResourceType) -> Generator[None, None, None]:
        """Context manager to serialize execution of a heavy model."""
        self.acquire(resource)
        try:
            yield
        finally:
            self.release(resource)

    @contextlib.contextmanager
    def maintenance(self, resource: HeavyResourceType | None = None) -> Generator[None, None, None]:
        """Runs load/unload lifecycle work after all heavy requests have drained."""
        with self.lock:
            self._wait_for_idle_locked(waiter="Maintenance")
            if resource is not None:
                self._unload_other_resources_locked(resource)
            try:
                yield
            finally:
                self.condition.notify_all()

    def wrap_stream(self, resource: HeavyResourceType, generator: Iterable[Any]) -> Iterable[Any]:
        """Wraps a streaming response generator to ensure the resource is held until exhaustion or completion."""
        # Note: self.acquire(resource) MUST be called before calling wrap_stream.
        try:
            for chunk in generator:
                yield chunk
        finally:
            self.release(resource)

    def _wait_for_idle_locked(self, *, waiter: str) -> None:
        while True:
            active = [resource for resource, count in self.active_jobs.items() if count > 0]
            if not active:
                return
            logger.debug("%s waiting because resources are active: %s", waiter, active)
            self.condition.wait()

    def _unload_other_resources_locked(self, resource: HeavyResourceType) -> None:
        unloaded_any = False
        for other_resource, manager in self.managers.items():
            if other_resource != resource and getattr(manager, "is_loaded", False):
                logger.info("Unloading resource '%s' to free VRAM for '%s'", other_resource, resource)
                try:
                    manager.unload()
                    unloaded_any = True
                except Exception as exc:
                    logger.error("Error unloading resource '%s': %s", other_resource, exc)
        if unloaded_any:
            self.clear_accelerator_cache()

    @staticmethod
    def clear_accelerator_cache() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch CUDA VRAM cache")
        except ImportError:
            pass
