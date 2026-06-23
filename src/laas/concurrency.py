from __future__ import annotations

import contextlib
import gc
import logging
import threading
import time
from typing import Any, Generator, Iterable, Literal

logger = logging.getLogger("laas.concurrency")

HeavyResourceType = Literal["llm", "image", "image_edit", "video"]


class ConcurrencyCoordinator:
    """Coordinates GPU VRAM occupancy and serialization of heavy model execution."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.active_jobs: dict[HeavyResourceType, int] = {
            "llm": 0,
            "image": 0,
            "image_edit": 0,
            "video": 0,
        }
        self.active_resource: HeavyResourceType | None = None
        self.managers: dict[HeavyResourceType, Any] = {}

    def register_manager(self, resource: HeavyResourceType, manager: Any) -> None:
        """Registers a ModelManager, ImageManager, or ImageEditManager."""
        with self.lock:
            self.managers[resource] = manager

    def status(self) -> dict[str, Any]:
        """Returns a lock-protected snapshot of heavy-resource coordination state."""
        with self.lock:
            resources = {
                resource: {
                    "active_jobs": self.active_jobs.get(resource, 0),
                    "registered": resource in self.managers,
                    "is_loaded": bool(getattr(self.managers.get(resource), "is_loaded", False)),
                    "manager": type(self.managers[resource]).__name__ if resource in self.managers else None,
                }
                for resource in self.active_jobs
            }
            return {
                "active_resource": self.active_resource,
                "active_jobs": dict(self.active_jobs),
                "total_active_jobs": sum(self.active_jobs.values()),
                "registered_resources": sorted(self.managers),
                "resources": resources,
            }

    def acquire(self, resource: HeavyResourceType) -> None:
        """Acquires execution rights for a resource, loading it and unloading others if needed."""
        with self.lock:
            self._wait_for_idle_locked(waiter=f"Resource '{resource}'")

            # No other resource has active jobs. We can safely unload other loaded resources.
            self._unload_other_resources_locked(resource)

            mgr = self.managers.get(resource)
            needs_load = mgr is not None and not getattr(mgr, "is_loaded", False)
            download_if_missing = self._download_if_missing(resource, mgr)

            # Increment active job counter
            self.active_jobs[resource] += 1
            self.active_resource = resource
            logger.debug(f"Acquired resource '{resource}' (active jobs: {self.active_jobs[resource]})")

        if mgr is not None and needs_load:
            try:
                logger.info("Loading resource '%s' under coordinator lease", resource)
                mgr.load(download_if_missing=download_if_missing)
            except Exception:
                self.release(resource)
                raise

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
    def _download_if_missing(resource: HeavyResourceType, manager: Any) -> bool:
        if not hasattr(manager, "settings"):
            return True
        if resource == "llm":
            return bool(getattr(manager.settings, "auto_download", True))
        if resource == "image":
            return bool(getattr(manager.settings, "image_auto_download", True))
        if resource == "image_edit":
            return bool(getattr(manager.settings, "image_edit_auto_download", True))
        if resource == "video":
            return bool(getattr(manager.settings, "video_generation_auto_download", True))
        return True

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
