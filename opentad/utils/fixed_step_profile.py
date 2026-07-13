"""Stateful fixed-event profiler that can span multiple training epochs."""

import time


class FixedStepProfileError(RuntimeError):
    pass


class TorchCudaProfileBackend:
    """Bind CUDA synchronization and peak-memory operations to one device."""

    def __init__(self, device):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise FixedStepProfileError("fixed-step profiling requires an available CUDA device")

    def synchronize(self):
        self._torch.cuda.synchronize(self.device)

    def reset_peak_memory(self):
        self._torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_bytes(self):
        return int(self._torch.cuda.max_memory_allocated(self.device))


class FixedStepProfiler:
    """Measure exactly N successful optimizer events after a warmup prefix."""

    def __init__(
        self,
        warmup_optimizer_events,
        measured_optimizer_events,
        *,
        backend,
        clock=time.perf_counter,
    ):
        if (
            isinstance(warmup_optimizer_events, bool)
            or not isinstance(warmup_optimizer_events, int)
            or warmup_optimizer_events < 0
        ):
            raise FixedStepProfileError("warmup optimizer events must be non-negative")
        if (
            isinstance(measured_optimizer_events, bool)
            or not isinstance(measured_optimizer_events, int)
            or measured_optimizer_events <= 0
        ):
            raise FixedStepProfileError("measured optimizer events must be positive")
        self.warmup_optimizer_events = warmup_optimizer_events
        self.measured_optimizer_events = measured_optimizer_events
        self.backend = backend
        self.clock = clock
        self.successful_optimizer_events = 0
        self.skipped_optimizer_events = 0
        self._measurement_started = False
        self._complete = False
        self._start_time = None
        self._elapsed_seconds = None
        self._peak_memory_bytes = None

    @property
    def total_optimizer_events(self):
        return self.warmup_optimizer_events + self.measured_optimizer_events

    @property
    def complete(self):
        return self._complete

    def start(self):
        if self._complete:
            return
        if self.warmup_optimizer_events == 0 and not self._measurement_started:
            self._begin_measurement()

    def _begin_measurement(self):
        self.backend.synchronize()
        self.backend.reset_peak_memory()
        self._start_time = float(self.clock())
        self._measurement_started = True

    def _finish_measurement(self):
        self.backend.synchronize()
        elapsed = float(self.clock()) - self._start_time
        if elapsed <= 0:
            raise FixedStepProfileError("fixed-step measured time must be positive")
        self._elapsed_seconds = elapsed
        self._peak_memory_bytes = int(self.backend.peak_memory_bytes())
        if self._peak_memory_bytes < 0:
            raise FixedStepProfileError("fixed-step peak memory cannot be negative")
        self._complete = True

    def record_optimizer_event(self):
        if self._complete:
            raise FixedStepProfileError("fixed-step profile already completed")
        self.successful_optimizer_events += 1
        if (
            self.successful_optimizer_events == self.warmup_optimizer_events
            and not self._measurement_started
        ):
            self._begin_measurement()
        if self.successful_optimizer_events == self.total_optimizer_events:
            if not self._measurement_started:
                raise FixedStepProfileError("fixed-step measurement never started")
            self._finish_measurement()
        elif self.successful_optimizer_events > self.total_optimizer_events:
            raise FixedStepProfileError("fixed-step optimizer event budget was exceeded")
        return self._complete

    def record_skipped_optimizer_event(self):
        self.skipped_optimizer_events += 1
        raise FixedStepProfileError(
            "fixed-step profile is invalid after a skipped optimizer event"
        )

    def measurements(self):
        if not self._complete:
            raise FixedStepProfileError("fixed-step profile is incomplete")
        throughput = self.measured_optimizer_events / self._elapsed_seconds
        return {
            "warmup_optimizer_events": self.warmup_optimizer_events,
            "measured_optimizer_events": self.measured_optimizer_events,
            "total_optimizer_events": self.successful_optimizer_events,
            "skipped_optimizer_events": self.skipped_optimizer_events,
            "elapsed_seconds": self._elapsed_seconds,
            "peak_memory_bytes": self._peak_memory_bytes,
            "throughput_optimizer_events_per_second": throughput,
        }


__all__ = [
    "FixedStepProfileError",
    "FixedStepProfiler",
    "TorchCudaProfileBackend",
]
