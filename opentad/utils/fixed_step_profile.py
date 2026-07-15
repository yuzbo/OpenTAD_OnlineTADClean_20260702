"""Stateful fixed-event profiler that can span multiple training epochs."""

import time


_WORKLOAD_FIELDS = {
    "optimizer_events",
    "episode_draws",
    "temporal_forward_tokens",
    "temporal_backward_tokens",
    "replay_tokens",
    "supervised_exposures",
    "unique_supervised_bins",
    "ipw_weight_sum",
    "ipw_weight_squared_sum",
    "visual_forward_frames",
    "visual_backward_frames",
    "data_wait_seconds",
    "control_unroll_seconds",
    "wall_seconds",
}
_WORKLOAD_FLOAT_FIELDS = {
    "ipw_weight_sum",
    "ipw_weight_squared_sum",
    "data_wait_seconds",
    "control_unroll_seconds",
    "wall_seconds",
}


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
        self._measurement_start_after_event_id = None
        self._measurement_end_event_id = None
        self._measured_workload = {
            field: 0.0 if field in _WORKLOAD_FLOAT_FIELDS else 0
            for field in _WORKLOAD_FIELDS
        }
        self._workload_event_count = 0

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
            self._begin_measurement("runtime-genesis")

    def _begin_measurement(self, boundary_event_id):
        if not isinstance(boundary_event_id, str) or not boundary_event_id:
            raise FixedStepProfileError("measurement boundary event ID is invalid")
        self.backend.synchronize()
        self.backend.reset_peak_memory()
        self._start_time = float(self.clock())
        self._measurement_start_after_event_id = boundary_event_id
        self._measurement_started = True

    def _finish_measurement(self, end_event_id):
        if not isinstance(end_event_id, str) or not end_event_id:
            raise FixedStepProfileError("measurement end event ID is invalid")
        self.backend.synchronize()
        elapsed = float(self.clock()) - self._start_time
        if elapsed <= 0:
            raise FixedStepProfileError("fixed-step measured time must be positive")
        self._elapsed_seconds = elapsed
        self._peak_memory_bytes = int(self.backend.peak_memory_bytes())
        if self._peak_memory_bytes < 0:
            raise FixedStepProfileError("fixed-step peak memory cannot be negative")
        self._measurement_end_event_id = end_event_id
        self._complete = True

    @staticmethod
    def _validated_workload(workload):
        if not isinstance(workload, dict) or set(workload) != _WORKLOAD_FIELDS:
            raise FixedStepProfileError("profile workload fields differ")
        normalized = {}
        for field, value in workload.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FixedStepProfileError(f"profile workload {field} must be numeric")
            value = float(value) if field in _WORKLOAD_FLOAT_FIELDS else int(value)
            if value < 0:
                raise FixedStepProfileError(f"profile workload {field} must be non-negative")
            normalized[field] = value
        if normalized["optimizer_events"] != 1:
            raise FixedStepProfileError("each profile workload must describe one optimizer event")
        return normalized

    def record_optimizer_event(self, event_id, *, workload=None):
        if self._complete:
            raise FixedStepProfileError("fixed-step profile already completed")
        if not isinstance(event_id, str) or not event_id.strip():
            raise FixedStepProfileError("fixed-step optimizer event ID is invalid")
        workload = None if workload is None else self._validated_workload(workload)
        self.successful_optimizer_events += 1
        if (
            self.successful_optimizer_events == self.warmup_optimizer_events
            and not self._measurement_started
        ):
            self._begin_measurement(event_id)
        if self.successful_optimizer_events == self.total_optimizer_events:
            if not self._measurement_started:
                raise FixedStepProfileError("fixed-step measurement never started")
            self._finish_measurement(event_id)
        elif self.successful_optimizer_events > self.total_optimizer_events:
            raise FixedStepProfileError("fixed-step optimizer event budget was exceeded")
        if (
            workload is not None
            and self.warmup_optimizer_events
            < self.successful_optimizer_events
            <= self.total_optimizer_events
        ):
            for field, value in workload.items():
                self._measured_workload[field] += value
            self._workload_event_count += 1
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
            "measurement_start_after_event_id": self._measurement_start_after_event_id,
            "measurement_end_event_id": self._measurement_end_event_id,
            "elapsed_seconds": self._elapsed_seconds,
            "peak_memory_bytes": self._peak_memory_bytes,
            "throughput_optimizer_events_per_second": throughput,
        }

    def workload_measurements(self, *, world_size=1):
        if not self._complete:
            raise FixedStepProfileError("fixed-step profile is incomplete")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise FixedStepProfileError("profile world size must be a positive integer")
        if self._workload_event_count != self.measured_optimizer_events:
            raise FixedStepProfileError(
                "every measured optimizer event requires a workload record"
            )
        totals = dict(self._measured_workload)
        weight_sum = totals["ipw_weight_sum"]
        weight_squared_sum = totals["ipw_weight_squared_sum"]
        effective_sample_size = (
            weight_sum * weight_sum / weight_squared_sum
            if weight_squared_sum > 0
            else 0.0
        )
        elapsed = self._elapsed_seconds
        return {
            "schema_version": "full-petal-multi-denominator-profile-v1",
            "measured_optimizer_events": self.measured_optimizer_events,
            "totals": {
                **totals,
                "effective_sample_size": effective_sample_size,
                "gpu_hours": elapsed * world_size / 3600.0,
                "peak_memory_bytes": self._peak_memory_bytes,
            },
            "rates": {
                "temporal_forward_tokens_per_second": totals[
                    "temporal_forward_tokens"
                ]
                / elapsed,
                "temporal_backward_tokens_per_second": totals[
                    "temporal_backward_tokens"
                ]
                / elapsed,
                "supervised_exposures_per_second": totals[
                    "supervised_exposures"
                ]
                / elapsed,
                "effective_samples_per_second": effective_sample_size / elapsed,
                "video_groups_per_second": self.measured_optimizer_events / elapsed,
            },
        }


__all__ = [
    "FixedStepProfileError",
    "FixedStepProfiler",
    "TorchCudaProfileBackend",
]
