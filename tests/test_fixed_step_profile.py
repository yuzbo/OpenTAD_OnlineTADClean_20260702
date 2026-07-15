import pytest

from opentad.utils.fixed_step_profile import FixedStepProfileError, FixedStepProfiler


class _Backend:
    def __init__(self, peak=4096):
        self.peak = peak
        self.synchronizations = 0
        self.resets = 0

    def synchronize(self):
        self.synchronizations += 1

    def reset_peak_memory(self):
        self.resets += 1

    def peak_memory_bytes(self):
        return self.peak


def _clock(*values):
    iterator = iter(values)
    return lambda: next(iterator)


def _workload(**overrides):
    values = {
        "optimizer_events": 1,
        "episode_draws": 2,
        "temporal_forward_tokens": 20,
        "temporal_backward_tokens": 12,
        "replay_tokens": 20,
        "supervised_exposures": 8,
        "unique_supervised_bins": 6,
        "ipw_weight_sum": 4.0,
        "ipw_weight_squared_sum": 2.0,
        "visual_forward_frames": 0,
        "visual_backward_frames": 0,
        "data_wait_seconds": 0.1,
        "control_unroll_seconds": 0.2,
        "wall_seconds": 1.0,
    }
    values.update(overrides)
    return values


def test_fixed_step_profile_measures_exact_events_after_warmup():
    backend = _Backend()
    profiler = FixedStepProfiler(2, 3, backend=backend, clock=_clock(10.0, 12.0))

    profiler.start()
    assert profiler.record_optimizer_event("event-0") is False
    assert profiler.record_optimizer_event("event-1") is False
    assert backend.resets == 1
    assert profiler.record_optimizer_event("event-2") is False
    assert profiler.record_optimizer_event("event-3") is False
    assert profiler.record_optimizer_event("event-4") is True

    assert profiler.complete is True
    assert profiler.measurements() == {
        "warmup_optimizer_events": 2,
        "measured_optimizer_events": 3,
        "total_optimizer_events": 5,
        "skipped_optimizer_events": 0,
        "measurement_start_after_event_id": "event-1",
        "measurement_end_event_id": "event-4",
        "elapsed_seconds": 2.0,
        "peak_memory_bytes": 4096,
        "throughput_optimizer_events_per_second": 1.5,
    }
    assert backend.synchronizations == 2


def test_zero_warmup_starts_before_first_measured_event():
    backend = _Backend(peak=1)
    profiler = FixedStepProfiler(0, 1, backend=backend, clock=_clock(5.0, 6.0))

    profiler.start()
    assert profiler.record_optimizer_event("event-0") is True
    assert profiler.measurements()["total_optimizer_events"] == 1


def test_profile_state_can_span_multiple_epoch_calls():
    profiler = FixedStepProfiler(1, 2, backend=_Backend(), clock=_clock(1.0, 3.0))

    profiler.start()
    profiler.record_optimizer_event("event-0")
    profiler.start()
    profiler.record_optimizer_event("event-1")
    profiler.start()
    assert profiler.record_optimizer_event("event-2") is True


def test_multi_denominator_workload_excludes_warmup_and_reports_pooled_ess():
    profiler = FixedStepProfiler(1, 2, backend=_Backend(peak=8192), clock=_clock(4.0, 8.0))

    profiler.start()
    profiler.record_optimizer_event("warmup", workload=_workload(temporal_forward_tokens=999))
    profiler.record_optimizer_event("measured-0", workload=_workload())
    assert profiler.record_optimizer_event(
        "measured-1",
        workload=_workload(ipw_weight_sum=2.0, ipw_weight_squared_sum=1.0),
    ) is True

    workload = profiler.workload_measurements(world_size=2)
    assert workload["measured_optimizer_events"] == 2
    assert workload["totals"]["optimizer_events"] == 2
    assert isinstance(workload["totals"]["optimizer_events"], int)
    assert workload["totals"]["temporal_forward_tokens"] == 40
    assert workload["totals"]["effective_sample_size"] == pytest.approx(12.0)
    assert workload["totals"]["gpu_hours"] == pytest.approx(8.0 / 3600.0)
    assert workload["rates"]["temporal_forward_tokens_per_second"] == 10.0
    assert workload["rates"]["effective_samples_per_second"] == 3.0


@pytest.mark.parametrize("world_size", [0, -1, True, 1.5])
def test_multi_denominator_workload_rejects_invalid_world_size(world_size):
    profiler = FixedStepProfiler(0, 1, backend=_Backend(), clock=_clock(1.0, 2.0))
    profiler.start()
    profiler.record_optimizer_event("event", workload=_workload())

    with pytest.raises(FixedStepProfileError, match="world size"):
        profiler.workload_measurements(world_size=world_size)


def test_any_skipped_optimizer_event_invalidates_profile():
    profiler = FixedStepProfiler(1, 1, backend=_Backend(), clock=_clock(1.0))

    with pytest.raises(FixedStepProfileError, match="skipped optimizer"):
        profiler.record_skipped_optimizer_event()
    assert profiler.skipped_optimizer_events == 1


@pytest.mark.parametrize(
    ("warmup", "measured"),
    [(-1, 1), (True, 1), (0, 0), (0, -1), (0, True)],
)
def test_profile_rejects_invalid_event_counts(warmup, measured):
    with pytest.raises(FixedStepProfileError):
        FixedStepProfiler(warmup, measured, backend=_Backend())
