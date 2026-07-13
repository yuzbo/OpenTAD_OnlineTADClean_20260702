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


def test_fixed_step_profile_measures_exact_events_after_warmup():
    backend = _Backend()
    profiler = FixedStepProfiler(2, 3, backend=backend, clock=_clock(10.0, 12.0))

    profiler.start()
    assert profiler.record_optimizer_event() is False
    assert profiler.record_optimizer_event() is False
    assert backend.resets == 1
    assert profiler.record_optimizer_event() is False
    assert profiler.record_optimizer_event() is False
    assert profiler.record_optimizer_event() is True

    assert profiler.complete is True
    assert profiler.measurements() == {
        "warmup_optimizer_events": 2,
        "measured_optimizer_events": 3,
        "total_optimizer_events": 5,
        "skipped_optimizer_events": 0,
        "elapsed_seconds": 2.0,
        "peak_memory_bytes": 4096,
        "throughput_optimizer_events_per_second": 1.5,
    }
    assert backend.synchronizations == 2


def test_zero_warmup_starts_before_first_measured_event():
    backend = _Backend(peak=1)
    profiler = FixedStepProfiler(0, 1, backend=backend, clock=_clock(5.0, 6.0))

    profiler.start()
    assert profiler.record_optimizer_event() is True
    assert profiler.measurements()["total_optimizer_events"] == 1


def test_profile_state_can_span_multiple_epoch_calls():
    profiler = FixedStepProfiler(1, 2, backend=_Backend(), clock=_clock(1.0, 3.0))

    profiler.start()
    profiler.record_optimizer_event()
    profiler.start()
    profiler.record_optimizer_event()
    profiler.start()
    assert profiler.record_optimizer_event() is True


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
