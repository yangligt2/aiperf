# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for open-loop session arrivals (``--session-arrival-rate``).

Covers the arrival driver's admission accounting and stop semantics, the
strategy-side hooks it drives (fresh turn-0 admission, per-arrival cache-bust
namespace, recycle suppression), and the config/scenario gates that keep the
flag off non-agentic runs.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.models import DatasetMetadata
from aiperf.common.scenario.base import ScenarioSpec
from aiperf.common.scenario.validator import _apply_session_arrival_rate
from aiperf.config.session_arrival import SessionArrivalConfig
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import ArrivalPattern, DatasetSamplingStrategy
from aiperf.timing.intervals import (
    ConstantIntervalGenerator,
    PoissonIntervalGenerator,
)
from aiperf.timing.session_arrival import SessionArrivalDriver
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource
from tests.unit.timing.strategies._shared_helpers import _make_dataset, _make_run

# Helpers


def _make_driver(
    *,
    rate: float = 10.0,
    pattern: ArrivalPattern = ArrivalPattern.POISSON,
    smoothness: float | None = None,
    strategy: MagicMock | None = None,
    can_start_new_session: list[bool] | bool = True,
    can_acquire: list[bool] | bool = True,
    admit_results: list[bool | None] | None = None,
    deterministic_intervals: bool = True,
) -> SessionArrivalDriver:
    """Build a driver over fully mocked collaborators.

    ``can_start_new_session`` / ``can_acquire`` accept a list consumed as a
    ``side_effect`` so a test can script the loop's exit point; a bare bool is
    a constant return.
    """
    strategy = strategy if strategy is not None else MagicMock()
    strategy.admit_session_arrival = AsyncMock(
        side_effect=admit_results if admit_results is not None else None,
        return_value=True if admit_results is None else None,
    )

    issuer = MagicMock()
    if isinstance(can_acquire, list):
        issuer.can_acquire_and_start_new_session.side_effect = can_acquire
    else:
        issuer.can_acquire_and_start_new_session.return_value = can_acquire

    stop_checker = MagicMock()
    if isinstance(can_start_new_session, list):
        stop_checker.can_start_new_session.side_effect = can_start_new_session
    else:
        stop_checker.can_start_new_session.return_value = can_start_new_session

    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = int(time.perf_counter() * NANOS_PER_SECOND)

    driver = SessionArrivalDriver(
        strategy=strategy,
        credit_issuer=issuer,
        stop_checker=stop_checker,
        lifecycle=lifecycle,
        rate=rate,
        pattern=pattern,
        smoothness=smoothness,
    )
    if deterministic_intervals:
        # asyncio.sleep is a no-op in unit tests, so the sampled interval only
        # affects the behind-schedule counter; pin it so counters are stable.
        driver._generator = MagicMock()
        driver._generator.next_interval.return_value = 0.0
    return driver


def _arrival_config(
    *,
    phase: CreditPhase = CreditPhase.PROFILING,
    rate: float | None = 5.0,
    pattern: ArrivalPattern = ArrivalPattern.POISSON,
    smoothness: float | None = None,
    concurrency: int = 4,
) -> MagicMock:
    """CreditPhaseConfig stand-in carrying the session-arrival fields.

    A bare MagicMock cannot be used: every attribute would be a truthy mock and
    the strategy would try to build an interval generator from it.
    """
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = concurrency
    cfg.agentic_cache_warmup_duration_sec = None
    cfg.session_arrival = (
        None
        if rate is None
        else SessionArrivalConfig(rate=rate, pattern=pattern, smoothness=smoothness)
    )
    return cfg


def _arrival_source(dataset: DatasetMetadata) -> TrajectorySource:
    """Open-loop TrajectorySource: no trajectories, roots-only sampler."""
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = dataset
    root_ids = [
        c.conversation_id for c in dataset.conversations if getattr(c, "is_root", True)
    ]
    src._dataset_sampler = SequentialSampler(root_ids)
    src._metadata_lookup = {c.conversation_id: c for c in dataset.conversations}
    src._random_seed = 0
    src._target_size = 0
    src._pool_size = len(root_ids)
    src.trajectories = []
    return src


def _make_arrival_strategy(
    *,
    num_traces: int = 2,
    turns_per_trace: int = 3,
    phase: CreditPhase = CreditPhase.PROFILING,
    rate: float | None = 5.0,
    pattern: ArrivalPattern = ArrivalPattern.POISSON,
    cache_bust: CacheBustTarget = CacheBustTarget.FIRST_TURN_PREFIX,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, TrajectorySource]:
    dataset = _make_dataset(num_traces, turns_per_trace)
    src = _arrival_source(dataset)
    issuer = issuer if issuer is not None else AsyncMock()
    issuer.replay_gate = MagicMock()
    issuer.replay_gate.completed_prefixes.return_value = ()
    issuer.replay_gate.pending_turns.return_value = ()
    issuer.replay_gate.pending_turns_by_root.return_value = {}
    scheduler = scheduler if scheduler is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=_arrival_config(phase=phase, rate=rate, pattern=pattern),
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        run=_make_run(target=cache_bust),
    )
    return strategy, issuer, scheduler, src


# SessionArrivalDriver: admission accounting


@pytest.mark.asyncio
async def test_driver_admits_one_session_per_arrival_instant():
    driver = _make_driver(can_start_new_session=[True, True, False])
    await driver.run()

    assert driver._arrivals == 2
    assert driver._admitted == 2
    assert driver._rejected_overload == 0
    assert driver._rejected_unspawnable == 0
    # arrival_index is the per-session ordinal that replaces the lane index.
    assert [
        call.args[0] for call in driver._strategy.admit_session_arrival.call_args_list
    ] == [0, 1]


@pytest.mark.asyncio
async def test_driver_stops_generating_when_session_budget_exhausted():
    """A False budget check must not admit the arrival that triggered it."""
    driver = _make_driver(can_start_new_session=False)
    await driver.run()

    assert driver._arrivals == 0
    assert driver._admitted == 0
    driver._strategy.admit_session_arrival.assert_not_awaited()


@pytest.mark.asyncio
async def test_driver_rejects_at_concurrency_ceiling_and_keeps_generating():
    """Overload rejects the arrival; the arrival process itself continues."""
    driver = _make_driver(
        can_start_new_session=[True, True, False],
        can_acquire=[False, True],
    )
    await driver.run()

    assert driver._arrivals == 2
    assert driver._rejected_overload == 1
    assert driver._admitted == 1
    # The rejected arrival still consumed ordinal 0: the ordinal counts offered
    # arrivals, so admitted sessions never reuse a cache-bust namespace.
    assert driver._strategy.admit_session_arrival.await_args.args[0] == 1


@pytest.mark.asyncio
async def test_driver_warns_once_on_overload(caplog: pytest.LogCaptureFixture):
    driver = _make_driver(
        can_start_new_session=[True, True, True, False],
        can_acquire=False,
    )
    with caplog.at_level(logging.WARNING):
        await driver.run()

    assert driver._rejected_overload == 3
    overload_warnings = [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "Session arrival rejected at the concurrency ceiling" in record.getMessage()
    ]
    assert len(overload_warnings) == 1


@pytest.mark.asyncio
async def test_driver_counts_unspawnable_arrival_and_keeps_generating():
    driver = _make_driver(
        can_start_new_session=[True, True, False],
        admit_results=[None, True],
    )
    await driver.run()

    assert driver._rejected_unspawnable == 1
    assert driver._admitted == 1
    assert driver._arrivals == 2


@pytest.mark.asyncio
async def test_driver_stops_when_credit_issuance_is_refused():
    driver = _make_driver(
        can_start_new_session=True,
        admit_results=[True, False, True],
    )
    await driver.run()

    assert driver._arrivals == 2
    assert driver._admitted == 1


@pytest.mark.asyncio
async def test_driver_requires_lifecycle_start_timestamp():
    driver = _make_driver()
    driver._lifecycle.started_at_perf_ns = None
    with pytest.raises(RuntimeError, match="started_at_perf_ns"):
        await driver.run()


@pytest.mark.asyncio
async def test_driver_resets_target_instead_of_bursting_when_behind():
    """A late loop resets the target to now; it never fires a catch-up burst."""
    driver = _make_driver(can_start_new_session=[True, True, False])
    driver._generator.next_interval.return_value = 0.0
    await driver.run()

    # Every iteration is behind by construction (zero interval + real clock),
    # and each one admits exactly one session, not a backlog.
    assert driver._behind_schedule == driver._arrivals + 1
    assert driver._admitted == 2


def test_stats_summary_reports_every_counter():
    driver = _make_driver(rate=2.5, pattern=ArrivalPattern.CONSTANT)
    driver._arrivals = 7
    driver._admitted = 4
    driver._rejected_overload = 2
    driver._rejected_unspawnable = 1
    driver._behind_schedule = 3

    summary = driver.stats_summary
    assert "pattern=constant" in summary
    assert "rate=2.5/s" in summary
    assert "generated=7" in summary
    assert "admitted=4" in summary
    assert "rejected_overload=2" in summary
    assert "rejected_unspawnable=1" in summary
    assert "behind_schedule_resets=3" in summary


# SessionArrivalDriver: interval generation


def test_driver_defaults_to_poisson_interarrivals():
    driver = _make_driver(deterministic_intervals=False)
    assert isinstance(driver._generator, PoissonIntervalGenerator)


def test_driver_honours_the_configured_arrival_pattern():
    driver = _make_driver(
        rate=4.0, pattern=ArrivalPattern.CONSTANT, deterministic_intervals=False
    )
    assert isinstance(driver._generator, ConstantIntervalGenerator)
    assert driver._generator.next_interval() == pytest.approx(0.25)


# AgenticReplayStrategy: driver construction gate


def test_strategy_builds_no_driver_for_the_warmup_phase():
    """Open loop synthesizes no t* snapshot, so WARMUP has nothing to prime."""
    strategy, _, _, _ = _make_arrival_strategy(phase=CreditPhase.WARMUP)
    assert strategy._arrival_driver is None
    assert strategy._arrival_driven is False


def test_strategy_builds_no_driver_without_an_arrival_rate():
    strategy, _, _, _ = _make_arrival_strategy(rate=None)
    assert strategy._arrival_driver is None


def test_strategy_builds_driver_for_profiling_with_an_arrival_rate():
    strategy, _, _, _ = _make_arrival_strategy(rate=0.5)
    assert strategy._arrival_driven is True
    assert strategy._arrival_driver._rate == 0.5


def test_strategy_warns_when_arrivals_run_without_cache_bust(
    caplog: pytest.LogCaptureFixture,
):
    """Sampling with replacement + no cache bust inflates the hit rate."""
    with caplog.at_level(logging.WARNING):
        _make_arrival_strategy(cache_bust=CacheBustTarget.NONE)

    assert any(
        "sample traces WITH REPLACEMENT" in record.getMessage()
        for record in caplog.records
    )


# AgenticReplayStrategy: admission


@pytest.mark.asyncio
async def test_admit_session_arrival_issues_a_turn_zero_credit():
    strategy, issuer, _, _ = _make_arrival_strategy()
    issuer.issue_credit.return_value = True

    assert await strategy.admit_session_arrival(0) is True

    turn = issuer.issue_credit.await_args.args[0]
    assert turn.turn_index == 0
    assert turn.conversation_id == "trace_0"


@pytest.mark.asyncio
async def test_admit_session_arrival_mints_a_distinct_namespace_per_instance():
    """Two replays of the SAME trace must not share a cache-bust marker."""
    dataset = _make_dataset(1, 3)
    src = _arrival_source(dataset)
    issuer = AsyncMock()
    issuer.replay_gate = MagicMock()
    issuer.issue_credit.return_value = True
    strategy = AgenticReplayStrategy(
        config=_arrival_config(),
        conversation_source=src,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        run=_make_run(target=CacheBustTarget.FIRST_TURN_PREFIX),
    )

    await strategy.admit_session_arrival(0)
    await strategy.admit_session_arrival(1)

    markers = [
        call.args[0].cache_bust_marker for call in issuer.issue_credit.await_args_list
    ]
    assert len(markers) == 2
    assert all(marker for marker in markers)
    assert markers[0] != markers[1]


@pytest.mark.asyncio
async def test_admit_session_arrival_returns_none_when_pool_is_unspawnable():
    strategy, issuer, _, src = _make_arrival_strategy()
    src.next_recycle_conversation_id = MagicMock(return_value=None)

    assert await strategy.admit_session_arrival(0) is None
    issuer.issue_credit.assert_not_awaited()


@pytest.mark.asyncio
async def test_admit_session_arrival_returns_none_for_a_zero_turn_trace():
    strategy, issuer, _, src = _make_arrival_strategy()
    src._metadata_lookup["trace_0"].turns = []
    src.next_recycle_conversation_id = MagicMock(return_value="trace_0")

    assert await strategy.admit_session_arrival(0) is None
    issuer.issue_credit.assert_not_awaited()


@pytest.mark.asyncio
async def test_admit_session_arrival_returns_none_for_an_unknown_trace_id():
    strategy, issuer, _, src = _make_arrival_strategy()
    src.next_recycle_conversation_id = MagicMock(return_value="not-in-corpus")

    assert await strategy.admit_session_arrival(0) is None
    issuer.issue_credit.assert_not_awaited()


@pytest.mark.asyncio
async def test_admit_session_arrival_keeps_bookkeeping_on_a_false_return():
    """False also means "issued, and that was the final credit".

    Rolling the marker back there would strip a live tree's cache namespace and
    let its subagents mint a second one, splitting the tree's prefix domain.
    """
    strategy, issuer, _, _ = _make_arrival_strategy()
    issuer.issue_credit.return_value = False

    assert await strategy.admit_session_arrival(0) is False
    assert len(strategy._session_marker) == 1
    assert len(strategy._root_to_lane) == 1


# AgenticReplayStrategy: recycle suppression


def test_arrival_driven_tree_drain_does_not_recycle():
    """A freed slot stays free until the next exogenous arrival claims it."""
    strategy, _, scheduler, _ = _make_arrival_strategy()
    strategy._correlation_to_lane["root-0"] = 0
    strategy._root_to_lane["root-0"] = 0
    strategy._session_marker["root-0"] = "marker"

    strategy._on_tree_drained("root-0", CreditPhase.PROFILING)

    scheduler.schedule_later.assert_not_called()
    # Bookkeeping is still torn down so a long run does not leak one entry per
    # completed session.
    assert "root-0" not in strategy._correlation_to_lane
    assert "root-0" not in strategy._root_to_lane
    assert "root-0" not in strategy._session_marker


def test_arrival_driven_tree_drain_does_not_warn_about_a_missing_lane(
    caplog: pytest.LogCaptureFixture,
):
    """No lane mapping is the normal case in open loop, not an invariant break."""
    strategy, _, _, _ = _make_arrival_strategy()
    with caplog.at_level(logging.WARNING):
        strategy._on_tree_drained("unknown-root", CreditPhase.PROFILING)

    assert not any("cannot recycle" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_arrival_driven_lane_recycle_is_a_no_op():
    strategy, issuer, _, _ = _make_arrival_strategy()
    await strategy._dispatch_recycled_on_lane(0)
    issuer.issue_credit.assert_not_awaited()


# AgenticReplayStrategy: phase wiring


@pytest.mark.asyncio
async def test_setup_phase_activates_the_gate_without_trajectories():
    """Open loop has no lanes to seed, but fan-out still uses the barrier."""
    strategy, issuer, _, _ = _make_arrival_strategy()
    await strategy.setup_phase()

    issuer.replay_gate.activate.assert_called_once()
    issuer.replay_gate.seed_completed_prefixes.assert_not_called()


@pytest.mark.asyncio
async def test_execute_phase_runs_the_arrival_driver():
    strategy, _, _, _ = _make_arrival_strategy()
    strategy._arrival_driver = MagicMock()
    strategy._arrival_driver.run = AsyncMock()

    await strategy.execute_phase()

    strategy._arrival_driver.run.assert_awaited_once()


# TrajectorySource: open-loop construction


def _open_loop_source(num_traces: int, concurrency: int | None) -> TrajectorySource:
    return TrajectorySource(
        dataset_metadata=_make_dataset(num_traces, 4),
        dataset_sampler=SequentialSampler([f"trace_{i}" for i in range(num_traces)]),
        concurrency=concurrency,
        random_seed=42,
        arrival_driven=True,
    )


def test_arrival_driven_source_builds_no_trajectories():
    src = _open_loop_source(3, None)
    assert src.trajectories == []
    assert src._target_size == 0
    assert src._pool_size == 3


def test_arrival_driven_source_skips_the_dataset_wrap_gate():
    """Open loop samples with replacement by design, so wrap is not a violation."""
    src = _open_loop_source(2, 64)
    assert src.trajectories == []


def test_arrival_driven_source_still_serves_roots_for_admission():
    src = _open_loop_source(2, None)
    assert src.next_recycle_conversation_id() == "trace_0"
    assert src.next_recycle_conversation_id() == "trace_1"


# Config gate


_BASE_BODY = {
    "models": ["test-model"],
    "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
}


def _make_config(**phase_overrides):
    from aiperf.config.config import AIPerfConfig

    return AIPerfConfig(
        benchmark={
            **_BASE_BODY,
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                    **phase_overrides,
                }
            ],
        }
    )


def test_session_arrival_rejected_without_agentic_replay():
    with pytest.raises(ValueError, match="requires the agentic_replay timing mode"):
        _make_config(session_arrival={"rate": 0.5})


def test_session_arrival_accepted_with_explicit_agentic_timing_mode():
    cfg = _make_config(session_arrival={"rate": 0.5}, timing_mode="agentic_replay")
    assert cfg.benchmark.phases[0].session_arrival.rate == 0.5
    assert cfg.benchmark.phases[0].session_arrival.pattern == ArrivalPattern.POISSON


def test_concurrency_burst_arrival_pattern_rejected():
    """Zero inter-arrival time would ignore the rate and spawn without bound."""
    with pytest.raises(ValueError, match="concurrency_burst is incompatible"):
        _make_config(
            session_arrival={"rate": 0.5, "pattern": "concurrency_burst"},
            timing_mode="agentic_replay",
        )


def test_no_session_arrival_accepted():
    cfg = _make_config()
    assert cfg.benchmark.phases[0].session_arrival is None


# Scenario gate


def _spec(*, forbid: bool) -> ScenarioSpec:
    from aiperf.common.scenario.registry import SCENARIOS

    return SCENARIOS["inferencex-agentx-mvp"].model_copy(
        update={"name": "test-scenario", "forbid_session_arrival_rate": forbid}
    )


def _run_with_rate(rate: float | None) -> MagicMock:
    run = MagicMock()
    phase = MagicMock()
    phase.session_arrival = None if rate is None else SessionArrivalConfig(rate=rate)
    run.cfg.get_profiling_phases.return_value = [phase]
    return run


def test_scenario_forbidding_arrivals_records_a_violation():
    violations = []
    _apply_session_arrival_rate(_run_with_rate(0.5), _spec(forbid=True), violations)

    assert len(violations) == 1
    assert violations[0].flag == "--session-arrival-rate"
    assert violations[0].current_value == 0.5


def test_scenario_forbidding_arrivals_ignores_an_unset_rate():
    violations = []
    _apply_session_arrival_rate(_run_with_rate(None), _spec(forbid=True), violations)
    assert violations == []


def test_scenario_not_forbidding_arrivals_records_nothing():
    violations = []
    _apply_session_arrival_rate(_run_with_rate(0.5), _spec(forbid=False), violations)
    assert violations == []


def test_dataset_metadata_helper_is_roots_only():
    """Guard the assumption the open-loop source fixtures rely on."""
    dataset = _make_dataset(2, 3)
    assert dataset.sampling_strategy == DatasetSamplingStrategy.SEQUENTIAL
    assert all(conv.is_root for conv in dataset.conversations)
