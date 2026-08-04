# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open-loop session-arrival admission for agentic trace replay.

Agentic replay is normally CLOSED loop: ``--concurrency`` trajectory lanes are
held occupied for the whole phase, and a lane that drains immediately recycles
into a fresh root. In-system session count is therefore constant by
construction.

This module drives the OPEN-loop alternative. Session arrivals are generated
by an exogenous point process (Poisson by default) at rate ``lambda_s`` and are
completely decoupled from completions: an arrival admits one fresh trace replay
starting at turn 0, and a drained tree admits nothing. In-system session count
is then a random variable determined by ``lambda_s`` and the per-session
residence time, which is what an M/G/inf-style workload model requires.

Only SESSION ARRIVALS are metered. Everything a session does after admission --
main-turn continuations at ``completion(turn k) + recorded think time``,
subagent fan-out at its recorded group offsets, subagent inner turns at their
own recorded gaps -- stays on the closed-loop causal path owned by
:class:`~aiperf.timing.strategies.agentic_replay.AgenticReplayStrategy` and the
:class:`~aiperf.timing.branch_orchestrator.BranchOrchestrator`. A simultaneous
5-request fan-out is dispatched as a simultaneous 5-request fan-out regardless
of ``lambda_s``; the arrival rate never thins endogenous traffic.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.utils import yield_to_event_loop
from aiperf.plugin import plugins
from aiperf.plugin.enums import ArrivalPattern, PluginType
from aiperf.timing.intervals import IntervalGeneratorConfig

if TYPE_CHECKING:
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy


class SessionArrivalDriver(AIPerfLoggerMixin):
    """Admits fresh trace-replay sessions on an exogenous arrival process.

    One instance per PROFILING phase, owned by the agentic-replay strategy.
    The driver only decides *when* a session starts and *whether* the
    concurrency ceiling allows it; the strategy owns everything about *what*
    that session then does.
    """

    def __init__(
        self,
        *,
        strategy: AgenticReplayStrategy,
        credit_issuer: CreditIssuer,
        stop_checker: StopConditionChecker,
        lifecycle: PhaseLifecycle,
        rate: float,
        pattern: ArrivalPattern = ArrivalPattern.POISSON,
        smoothness: float | None = None,
    ) -> None:
        super().__init__(logger_name="SessionArrival")
        self._strategy = strategy
        self._credit_issuer = credit_issuer
        self._stop_checker = stop_checker
        self._lifecycle = lifecycle
        self._rate = rate
        self._pattern = pattern

        interval_config = IntervalGeneratorConfig(
            arrival_pattern=pattern,
            request_rate=rate,
            arrival_smoothness=smoothness,
        )
        GeneratorClass = plugins.get_class(PluginType.ARRIVAL_PATTERN, pattern)
        self._generator = GeneratorClass(interval_config)

        self._arrivals = 0
        self._admitted = 0
        self._rejected_overload = 0
        self._rejected_unspawnable = 0
        self._behind_schedule = 0
        self._overload_warned = False

    @property
    def stats_summary(self) -> str:
        """One-line accounting of the arrival process, logged at phase end."""
        return (
            f"Session arrivals: pattern={self._pattern} rate={self._rate:g}/s, "
            f"generated={self._arrivals} admitted={self._admitted} "
            f"rejected_overload={self._rejected_overload} "
            f"rejected_unspawnable={self._rejected_unspawnable} "
            f"behind_schedule_resets={self._behind_schedule}"
        )

    async def run(self) -> None:
        """Generate arrivals until the phase's stop conditions are reached.

        Absolute (drift-free) scheduling: the next arrival instant is advanced
        by a freshly drawn inter-arrival time BEFORE the admission work, so
        variable admission latency does not bias the realized rate. Falling
        behind resets the target to now rather than firing a catch-up burst --
        catch-up bursts would corrupt the inter-arrival distribution far more
        than the dropped idle time does.

        Returning does not end the phase. Endogenous traffic from every
        already-admitted session keeps flowing; the phase completes on its own
        stop conditions (``--benchmark-duration`` / ``--request-count`` /
        ``--session-count``) exactly as it does in closed-loop replay.
        """
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        self.info(
            f"PROFILING execute: open-loop session arrivals "
            f"(pattern={self._pattern}, rate={self._rate:g} sessions/s); "
            "continuations, subagent fan-out and inner turns stay on the "
            "recorded causal path and are not metered by the arrival rate"
        )

        perf_start = self._lifecycle.started_at_perf_ns / NANOS_PER_SECOND
        next_target_perf = perf_start + self._generator.next_interval()

        while True:
            now = time.perf_counter()
            if next_target_perf < now:
                self._behind_schedule += 1
                next_target_perf = now

            sleep_duration = next_target_perf - now
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            else:
                # Zero-interval patterns (CONCURRENCY_BURST) would otherwise
                # busy-loop and starve the credit-return callbacks this phase
                # depends on to make progress.
                await yield_to_event_loop()

            next_target_perf += self._generator.next_interval()

            # Session budget exhausted. Already-admitted sessions may still have
            # turns left to send, so this only stops arrival generation; the
            # phase itself ends on its own stop conditions.
            if not self._stop_checker.can_start_new_session():
                return

            if not await self._admit_one():
                return

    async def _admit_one(self) -> bool:
        """Admit a single arrival. Returns False when the loop must stop.

        ``can_start_new_session`` was checked by the caller with no intervening
        await, so a negative ``can_acquire_and_start_new_session`` here is
        attributable to the session-slot ceiling alone -- i.e. genuine
        overload, where the offered arrival rate exceeds what ``--concurrency``
        admits. The arrival is rejected rather than queued: blocking on the
        slot would silently convert the open loop back into a closed one and
        make the realized in-system session count a function of the server's
        speed instead of ``lambda_s``.
        """
        index = self._arrivals
        self._arrivals += 1

        if not self._credit_issuer.can_acquire_and_start_new_session():
            self._rejected_overload += 1
            self._warn_overload_once()
            return True

        admitted = await self._strategy.admit_session_arrival(index)
        if admitted is None:
            self._rejected_unspawnable += 1
            return True
        if admitted is False:
            return False
        self._admitted += 1
        return True

    def _warn_overload_once(self) -> None:
        """Warn the first time an arrival is rejected by the session ceiling."""
        if self._overload_warned:
            return
        self._overload_warned = True
        self.warning(
            lambda: (
                f"Session arrival rejected at the concurrency ceiling "
                f"(arrival {self._arrivals}): the offered rate "
                f"{self._rate:g} sessions/s exceeds what the configured "
                "--concurrency admits, so the realized arrival process is "
                "truncated and no longer matches the requested lambda. Raise "
                "--concurrency (or drop it entirely for an uncapped open "
                "loop), or lower --session-arrival-rate. Further rejections "
                "are counted, not logged."
            )
        )
