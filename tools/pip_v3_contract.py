"""Small executable contract model for the v3 PIP queue state machine.

This module is intentionally independent of the Nordic controller.  It is used
by offline tests to pin down the accounting rules that the controller logs:
building a packet is not a TX, an incompatible event is an empty wait, and a
queued source is released only after one advancing ACK.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PIPCounters:
    app_enqueued: int = 0
    build_ok: int = 0
    wait_incompatible_channel: int = 0
    source_mismatch_defer: int = 0
    tx_activated: int = 0
    tx_completed: int = 0
    event_closed: int = 0
    ack_checked: int = 0
    ack_nesn_advanced: int = 0
    ack_nesn_not_advanced: int = 0
    node_release: int = 0
    duplicate_release: int = 0
    scratch_canary_fail: int = 0


@dataclass
class PIPContractState:
    """Deterministic model of one queued PIP source and its lifecycle."""

    queued_source: str | None = None
    prepared_source: str | None = None
    prepared_generation: int | None = None
    active_generation: int | None = None
    pending_ack: bool = False
    released: bool = False
    counters: PIPCounters = field(default_factory=PIPCounters)

    def enqueue(self, source: str) -> None:
        if self.queued_source is not None:
            raise OverflowError("v3 queue depth is one")
        self.queued_source = source
        self.released = False
        self.counters.app_enqueued += 1

    def build(self, source: str, generation: int) -> None:
        if source != self.queued_source:
            raise ValueError("builder source is not the queue head")
        self.prepared_source = source
        self.prepared_generation = generation
        self.counters.build_ok += 1

    def prepare_event(self, *, channel_compatible: bool, source: str, generation: int) -> str:
        """Return ``pip`` or ``empty`` without changing queue ownership."""

        if not channel_compatible:
            self.counters.wait_incompatible_channel += 1
            return "empty"
        if (
            source != self.prepared_source
            or generation != self.prepared_generation
            or source != self.queued_source
        ):
            self.counters.source_mismatch_defer += 1
            return "empty"
        self.active_generation = generation
        self.counters.tx_activated += 1
        return "pip"

    def complete_tx(self) -> None:
        if self.active_generation is None:
            raise ValueError("cannot complete an inactive PIP")
        self.active_generation = None
        self.pending_ack = True
        self.counters.tx_completed += 1
        self.counters.event_closed += 1

    def check_ack(self, *, nesn_advanced: bool) -> str:
        if not self.pending_ack:
            return "ignored"
        self.counters.ack_checked += 1
        if not nesn_advanced:
            self.counters.ack_nesn_not_advanced += 1
            return "retained"
        self.counters.ack_nesn_advanced += 1
        self.pending_ack = False
        if self.released:
            self.counters.duplicate_release += 1
            return "duplicate"
        self.released = True
        self.queued_source = None
        self.counters.node_release += 1
        return "released"

    def assert_invariants(self) -> None:
        c = self.counters
        assert c.tx_completed <= c.tx_activated <= c.build_ok
        assert c.event_closed == c.tx_completed
        assert c.ack_nesn_advanced <= c.ack_checked <= c.tx_completed
        assert c.node_release <= c.ack_nesn_advanced
        assert c.duplicate_release == 0
        assert c.scratch_canary_fail == 0
