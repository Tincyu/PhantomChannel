from tools.pip_v3_contract import PIPContractState


def test_build_is_not_tx_or_ack():
    state = PIPContractState()
    state.enqueue("seq-0")
    state.build("seq-0", generation=1)

    assert state.counters.build_ok == 1
    assert state.counters.tx_completed == 0
    assert state.counters.ack_checked == 0
    assert state.queued_source == "seq-0"
    state.assert_invariants()


def test_incompatible_wait_preserves_head_and_does_not_release():
    state = PIPContractState()
    state.enqueue("seq-0")
    state.build("seq-0", generation=1)

    assert state.prepare_event(
        channel_compatible=False, source="seq-0", generation=1
    ) == "empty"
    assert state.queued_source == "seq-0"
    assert state.counters.wait_incompatible_channel == 1
    assert state.counters.tx_activated == 0
    assert state.counters.node_release == 0
    state.assert_invariants()


def test_source_or_generation_mismatch_defers_without_using_old_dma():
    state = PIPContractState()
    state.enqueue("seq-0")
    state.build("seq-0", generation=1)

    assert state.prepare_event(
        channel_compatible=True, source="seq-1", generation=1
    ) == "empty"
    assert state.prepare_event(
        channel_compatible=True, source="seq-0", generation=2
    ) == "empty"
    assert state.counters.source_mismatch_defer == 2
    assert state.counters.tx_activated == 0
    assert state.queued_source == "seq-0"
    state.assert_invariants()


def test_complete_then_advancing_ack_releases_exactly_once():
    state = PIPContractState()
    state.enqueue("seq-0")
    state.build("seq-0", generation=1)
    assert state.prepare_event(
        channel_compatible=True, source="seq-0", generation=1
    ) == "pip"
    state.complete_tx()

    assert state.counters.tx_completed == 1
    assert state.counters.event_closed == 1
    assert state.check_ack(nesn_advanced=True) == "released"
    assert state.queued_source is None
    assert state.counters.node_release == 1
    assert state.counters.duplicate_release == 0
    assert state.check_ack(nesn_advanced=True) == "ignored"
    state.assert_invariants()


def test_nonadvancing_ack_retains_source_without_release():
    state = PIPContractState()
    state.enqueue("seq-0")
    state.build("seq-0", generation=1)
    assert state.prepare_event(
        channel_compatible=True, source="seq-0", generation=1
    ) == "pip"
    state.complete_tx()

    assert state.check_ack(nesn_advanced=False) == "retained"
    assert state.queued_source == "seq-0"
    assert state.counters.node_release == 0
    state.assert_invariants()


def test_queue_depth_one_rejects_a_second_source_until_ack_release():
    state = PIPContractState()
    state.enqueue("seq-0")
    try:
        state.enqueue("seq-1")
    except OverflowError:
        pass
    else:
        raise AssertionError("a second source bypassed the depth-one queue")
    assert state.queued_source == "seq-0"
