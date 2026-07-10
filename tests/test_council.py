# tests for the council's pure logic.
# we test the deterministic bits (routing + parsing), never the agents themselves,
# those are non-deterministic and cost money to run.

from agent.council import route, parse_verdict


# route() decides whether the graph loops back to the architect or ends

def test_route_ends_when_consistent():
    # keeper approved the lore, so the graph should finish
    state = {"is_consistent": True, "attempts": 1}
    assert route(state) == "done"


def test_route_loops_when_inconsistent():
    # keeper found a contradiction and we still have retries left,
    # so send it back to the architect with feedback
    state = {"is_consistent": False, "attempts": 1}
    assert route(state) == "loop"


def test_route_stops_at_retry_cap():
    # even if it's still inconsistent, we give up after 3 attempts so a stubborn contradiction can't loop forever
    state = {"is_consistent": False, "attempts": 3}
    assert route(state) == "done"


# --- parse_verdict() turns the keeper's text reply into a bool ---

def test_consistent_is_approved():
    assert parse_verdict("CONSISTENT, the new lore fits the canon.") is True


def test_contradiction_is_rejected():
    assert parse_verdict("CONTRADICTION, Brock was the first to summit.") is False


def test_case_and_whitespace_are_ignored():
    # the parse strips and uppercases, so messy casing still reads as approval
    assert parse_verdict("  consistent - looks good") is True


def test_empty_reply_is_rejected():
    # a blank reply is not an approval, so we default to rejecting
    assert parse_verdict("") is False


def test_must_start_with_the_marker():
    # documents a real limitation: the marker has to be at the START.
    # if the keeper buries "consistent" mid-sentence instead of leading with it,
    # we treat it as a rejection, safer to reject than to wrongly approve.
    assert parse_verdict("The proposal is consistent with canon.") is False
