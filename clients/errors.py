"""Shared client exceptions."""


class APICallFailed(RuntimeError):
    """Raised by a model client when all retries are exhausted.

    Lets the eval harness tell a genuine infrastructure failure apart from a real
    (even if empty or wrong) model answer, and SKIP the item — leaving it absent
    from the CSV so it is retried on the next resume — instead of writing a poisoned
    row whose `raw_response` is an error string rather than real model output. This
    keeps every committed `raw_response` a true API output (RESEARCH_PLAN §1.0 rule 6).
    """
