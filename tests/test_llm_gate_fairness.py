"""Tenant-fair admission control in src/llm_gate.py.

Pure-stdlib (threading/time only) — no model load, no network, safe to run
anywhere: `python3 -m unittest tests.test_llm_gate_fairness -v`.

Covers:
  * uncontended callers are admitted immediately (no artificial latency)
  * with size=1 and two tenants contending, admission alternates between
    them (round-robin), not first-come-first-served / one tenant hogging
  * a caller that waits past wait_timeout_s gets LlmGateFull, not a hang
  * callers that never pass tenant_key (existing pipeline_fallback.py
    behavior) still work, landing in the shared "unknown" bucket
  * the circuit breaker (open / half-open probe / close) is unchanged
"""
from __future__ import annotations

import threading
import time
import unittest

from src.llm_gate import LlmCircuitOpen, LlmGate, LlmGateFull


class UncontendedLatency(unittest.TestCase):
    def test_single_caller_admitted_immediately(self):
        gate = LlmGate(size=2, wait_timeout_s=5)
        started = time.monotonic()
        gate.acquire("tenant-a")
        elapsed = time.monotonic() - started
        gate.release()
        self.assertLess(elapsed, 0.05, "uncontended acquire should not block")

    def test_no_tenant_key_defaults_to_unknown_bucket(self):
        """Existing callers (pipeline_fallback.py) call acquire()/slot() with
        no tenant_key at all — must keep working exactly as before."""
        gate = LlmGate(size=1, wait_timeout_s=5)
        gate.acquire()  # no tenant_key, like the pre-fairness call sites
        stats = gate.stats()
        self.assertEqual(stats["in_flight"], 1)
        self.assertIn("unknown", stats["by_tenant"])
        gate.release()


class RoundRobinFairness(unittest.TestCase):
    def test_two_tenants_alternate_under_contention(self):
        """size=1 forces strict serialization. Tenant A holds the slot in a
        tight acquire/release loop; Tenant B joins mid-stream. If admission
        were plain FIFO/first-come, A (already looping, always first back in
        line) would starve B. With round-robin, the two must interleave."""
        gate = LlmGate(size=1, wait_timeout_s=10)
        order: list[str] = []
        order_lock = threading.Lock()
        stop = threading.Event()
        rounds_per_tenant = 8

        def hammer(tenant: str, rounds: int):
            for _ in range(rounds):
                gate.acquire(tenant)
                with order_lock:
                    order.append(tenant)
                time.sleep(0.01)  # hold the slot briefly so the other thread queues up
                gate.release()
                if stop.is_set():
                    return

        ta = threading.Thread(target=hammer, args=("tenant-a", rounds_per_tenant))
        ta.start()
        time.sleep(0.03)  # let A get a head start / establish itself as "already waiting"
        tb = threading.Thread(target=hammer, args=("tenant-b", rounds_per_tenant))
        tb.start()
        ta.join(timeout=15)
        tb.join(timeout=15)
        stop.set()

        self.assertEqual(len(order), rounds_per_tenant * 2, f"not all acquires completed: {order}")

        # The real invariant: no monopoly WHILE BOTH tenants are actually
        # contending. Two legitimate exceptions are not fairness violations:
        # (1) A's deliberate 0.03s head start before B's thread even exists
        #     — trim the leading run before B's first appearance;
        # (2) whichever tenant finishes its rounds first naturally leaves
        #     the other running alone at the end — trim the trailing run
        #     after the other tenant's last appearance.
        first_b = order.index("tenant-b")
        last_a = len(order) - 1 - order[::-1].index("tenant-a")
        contended = order[first_b : last_a + 1]
        self.assertGreater(
            len(contended), 2, f"not enough overlap to test contention: {order}"
        )

        max_consecutive = 1
        run = 1
        for i in range(1, len(contended)):
            if contended[i] == contended[i - 1]:
                run += 1
                max_consecutive = max(max_consecutive, run)
            else:
                run = 1
        self.assertLessEqual(
            max_consecutive, 2,
            f"one tenant monopolized the gate for {max_consecutive} consecutive "
            f"turns while both were contending — round-robin fairness is not "
            f"holding. Full order: {order}, contended region: {contended}",
        )
        # Both tenants must have actually gotten a meaningful share, not just
        # "not zero".
        a_count = order.count("tenant-a")
        b_count = order.count("tenant-b")
        self.assertGreaterEqual(min(a_count, b_count), rounds_per_tenant // 2)

    def test_timeout_raises_gate_full_not_hang(self):
        gate = LlmGate(size=1, wait_timeout_s=0.2)
        gate.acquire("holder")  # take the only slot and never release it here
        started = time.monotonic()
        with self.assertRaises(LlmGateFull):
            gate.acquire("waiter")
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.2)
        self.assertLess(elapsed, 1.5, "should time out close to wait_timeout_s, not hang")
        gate.release()

    def test_new_tenant_joins_rotation_without_waiting_a_full_cycle(self):
        """A brand-new tenant_key with no prior history must be able to get
        an immediately-free slot right away, not wait for some artificial
        'first pass through the rotation'."""
        gate = LlmGate(size=1, wait_timeout_s=2)
        gate.acquire("tenant-a")
        gate.release()
        started = time.monotonic()
        gate.acquire("brand-new-tenant")  # first time this key is ever seen
        elapsed = time.monotonic() - started
        gate.release()
        self.assertLess(elapsed, 0.05)


class CircuitBreakerUnchanged(unittest.TestCase):
    """The tenant-fairness rewrite must not change circuit-breaker behavior
    — it's a shared health signal, not a per-tenant fairness concern."""

    def test_opens_after_consecutive_failures(self):
        gate = LlmGate(size=2, failure_threshold=3, cooldown_s=100, wait_timeout_s=2)
        for _ in range(3):
            gate.acquire("t")
            gate.record_failure()
            gate.release()
        self.assertEqual(gate.circuit, "open")
        with self.assertRaises(LlmCircuitOpen):
            gate.acquire("t")

    def test_success_resets_consecutive_failures(self):
        gate = LlmGate(size=2, failure_threshold=3, cooldown_s=100, wait_timeout_s=2)
        gate.acquire("t")
        gate.record_failure()
        gate.release()
        gate.acquire("t")
        gate.record_success(latency_ms=10)
        gate.release()
        self.assertEqual(gate.circuit, "closed")
        stats = gate.stats()
        self.assertEqual(stats["consecutive_failures"], 0)

    def test_half_open_allows_single_probe(self):
        gate = LlmGate(size=2, failure_threshold=2, cooldown_s=0.1, wait_timeout_s=2)
        for _ in range(2):
            gate.acquire("t")
            gate.record_failure()
            gate.release()
        self.assertEqual(gate.circuit, "open")
        time.sleep(0.15)
        self.assertEqual(gate.circuit, "half_open")
        gate.acquire("t")  # the one allowed probe
        with self.assertRaises(LlmCircuitOpen):
            gate.acquire("other")  # a second concurrent probe must be rejected
        gate.record_success(latency_ms=5)
        gate.release()
        self.assertEqual(gate.circuit, "closed")


if __name__ == "__main__":
    unittest.main()
