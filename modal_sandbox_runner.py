"""
Modal Sandbox Runner for RLM_Sandbox-Model
==========================================
A secure, isolated sandbox runner for testing RLM functionality on Modal.com.

For SVCSTACK / Paul McConville
Created: 2026-02-09

Usage:
    python modal_sandbox_runner.py              # Run basic test
    python modal_sandbox_runner.py --test       # Run connectivity test only
    python modal_sandbox_runner.py --timeout 300  # Custom timeout (seconds)

Prerequisites:
    1. modal installed: pip install modal
    2. modal authenticated: modal setup
    3. rlm installed: pip install -e ".[modal]"
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def preflight_checks() -> bool:
    """Verify all prerequisites before running."""
    errors = []

    # Check modal is importable
    try:
        import modal  # noqa: F401
    except ImportError:
        errors.append("modal is not installed. Run: pip install modal")

    # Check rlm is importable
    try:
        import rlm  # noqa: F401
    except ImportError:
        errors.append("rlm is not installed. Run: pip install -e '.[modal]'")

    # Check dill is importable
    try:
        import dill  # noqa: F401
    except ImportError:
        errors.append("dill is not installed. Run: pip install dill")

    if errors:
        print("PRE-FLIGHT CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("Pre-flight checks passed.")
    return True


def test_modal_connection() -> bool:
    """Test that Modal is authenticated and reachable."""
    try:
        import modal

        # This will fail if not authenticated
        app = modal.App.lookup("rlm-svcstack-test", create_if_missing=True)
        print(f"Modal connection OK. App: {app}")
        return True
    except Exception as e:
        print(f"Modal connection FAILED: {e}")
        return False


def run_basic_sandbox_test(timeout: int = 600) -> bool:
    """
    Run a basic test inside a Modal sandbox.
    This verifies:
    - Sandbox creation works
    - Code execution works
    - State persistence works
    - Sandbox cleanup works
    """
    from rlm.environments.modal_repl import ModalREPL

    print(f"\nStarting Modal sandbox test (timeout={timeout}s)...")
    start = time.perf_counter()

    try:
        with ModalREPL(
            app_name="rlm-svcstack-test",
            timeout=timeout,
        ) as repl:
            # Test 1: Basic arithmetic
            print("  [1/4] Testing basic code execution...")
            result = repl.execute_code("x = 2 + 3")
            assert "x" in result.locals, f"Expected 'x' in locals, got: {result.locals}"
            print(f"         OK: x = {result.locals.get('x')}")

            # Test 2: Print output
            print("  [2/4] Testing stdout capture...")
            result = repl.execute_code("print('Hello from Modal sandbox!')")
            assert "Hello" in result.stdout, f"Expected 'Hello' in stdout, got: {result.stdout}"
            print(f"         OK: stdout = {result.stdout.strip()}")

            # Test 3: State persistence between executions
            print("  [3/4] Testing state persistence...")
            repl.execute_code("my_list = [1, 2, 3]")
            result = repl.execute_code("print(sum(my_list))")
            assert "6" in result.stdout, f"Expected '6' in stdout, got: {result.stdout}"
            print(f"         OK: sum(my_list) = {result.stdout.strip()}")

            # Test 4: FINAL_VAR
            print("  [4/4] Testing FINAL_VAR...")
            repl.execute_code("answer = 'sandbox works'")
            result = repl.execute_code('print(FINAL_VAR("answer"))')
            assert "sandbox works" in result.stdout, f"Expected answer, got: {result.stdout}"
            print(f"         OK: FINAL_VAR = {result.stdout.strip()}")

        elapsed = time.perf_counter() - start
        print(f"\nAll tests passed in {elapsed:.1f}s. Sandbox terminated cleanly.")
        return True

    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"\nTest FAILED after {elapsed:.1f}s: {e}")
        return False


def run_rlm_sandbox_demo(timeout: int = 600) -> None:
    """
    Run a simple RLM completion inside a Modal sandbox.
    This demonstrates the full RLM flow without needing an LLM API key.
    Uses a mock LLM client for safety.
    """
    from rlm.clients.base_lm import BaseLM
    from rlm.core.lm_handler import LMHandler
    from rlm.core.types import ModelUsageSummary, UsageSummary
    from rlm.environments.modal_repl import ModalREPL

    class MockLM(BaseLM):
        """Mock LLM that echoes prompts. No API key needed."""

        def __init__(self):
            super().__init__(model_name="mock-model")

        def completion(self, prompt):
            if isinstance(prompt, list):
                prompt = str(prompt[-1].get("content", ""))[:100]
            return f"Mock response to: {str(prompt)[:100]}"

        async def acompletion(self, prompt):
            return self.completion(prompt)

        def get_usage_summary(self):
            return UsageSummary(
                model_usage_summaries={
                    "mock-model": ModelUsageSummary(
                        total_calls=1, total_input_tokens=10, total_output_tokens=10
                    )
                }
            )

        def get_last_usage(self):
            return self.get_usage_summary()

    print("\nRunning RLM demo with Mock LLM in Modal sandbox...")
    mock_client = MockLM()

    with LMHandler(client=mock_client) as handler:
        print(f"  LM Handler started at {handler.address}")

        with ModalREPL(
            app_name="rlm-svcstack-demo",
            lm_handler_address=handler.address,
            timeout=timeout,
        ) as repl:
            # Test llm_query from inside sandbox
            result = repl.execute_code('response = llm_query("Test query from sandbox")')
            print(f"  llm_query stderr: {result.stderr or '(none)'}")

            result = repl.execute_code("print(response)")
            print(f"  llm_query result: {result.stdout.strip()}")

            # Test batched query
            result = repl.execute_code(
                'responses = llm_query_batched(["Q1", "Q2"])'
            )
            result = repl.execute_code("print(len(responses))")
            print(f"  Batched responses: {result.stdout.strip()}")

    print("\nDemo complete. Sandbox terminated cleanly.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLM Modal Sandbox Runner")
    parser.add_argument("--test", action="store_true", help="Run connectivity test only")
    parser.add_argument("--demo", action="store_true", help="Run full RLM demo with mock LLM")
    parser.add_argument("--timeout", type=int, default=600, help="Sandbox timeout in seconds (default: 600)")
    args = parser.parse_args()

    if not preflight_checks():
        sys.exit(1)

    if args.test:
        success = test_modal_connection()
        if success:
            success = run_basic_sandbox_test(timeout=args.timeout)
        sys.exit(0 if success else 1)

    if args.demo:
        if not test_modal_connection():
            sys.exit(1)
        run_rlm_sandbox_demo(timeout=args.timeout)
        sys.exit(0)

    # Default: run basic test
    if not test_modal_connection():
        sys.exit(1)
    run_basic_sandbox_test(timeout=args.timeout)
