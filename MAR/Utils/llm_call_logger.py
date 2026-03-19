"""
LLM Call Logger — instruments LLM.gen() to record timing, model, and
prompt/response sizes.  Designed for sanity-testing / debugging runs.

Usage:
    from MAR.Utils.llm_call_logger import LLMCallLogger

    llm_logger = LLMCallLogger()
    llm_logger.enable()              # start recording
    # ... run your MasRouter forward pass ...
    llm_logger.print_sample_summary("Sample 1")
    llm_logger.reset_sample()        # clear per-sample counters
    # ... next sample ...
    llm_logger.print_total_summary() # overall stats
    llm_logger.disable()             # stop recording (restores originals)
"""

import time
from collections import defaultdict
from loguru import logger


class LLMCallLogger:
    """Monkey-patches all concrete LLM classes to add timing instrumentation."""

    def __init__(self):
        self._enabled = False
        self._patched_classes = {}   # cls -> original_gen
        self._sample_calls = []      # calls within current sample
        self._total_calls = []       # all calls across the run

    # ── Public API ─────────────────────────────────────────────────────────

    def enable(self):
        """Patch LLM classes and start recording."""
        if self._enabled:
            return
        self._enabled = True
        self._patch_all()

    def disable(self):
        """Restore original gen() methods and stop recording."""
        if not self._enabled:
            return
        self._enabled = False
        self._unpatch_all()

    def reset_sample(self):
        """Clear per-sample call list (keeps total)."""
        self._sample_calls = []

    def get_sample_calls(self):
        """Return list of call dicts for the current sample."""
        return list(self._sample_calls)

    # ── Summaries ──────────────────────────────────────────────────────────

    def sample_summary(self, label: str = "") -> str:
        """Return a formatted summary for the current sample's LLM calls."""
        return self._format_summary(self._sample_calls, label or "Sample")

    def total_summary(self) -> str:
        """Return a formatted summary for all LLM calls so far."""
        return self._format_summary(self._total_calls, "Total")

    def print_sample_summary(self, label: str = ""):
        logger.info("\n" + self.sample_summary(label))

    def print_total_summary(self):
        logger.info("\n" + self.total_summary())

    # ── Batch-level helpers ────────────────────────────────────────────────

    def get_total_call_count(self) -> int:
        return len(self._total_calls)

    def get_total_time(self) -> float:
        return sum(c['elapsed'] for c in self._total_calls)

    # ── Internals ──────────────────────────────────────────────────────────

    def _patch_all(self):
        """Find and patch all concrete LLM subclasses imported so far."""
        from MAR.LLM.gpt_chat import ALLChat, DSChat
        # These two cover all models routed by LLMRegistry.get()
        for cls in (ALLChat, DSChat):
            if cls not in self._patched_classes:
                original_gen = cls.gen
                self._patched_classes[cls] = original_gen
                cls.gen = self._make_timed_gen(original_gen)

    def _unpatch_all(self):
        """Restore original gen() methods."""
        for cls, original_gen in self._patched_classes.items():
            cls.gen = original_gen
        self._patched_classes.clear()

    def _make_timed_gen(self, original_gen):
        """Create a wrapper that times gen() and logs each call."""
        logger_ref = self  # avoid shadowing `logger` from loguru

        def timed_gen(llm_self, messages, *args, **kwargs):
            start = time.time()
            result = original_gen(llm_self, messages, *args, **kwargs)
            elapsed = time.time() - start

            prompt_text = (
                "".join(m['content'] for m in messages)
                if isinstance(messages, list)
                else str(messages)
            )
            resp_text = str(result) if result else ""

            call_info = {
                'call_id': len(logger_ref._total_calls) + 1,
                'model': llm_self.model_name,
                'elapsed': elapsed,
                'prompt_chars': len(prompt_text),
                'response_chars': len(resp_text),
            }
            logger_ref._sample_calls.append(call_info)
            logger_ref._total_calls.append(call_info)

            logger.info(
                f'  🔹 LLM call #{call_info["call_id"]:>3d} │ '
                f'model={llm_self.model_name:<30s} │ '
                f'{elapsed:>6.1f}s │ '
                f'prompt≈{len(prompt_text):>6,d} chars │ '
                f'response≈{len(resp_text):>5,d} chars'
            )
            return result

        return timed_gen

    @staticmethod
    def _format_summary(calls, label: str) -> str:
        if not calls:
            return f"  📊 {label} LLM Call Summary: (no calls)"

        total_time = sum(c['elapsed'] for c in calls)
        total_prompt = sum(c['prompt_chars'] for c in calls)
        total_resp = sum(c['response_chars'] for c in calls)

        by_model = defaultdict(list)
        for c in calls:
            by_model[c['model']].append(c)

        lines = [
            f"  📊 {label} LLM Call Summary",
            f"  {'─' * 60}",
            f"  Total calls: {len(calls)}   │   Total time: {total_time:.1f}s   │"
            f"   Avg: {total_time / len(calls):.1f}s/call",
            f"  Total prompt: {total_prompt:,d} chars   │   "
            f"Total response: {total_resp:,d} chars",
            f"  {'─' * 60}",
        ]
        for model, model_calls in sorted(by_model.items()):
            model_time = sum(c['elapsed'] for c in model_calls)
            model_avg = model_time / len(model_calls)
            lines.append(
                f"  {model:<30s} │ {len(model_calls):>3d} calls │ "
                f"total {model_time:>6.1f}s │ avg {model_avg:>5.1f}s"
            )
        lines.append(f"  {'─' * 60}")
        return "\n".join(lines)
