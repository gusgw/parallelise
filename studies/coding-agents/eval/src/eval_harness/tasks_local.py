"""The concrete tasks the agents were scored on in the study.

Loaded by eval_harness.tasks._load_tasks() if present. Each task names a target
codebase from harness_common.targets_local and describes the work in plain words.
"""
from __future__ import annotations

from eval_harness.tasks import (
    Task, check_file_contains, check_frozen_test, check_tests_pass)

TASKS = [
    Task("debug-best-weather", "best_weather", "debug",
         "The test suite fails. Use the tools to find and fix the cause so the "
         "tests pass. Make minimal changes.",
         check=check_tests_pass(),
         # narrow reset: only revert the task's files, never the user's other edits
         reset=[["git", "checkout", "--", "test_pipeline.py", "pipeline.py"]]),
    Task("test-mealworm", "mealworm", "test",
         "Add a pytest test for the S3-reference middleware's handling of empty "
         "metadata, then run the tests so they pass.",
         check=check_tests_pass()),
    Task("document-weather", "weather", "document",
         "Document the pipeline stages (topics in/out, message models) from the "
         "code and the AsyncAPI topology. Write it to STAGES.md.",
         check=check_file_contains("STAGES.md", "topic")),
    Task("review-s3ref", "mealworm", "review",
         "Review the S3-reference middleware for correctness and edge cases. "
         "Write your findings to REVIEW.md.",
         check=check_file_contains("REVIEW.md", "s3")),
    Task("optimise-s3ref", "mealworm", "optimise",
         "The S3-reference middleware uploads every message to S3 with no size "
         "threshold. Add an inline-vs-S3 size threshold and keep the tests green.",
         check=check_tests_pass()),
    # --- harder, temperature-study tasks (each on its own mealworm/weather branch) ---
    Task("feature-inline-threshold", "mealworm", "feature",
         "The test suite has a failing test specifying an opt-in inline-vs-S3 size "
         "threshold for the message middleware. When the environment variable "
         "MEALWORM_S3_INLINE_THRESHOLD is set, a message whose pickled content+metadata "
         "is smaller than it must travel inline: not uploaded to S3, its fields left on "
         "the Kafka wire, and s3_ref left None; it must be restored on the consume side "
         "without touching S3. At or above the threshold, and whenever the variable is "
         "unset, behaviour must be unchanged (via S3). Implement this in the pipeline "
         "source so the whole suite passes. Do not edit the tests.",
         check=check_frozen_test(["tests"]),
         reset=[["git", "checkout", "--", "src/mealworm/pipeline_inline.py", "tests"]]),
    Task("debug-s3-retry", "weather", "debug",
         "The S3 client (lib/s3_client.py) has no retry logic: a single transient "
         "connection or timeout failure aborts the whole job. Add exponential-backoff "
         "retries to download_file and upload_file so transient failures (connection / "
         "timeout errors) are retried a few times, but the exception still propagates "
         "once the retries are exhausted. A call that succeeds on the first try must not "
         "be retried. Make the failing tests pass without editing the tests.",
         check=check_frozen_test(["tests"]),
         reset=[["git", "checkout", "--", "lib/s3_client.py", "tests"]]),
    Task("debug-msgid-trace", "mealworm", "debug",
         "A test fails: a message's message_id is not preserved across pipeline "
         "stages. The S3-reference middleware's consume step in pipeline_inline.py "
         "rebuilds the message after fetching its content back from S3, and that "
         "rebuild does not carry the original message_id through. Fix the consume "
         "step so message_id stays stable end to end. Make the failing test pass "
         "without editing the tests.",
         check=check_frozen_test(["tests"]),
         reset=[["git", "checkout", "--", "src/mealworm/pipeline_inline.py", "tests"]]),
]
