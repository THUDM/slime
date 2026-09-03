<!-- Vendored from marin-community/marin-style v0.4.0 — do not edit; re-run `marin-style sync`. -->

# Testing Guidelines (Core)

Portable, behavior-focused testing policy shared across Marin-community
repositories. Read `AGENTS.md` and this file before writing or reviewing tests.

## Core Rule

A test must fail when behavior is wrong. It should not fail only because an
implementation detail, wording choice, helper call, or command assembly changed.

Prefer integration-style tests that validate externally observable behavior:

- public API return values
- structured output or machine-readable fields
- persisted state
- real side effects in a temp directory or in-memory fake
- state transitions visible through the public event/status API
- numerical value and gradient parity against an independent reference

It is better to not have a test than to have "slop" tests (tautological or other
useless tests). See below.

## Good Targets

- A regression for a reported bug.
- A boundary case where user-visible behavior changes: empty input, duplicate
  names, incompatible shapes, quota or capacity limits, timeout behavior, failed
  workers, bad files, or invalid configs.
- A round trip through the public API: submit a job, observe status, write
  chunks, read chunks, serialize, deserialize, reload.
- A stable contract: wire format, schema field, CLI JSON output, checkpoint
  layout, or public exception type.
- For numerical code, parity against an obvious reference implementation across
  a small shape/dtype grid.

## Slop Tests To Reject

Delete or rewrite tests with negative value:

- Tautologies: constant equals itself, method exists, object constructor assigns
  attributes, `len(list) >= 0`, type exists. This definition should be construed
  broadly.
- Private state: assertions on private attributes or `_`-prefixed state.
- Incidental strings: assertions on human log text, progress messages, command
  fragments, or copied prose.
- Internal helper dispatch: assertions on `assert_called_once_with`, call count,
  or "helper X was invoked" for in-process helpers.
- Reimplementation: Tests that reimplement the production logic and compare the
  implementation to itself.
- Obvious error-condition behavior: tests that only prove a type checker,
  dataclass constructor, or standard library function works. Infrastructure
  failure modes are valid when the failure is externally observable.
- Tests for Python language semantics.
- Registration tests: Tests that check that specific items are registered in
  global registries.
- Permanently skipped tests or empty test files.
- Screenshot-only tests without behavioral assertions.
- "Does not raise" tests without a comment explaining why that is the contract.
- "Does raise" tests if the exception type is not part of the contract or an
  important regression signal.

Disposable smoke probes are different from checked-in tests. During development,
use scratch scripts, REPL snippets, or temporary local assertions to confirm
imports, object construction, or fixture wiring. Do not leave those as pytest
tests. Before a PR-ready commit, replace them with behavior assertions or delete
them.

String assertions are allowed when the string is the contract: a wire format,
machine-readable output, a downstream-parsed log line, or an exact user-facing
error promised by the API. Assert on structured fields when possible.

Command construction is rarely the contract. Prefer to run through the boundary
and assert the effect. If the command line is the contract, assert the parsed
argv or structured command object, not a substring in rendered shell text.

## Mocks And Fakes

Default to real behavior. Use mocks only at I/O boundaries:

- subprocesses such as `gcloud` or `kubectl`
- HTTP or remote APIs
- GCS, W&B, or other external services
- filesystem boundaries that would be slow, expensive, or destructive

Prefer fakes over mocks when practical: in-memory services, temp directories,
local clients, fake schedulers, and fake clocks. Check for existing fakes before
adding a new one.

Do not mock internal functions to prove wiring. If a side effect matters, expose
or observe it through a public API or use a fake that records stable state. If
this is difficult, it may be a sign that you shouldn't test that behavior or that
the API needs to be redesigned to make it testable without mocks.

## Timing

Do not use `time.sleep()` in tests. Inject `now=time.time()`, use a fake clock,
or use existing deadline/backoff helpers.

Use deadline, backoff, polling, or fake-clock helpers when they exist. A single
short sleep to let a background thread start is acceptable only with a comment
that names the race being avoided.

## Numerical Tests

Do not relax tolerances without human agreement. Exact tolerance values may be
component-specific; the global rule is that agents do not weaken them without
explicit agreement from the user.

Use slow-test markers and optional-dependency markers as appropriate.

For kernels, start from an obvious reference: existing in-repo code, a PyTorch
reference, Optax/JAX baseline, or clear pseudocode. Check:

- value parity over a small shape/dtype grid
- gradient parity on small shapes
- CPU numerics and accelerator-aligned fast paths when applicable
- pointwise deviation metrics such as max and mean absolute difference, not only
  `allclose`
- If sharding is relevant, check parallelization and shard-local numerics
  separately.

Keep the reference independent and simple. A clever reference that shares the
same bug as the implementation is not useful.

## Pytest Style

- Prefer top-level `def test_*` functions with fixtures over test classes.
- Name tests as `test_<subject>_<scenario>_<expected_outcome>`.
- Use parameterization for meaningful behavior variation.
- Extend existing test files before creating new ones.
- Every test must contain an assertion or `pytest.raises`.
- Remove dead helpers, unused fakes, unused imports, and empty stubs.
- Mark slow, Docker, and cluster-dependent tests with the package's established
  markers.
- For non-trivial public classes with protocols, test the protocol behavior
  rather than concrete private state.

## Review Checklist

When reviewing tests, flag the test if the answer to any question is "yes":

- Would this test pass if the behavior were wrong?
- Would this test fail only because a helper was renamed or a log line changed?
- Is it asserting on private state, call counts, or internal dispatch?
- Does it duplicate the implementation instead of checking an independent oracle
  or observable behavior?
- Is the mock inside the system under test rather than at an external boundary?
- Does it sleep, skip permanently, or lack a real assertion?
- Did the change weaken a numerical tolerance or test expectation without
  justification?

Ask for low-value tests to be deleted or rewritten. More tests are not better
when they pin implementation details and miss real regressions.
