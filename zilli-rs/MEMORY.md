# MEMORY — Zilli Rust Fix Session 2026-06-24

## What Changed

### P0 — All 5 Critical issues fixed
1. **HTTP timeout**: All 3 model backends (vllm/ollama/llamacpp) now use `Client::builder().timeout(60s)`
2. **ReDoS**: Address regex (`[A-Za-z0-9\s,]+` → `[A-Za-z0-9,]+(?:\s+...)*`), phone regex (capture→non-capture), bias/dangerous patterns (atomic groups), input size capped at 1MB
3. **Cache bypass**: Cached responses now go through `output_sanitizer.check()` before return
4. **Div by zero**: `CompositeVerifier` confidence uses `results.len().max(1)` guard
5. **unwrap/expect**: Server returns `Result` with graceful error logging; semaphore unwrap→Failed result; `duration_since` uses `unwrap_or_default`; NaN `partial_cmp` uses `unwrap_or(Equal)`

### P1 — Medium issues
- **Config wiring**: `load_config()` now called at startup with warning on missing/malformed file
- **Clock trait**: `infra::clock::{Clock, RealClock, MockClock}` created; `CacheEngine::with_clock()` for testability
- **Enum consistency**: `ActionType` gets `PartialEq+Eq`; `ModelRole`/`DeploymentType`/`RouteType` get snake_case serde + `Eq`
- **CLI warnings**: Removed 3 unnecessary `mut`s, fixed parens

### P2 — Low issues
- **purify()**: Return value logic corrected (tracks golden+failed purges separately)
- **Reward boundaries**: `>=0.8` / `<=0.3` instead of `>0.8` / `<0.3` (catches edge values)

### Tests
- 4 new cache engine tests (set/get, miss, expiry, disabled)
- All 15 tests pass, 0 compilation errors, 45 warnings (pre-existing)

## Remaining
- P2.2: mockall + core component unit tests
- Pre-existing warnings from dead code fields (45 total)
