# Fixtures schema (SPEC §7.1)

Each fixture .json is a list of turn records used to drive
`smoke_test_middleware_e2e`-style tests. Top-level schema:

```json
{
  "version": 1,
  "turns": [
    {
      "messages": [...],      // Anthropic-shape messages list for this turn
      "system": "...",        // optional; can be omitted on append turns
      "tools": [...],         // optional; only on first turn typically
      "raw_output_ids": [142, 891, ...]   // ids the mocked sglang returns
    }
  ],
  "expected_segments": [
    {"segment_kind": "pre_wipe", "response_len": 12044, "completed_turns": 5},
    {"segment_kind": "final",    "response_len": 4201,  "completed_turns": 3}
  ]
}
```

Tests use these to:
1. boot a Session, replay each turn through `_handle_messages` (mocked sglang)
2. call `pop_session_split(session)`
3. assert observed `segment_kind` sequence matches `expected_segments`

`raw_output_ids` arrays are short integer sequences encoding "what sglang
returned this turn" — used by the mock to simulate generation deterministically.
