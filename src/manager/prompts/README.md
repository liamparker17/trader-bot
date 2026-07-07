# Manager system prompts

Versioned system-prompt files for the Claude portfolio manager
(`src/manager/client.py`), plus the champion pointer used to pick one.

## Files

- `vNNN.md` — one prompt variant per file (`v001.md`, `v002.md`, …).
  Never deleted, even after losing a prompt-lab run (audit trail).
- `champion.txt` — plain text containing exactly one filename
  (e.g. `v001.md`): the prompt the live manager uses.

## Loading rules (`load_champion_prompt`)

1. If `champion.txt` exists and names a file that exists → that file wins,
   even if higher-numbered versions exist.
2. If `champion.txt` is missing, empty, or names a missing file → fall back
   to the highest `vNNN.md` by numeric (not lexical) sort, with a warning
   logged for the invalid-pointer case.
3. No prompt files at all → `FileNotFoundError` (the manager cannot start).

## When changes take effect

`ManagerClient` reads the champion prompt **once, at construction** — i.e.
at `python -m src.manager` startup. Editing `champion.txt` or a prompt file
does NOT affect a running manager process; restart the service to pick up
a newly promoted champion. (The prompt lab, Task 18, promotes champions by
rewriting `champion.txt`; the live service still needs a restart.)

Tested in `tests/test_manager_client.py` (`test_load_champion_prompt_*`,
`test_champion_beats_higher_version`, `test_empty_champion_falls_back_to_highest_version`,
`test_repo_champion_txt_names_an_existing_prompt`).
