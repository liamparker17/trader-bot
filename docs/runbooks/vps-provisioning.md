# VPS Provisioning Runbook — Hetzner Windows Server

**Scope:** stand up TraderBot on an always-on Hetzner Windows VPS, running both the trading loop
(`traderbot`) and the Claude portfolio-manager (`manager`) as separate, independently-restarting
NSSM services, controllable from the developer's local machine via the `tb` CLI over SSH.

**Source:** `docs/superpowers/specs/2026-05-02-autonomous-vps-and-cli-control-design.md` (architecture,
NSSM watchdog requirement) and `docs/superpowers/specs/2026-07-07-claude-manager-live-r1000-design.md`
(second service for the manager). This runbook is the executable version of both.

> The `src/manager/client.py`, `scheduler.py`, `runner.py` files and the `manager:` settings.yaml block
> are Task 13 (in progress in parallel with this doc). Everything below that references the manager
> service's exact command line or settings keys is marked **verify after Task 13 lands** — the shape
> (`python -m src.manager`, a second NSSM service, same log/restart conventions as `traderbot`) will not
> change; only field-level details might.

## 1. Provision the VPS

1. Create a Hetzner Cloud server: **CX22 or CPX21, Windows Server (EU region)**. 2 vCPU / 4-8GB RAM is
   sufficient for MT5 + Python; EU region minimizes latency to Exness servers.
2. Note the public IP and the Administrator password from the Hetzner console (or set your own on
   creation).
3. RDP in (`mstsc /v:<ip>`) to do the interactive setup steps below (MT5 install, auto-login). Once SSH
   is configured (step 4), day-to-day control no longer needs RDP.
4. **Enable Windows OpenSSH Server** (needed for the local `tb` alias):
   ```powershell
   # On the VPS, as Administrator
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service -Name sshd -StartupType Automatic
   # Allow SSH through Windows Firewall (usually created automatically by the capability install;
   # verify it exists)
   Get-NetFirewallRule -Name *ssh*
   ```
5. **Set up key-based auth** (avoid password auth for the service account):
   - On your local machine: `ssh-keygen -t ed25519 -f ~/.ssh/traderbot_vps` (if you don't already have a
     key you want to reuse).
   - Copy the public key into `C:\Users\<vps-user>\.ssh\authorized_keys` on the VPS (for an
     Administrator account, Windows OpenSSH instead reads
     `C:\ProgramData\ssh\administrators_authorized_keys` — set matching ACLs per Microsoft's OpenSSH
     docs, or use a non-admin service account with the standard per-user `authorized_keys` path).
   - Test: `ssh -i ~/.ssh/traderbot_vps <user>@<vps-ip> "echo ok"`.

## 2. Install Python + clone the repo

1. Install Python 3.11 (matches the project's tested version) and Git for Windows on the VPS.
2. Clone the repo to `C:\traderbot` (or your preferred path — keep it consistent, the NSSM configs
   below assume this).
3. Create a venv and install dependencies:
   ```powershell
   cd C:\traderbot
   python -m venv venv
   venv\Scripts\pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`,
   `MT5_TERMINAL_PATH` (if not auto-detected), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and
   `ANTHROPIC_API_KEY` (required once the manager service is enabled — Task 13; add the key alongside
   the existing MT5/Telegram vars in the same `.env`, there is no separate secrets file).
   - **Demo first**: use demo MT5 credentials here. Do not put live credentials in until the live
     cutover checklist (`docs/runbooks/live-cutover-r1000.md`) is complete.

## 3. Install MT5 + auto-login

1. Install the MetaTrader 5 terminal (from Exness's MT5 download link, or the generic MetaQuotes
   installer pointed at the Exness server).
2. Log in once interactively with the demo account credentials from `.env`, and in
   **Tools → Options → Server**, ensure "Save account information" / auto-login is enabled so the
   terminal reconnects without a prompt after a VPS reboot.
3. Confirm `MT5_SERVER` in `.env` matches the exact server name shown in the terminal (e.g.
   `Exness-MT5Trial7`) — `src/data/mt5_client.py` connects using this value plus `MT5_LOGIN`/`MT5_PASSWORD`.
4. Leave the terminal running (or closed — `MetaTrader5` Python package can launch it headless via
   `MT5_TERMINAL_PATH` if the path is set and no instance is running); either way, `mt5_client.py`
   auto-detects the broker's symbol suffix (e.g. `EURUSD` vs `EURUSDm`) on connect and again on every
   reconnect.

## 4. Install both services with NSSM

Download [NSSM](https://nssm.cc/) and put `nssm.exe` somewhere on `PATH` (e.g. `C:\tools\nssm.exe`).

### Service 1 — `traderbot` (the trading loop)

```powershell
nssm install traderbot "C:\traderbot\venv\Scripts\python.exe" "-m src.main"
nssm set traderbot AppDirectory "C:\traderbot"
nssm set traderbot AppStdout "C:\traderbot\logs\traderbot_service.log"
nssm set traderbot AppStderr "C:\traderbot\logs\traderbot_service_err.log"
nssm set traderbot AppRotateFiles 1
nssm set traderbot AppRotateOnline 1
nssm set traderbot AppRotateBytes 10485760
nssm set traderbot AppExit Default Restart
nssm set traderbot AppRestartDelay 10000
nssm set traderbot Start SERVICE_AUTO_START
nssm start traderbot
```

### Service 2 — `manager` (the Claude portfolio manager)

> **Verify after Task 13 lands** — confirm the module path is exactly `python -m src.manager` and that
> no additional CLI flags are required; the NSSM config pattern itself (separate service, same
> restart/rotation settings) is fixed by this runbook and the spec.

```powershell
nssm install manager "C:\traderbot\venv\Scripts\python.exe" "-m src.manager"
nssm set manager AppDirectory "C:\traderbot"
nssm set manager AppStdout "C:\traderbot\logs\manager_service.log"
nssm set manager AppStderr "C:\traderbot\logs\manager_service_err.log"
nssm set manager AppRotateFiles 1
nssm set manager AppRotateOnline 1
nssm set manager AppRotateBytes 10485760
nssm set manager AppExit Default Restart
nssm set manager AppRestartDelay 10000
nssm set manager Start SERVICE_AUTO_START
nssm start manager
```

**Required watchdog settings for both services** (per the 2026-05-02 spec, fix J):
`AppRestartDelay=10000` (10s backoff before NSSM restarts a crashed process — avoids a hot-crash-loop)
and `AppRotateFiles=1` (log rotation so the service log files don't grow unbounded over a month+ of
unattended running). Both are set explicitly above; don't rely on NSSM defaults.

The two services are independent: if `manager` crashes or is stopped, `traderbot` keeps trading on its
last effective config (the manager has no privileged control-plane access — see CLAUDE.md and the
2026-07-07 spec's "least privilege by construction" principle). If `traderbot` restarts, it reconciles
open positions from MT5 before resuming.

## 5. Local `tb` alias

On your local development machine (not the VPS), add an alias/function that tunnels `tb` commands over
SSH to the VPS venv's `cli.tb`:

**PowerShell profile** (`$PROFILE`):
```powershell
function tb {
    ssh -i ~/.ssh/traderbot_vps <user>@<vps-ip> "cd C:\traderbot && venv\Scripts\python -m cli.tb $args"
}
```

**bash/zsh** (`~/.bashrc` or `~/.zshrc`):
```bash
tb() {
    ssh -i ~/.ssh/traderbot_vps <user>@<vps-ip> "cd C:/traderbot && venv/Scripts/python -m cli.tb $*"
}
```

Verify: `tb status` should return a JSON status snapshot (or a journal-derived degraded status with
`"bot_running": false` if the trading service isn't up yet).

## 6. Log locations

| Component | Path |
|---|---|
| `traderbot` service stdout/stderr | `C:\traderbot\logs\traderbot_service.log` / `_err.log` (NSSM-rotated) |
| `manager` service stdout/stderr | `C:\traderbot\logs\manager_service.log` / `_err.log` (NSSM-rotated) — verify after Task 13 lands |
| Application-level log (`src/main.py:main()`, `logging.handlers.RotatingFileHandler`, 10MB/5 backups) | `data/logs/traderbot.log` — `cli.tb logs [--tail N] [--level L]` tails this file |
| Trade journal (SQLite, `monitoring.trade_journal_db` in settings.yaml) | `data/trade_logs/trades.db` — holds `trades`, `daily_summary`, `events`, `control_log`, `manager_log` tables |
| Ratchet-floor state | `data/account_state.json` (`{"high_water_mark": ..., "updated_utc": ...}`) |
| Instance lock | `data/traderbot.lock` |
| Control queue | `control/inbox/*.cmd.json`, `control/outbox/*.result.json`, `control/inbox/deadletter/` |

## 7. Verifying both services after a reboot

1. Reboot the VPS (or simulate via `Restart-Service` for each service individually first).
2. Confirm both services are running:
   ```powershell
   Get-Service traderbot, manager
   ```
   Both should show `Status: Running` and `StartType: Automatic`.
3. Confirm the trading loop reconnected to MT5 and reconciled state — check the app log for a
   reconciliation line, and `tb status` from the local alias should return `"bot_running": true` with a
   sane balance.
4. Confirm MT5 auto-login survived the reboot (the terminal should already be logged in — if not, fix
   the "save account info" setting from step 3.2 and note it as a Telegram-alert-worthy VPS quirk, per
   the spec's risk note "MT5 quirks on VPS (auto-login failure after Windows update) → 'no candles in N
   min' alert; RDP in").
5. Confirm the manager service produced (or is due to produce) its next `manager_log` row —
   `tb manager --days 1` — verify after Task 13 lands.
6. Tail both service log files (`Get-Content <path> -Tail 50`) and confirm no repeated crash-restart
   loop (a healthy log shows one startup sequence, not the same startup lines repeating every ~10s).
