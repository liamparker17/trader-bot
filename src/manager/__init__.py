"""Claude-manager core (Task 12+): policy, briefing, manager_log audit trail.

No Anthropic SDK / network calls live here — that's Task 13's job. This
package is pure functions + sqlite (via TradeJournal) so it can be unit
tested without any live model or broker connection.
"""
