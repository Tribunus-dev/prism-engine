#!/bin/zsh
set -euo pipefail

agent_path="${HOME}/Library/LaunchAgents/com.prism.engine.mcpd.plist"
rotate_agent_path="${HOME}/Library/LaunchAgents/com.prism.engine.mcpd-logrotate.plist"
domain="gui/$(id -u)"

launchctl bootout "${domain}" "${agent_path}" 2>/dev/null || true
launchctl bootout "${domain}" "${rotate_agent_path}" 2>/dev/null || true
rm -f "${agent_path}"
rm -f "${rotate_agent_path}"
rm -f "${HOME}/.local/bin/prism-mcpd-rotate-logs"
rm -f "${PRISM_MCPD_STATE_DIR:-${HOME}/.local/state/prism-mcpd}/supervised"
