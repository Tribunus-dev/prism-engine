#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h:h}"
binary_dir="${HOME}/.local/bin"
binary_path="${binary_dir}/prism-mcpd"
state_dir="${PRISM_MCPD_STATE_DIR:-${HOME}/.local/state/prism-mcpd}"
artifact_dir="${PRISM_MCPD_ARTIFACT_DIR:-${state_dir}/artifacts}"
postgres_url="${PRISM_MCPD_POSTGRES_URL:?PRISM_MCPD_POSTGRES_URL is required for the production trifecta profile}"
valkey_url="${PRISM_MCPD_VALKEY_URL:?PRISM_MCPD_VALKEY_URL is required for the production trifecta profile}"
duckdb_path="${PRISM_MCPD_DUCKDB_PATH:?PRISM_MCPD_DUCKDB_PATH is required for the production trifecta profile}"
log_dir="${HOME}/Library/Logs/Prism"
agent_dir="${HOME}/Library/LaunchAgents"
agent_path="${agent_dir}/com.prism.engine.mcpd.plist"
rotate_agent_path="${agent_dir}/com.prism.engine.mcpd-logrotate.plist"
template="${repo_root}/prism-mcpd/launchd/com.prism.engine.mcpd.plist.in"
rotate_template="${repo_root}/prism-mcpd/launchd/com.prism.engine.mcpd-logrotate.plist.in"
rotate_script="${binary_dir}/prism-mcpd-rotate-logs"
domain="gui/$(id -u)"

cargo build --release -p prism-mcpd --manifest-path "${repo_root}/Cargo.toml"
mkdir -p "${binary_dir}" "${state_dir}" "${artifact_dir}" "${log_dir}" "${agent_dir}"
install -m 0755 "${repo_root}/target/release/prism-mcpd" "${binary_path}"
install -m 0755 "${repo_root}/prism-mcpd/scripts/rotate-logs.sh" "${rotate_script}"

sed \
  -e "s|__BINARY__|${binary_path}|g" \
  -e "s|__HOME__|${HOME}|g" \
  -e "s|__STATE_DIR__|${state_dir}|g" \
  -e "s|__ARTIFACT_DIR__|${artifact_dir}|g" \
  -e "s|__POSTGRES_URL__|${postgres_url}|g" \
  -e "s|__VALKEY_URL__|${valkey_url}|g" \
  -e "s|__DUCKDB_PATH__|${duckdb_path}|g" \
  -e "s|__LOG_DIR__|${log_dir}|g" \
  "${template}" > "${agent_path}.tmp"
plutil -lint "${agent_path}.tmp"
mv "${agent_path}.tmp" "${agent_path}"

sed \
  -e "s|__SCRIPT__|${rotate_script}|g" \
  -e "s|__LOG_DIR__|${log_dir}|g" \
  "${rotate_template}" > "${rotate_agent_path}.tmp"
plutil -lint "${rotate_agent_path}.tmp"
mv "${rotate_agent_path}.tmp" "${rotate_agent_path}"

launchctl bootout "${domain}" "${agent_path}" 2>/dev/null || true
launchctl bootout "${domain}" "${rotate_agent_path}" 2>/dev/null || true
if [[ -f "${state_dir}/mcpd.pid" ]]; then
  daemon_pid="$(<"${state_dir}/mcpd.pid")"
  if [[ "${daemon_pid}" == <-> ]] && ps -p "${daemon_pid}" -o command= 2>/dev/null | grep -q prism-mcpd; then
    kill -TERM "${daemon_pid}" 2>/dev/null || true
    for _ in {1..40}; do
      kill -0 "${daemon_pid}" 2>/dev/null || break
      sleep 0.05
    done
  fi
fi
touch "${state_dir}/supervised"
launchctl bootstrap "${domain}" "${agent_path}"
launchctl bootstrap "${domain}" "${rotate_agent_path}"
launchctl enable "${domain}/com.prism.engine.mcpd"
launchctl enable "${domain}/com.prism.engine.mcpd-logrotate"
launchctl kickstart -k "${domain}/com.prism.engine.mcpd"
launchctl kickstart -k "${domain}/com.prism.engine.mcpd-logrotate"
launchctl print "${domain}/com.prism.engine.mcpd"
