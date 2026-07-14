# Prism MCP daemon

`prism-mcpd` provides the persistent local service behind Prism MCP clients. In development, invoking the binary without `--daemon` starts a stdio proxy and repairs a missing or unhealthy daemon. The proxy requires a deep health response with matching protocol and build identities before forwarding MCP traffic.

For production use on macOS, configure `PRISM_MCPD_POSTGRES_URL`, `PRISM_MCPD_VALKEY_URL`, and `PRISM_MCPD_DUCKDB_PATH`, then install the supervised user service with `prism-mcpd/scripts/install-launch-agent.sh`. The production profile is fail-closed and will not silently use SQLite. The installer builds a release binary, installs it at `~/.local/bin/prism-mcpd`, validates the generated property lists, and registers `com.prism.engine.mcpd` in the current GUI launch domain. `launchd` then owns startup and crash recovery; proxy repair remains a secondary safeguard.

SQLite is available only through the explicit `PRISM_MCPD_STORAGE=sqlite` local/test profile. It is not the production default.

The service stores runtime state under `~/.local/state/prism-mcpd`, artifacts beneath its `artifacts` directory, and rotating-service input logs under `~/Library/Logs/Prism`. Run `prism-mcpd/scripts/uninstall-launch-agent.sh` to remove supervision without deleting durable state or artifacts.
