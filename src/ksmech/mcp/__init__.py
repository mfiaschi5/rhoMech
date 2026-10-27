"""`ksmech-mcp`: the MCP server of the ksmech-serve API, for LLM clients (Claude Code, Claude
Desktop). It uses only `ksmech.client`, so an LLM acts through the same HTTP API, permissions
and audit as the web UI. The tools are in `ksmech.mcp.tools`, the server in
`ksmech.mcp.server`; `docs/mcp.md` lists them."""
