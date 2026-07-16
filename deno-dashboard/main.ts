// Prism Deno Desktop — MCP client + web UI server + agent browser workspace
// Serves the dashboard UI, proxies MCP commands to prism-mcpd, and
// provides headless Chrome control for agent browsing.

import * as browser from "./browser.ts";

const MCP_WS_URL = Deno.env.get("PRISM_MCP_WS") || "ws://127.0.0.1:8080/api/ws";
const HTTP_PORT = parseInt(Deno.env.get("PRISM_DENO_PORT") || "8081");

// ── MCP WebSocket client ─────────────────────────────────────────

class McpClient {
  private ws!: WebSocket;
  private pending = new Map<string, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
  private idCounter = 0;

  async connect(): Promise<void> {
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    this.ws = new WebSocket(MCP_WS_URL);
    this.ws.onopen = () => resolve();
    this.ws.onerror = (e) => reject(e as unknown as Error);
    this.ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        const pending = this.pending.get(msg.id);
        if (pending) {
          pending.resolve(msg);
          this.pending.delete(msg.id);
        }
      } catch {
        // Ignore malformed messages
      }
    };
    return promise;
  }

  async call(tool: string, args: Record<string, unknown>): Promise<unknown> {
    const id = String(++this.idCounter);
    const { promise, resolve, reject } = Promise.withResolvers<unknown>();
    this.pending.set(id, { resolve, reject });
    this.ws.send(JSON.stringify({ id, tool, args }));
    // Timeout after 30s
    setTimeout(() => {
      if (this.pending.has(id)) {
        this.pending.delete(id);
        reject(new Error(`MCP call ${tool} timed out`));
      }
    }, 30_000);
    return promise;
  }

  close() {
    this.ws.close();
  }
}

// ── Browser workspace handlers ──────────────────────────────────────

const browserHandlers: Record<string, (body: Record<string, unknown>) => Promise<unknown>> = {
  async navigate(body) {
    const url = body.url as string;
    if (!url) throw new Error("Missing url");
    return await browser.navigate(url);
  },
  async extract(_body) {
    return await browser.extract();
  },
  async screenshot(_body) {
    const b64 = await browser.screenshot();
    return { data: b64, mime: "image/png" };
  },
  async click(body) {
    const selector = body.selector as string;
    if (!selector) throw new Error("Missing selector");
    return await browser.click(selector);
  },
  async type(body) {
    const selector = body.selector as string;
    const text = body.text as string;
    if (!selector || text === undefined) throw new Error("Missing selector or text");
    return await browser.typeText(selector, text);
  },
  async highlight(body) {
    const selector = body.selector as string;
    if (!selector) throw new Error("Missing selector");
    return await browser.highlight(selector);
  },
  async close(_body) {
    await browser.close();
    return { ok: true };
  },
};

async function handleBrowserApi(path: string, body: Record<string, unknown>): Promise<unknown> {
  // path is e.g. "browser/navigate" or "browser/screenshot"
  const parts = path.split("/");
  if (parts.length < 2) throw new Error("Invalid browser API path");
  const action = parts[1]; // e.g. "navigate", "screenshot"
  const handler = browserHandlers[action];
  if (!handler) throw new Error(`Unknown browser action: ${action}`);
  return await handler(body);
}

// ── HTTP server ────────────────────────────────────────────────────

async function handler(req: Request): Promise<Response> {
  const url = new URL(req.url);

  // Browser workspace API — handled locally
  if (url.pathname.startsWith("/api/browser/")) {
    try {
      const body = req.body ? await req.json() : {};
      const result = await handleBrowserApi(url.pathname.slice("/api/".length), body as Record<string, unknown>);
      return new Response(JSON.stringify(result), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (e) {
      return new Response(JSON.stringify({ error: (e as Error).message }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  // Proxy other /api/* calls to prism-mcpd
  if (url.pathname.startsWith("/api/")) {
    const method = url.pathname.slice("/api/".length); // e.g. "hw/probe"
    try {
      const client = new McpClient();
      await client.connect();
      const body = req.body ? await req.json() : {};
      const result = await client.call(method, body);
      client.close();
      return new Response(JSON.stringify(result), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (e) {
      return new Response(JSON.stringify({ error: (e as Error).message }), {
        status: 500,
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  // Serve the dashboard HTML
  if (url.pathname === "/" || url.pathname === "/index.html") {
    const html = await Deno.readTextFile(new URL("./dashboard.html", import.meta.url));
    return new Response(html, {
      headers: { "Content-Type": "text/html" },
    });
  }

  return new Response("Not found", { status: 404 });
}

console.log(`Prism Deno Desktop listening on http://127.0.0.1:${HTTP_PORT}`);
Deno.serve({ port: HTTP_PORT }, handler);
