/// <reference lib="dom" />
// Prism Deno Desktop — Agent browser workspace module
// Provides headless Chrome control via Puppeteer for agent browsing.
// Used by the Deno HTTP server for /api/browser/* endpoints.

import puppeteer, { type Browser, type Page } from "npm:puppeteer";

const CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const VIEWPORT = { width: 1280, height: 800 };

let browser: Browser | null = null;
let page: Page | null = null;

/** Lazily launch headless Chrome. */
async function ensureBrowser(): Promise<void> {
  if (browser?.connected) return;
  browser = await puppeteer.launch({
    headless: true,
    executablePath: CHROME_PATH,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--single-process",
    ],
  });
  page = await browser.newPage();
  await page.setViewport(VIEWPORT);
}

/** Ensure a page exists and return it. */
function getPage(): Page {
  if (!page) throw new Error("Browser not launched. Call navigate() first.");
  return page;
}

/** Sleep helper for async contexts where page.waitForTimeout may not exist. */
function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Public API ──────────────────────────────────────────────────────

export async function navigate(url: string): Promise<{ title: string; url: string }> {
  await ensureBrowser();
  await getPage().goto(url, { waitUntil: "networkidle2", timeout: 30_000 });
  return { title: await getPage().title(), url: getPage().url() };
}

export async function extract(): Promise<{
  title: string;
  url: string;
  headings: { level: number; text: string }[];
  links: { text: string; href: string }[];
  buttons: { text: string; selector: string }[];
  forms: {
    action: string;
    method: string;
    fields: { name: string; type: string; placeholder?: string }[];
  }[];
}> {
  await ensureBrowser();
  const p = getPage();
  const title = await p.title();
  const url = p.url();

  const headings = await p.evaluate(() =>
    Array.from(document.querySelectorAll("h1,h2,h3,h4,h5,h6")).map((h) => ({
      level: parseInt(h.tagName[1], 10),
      text: h.textContent?.trim() || "",
    }))
  );

  const links = await p.evaluate(() =>
    Array.from(document.querySelectorAll("a[href]")).map((a) => ({
      text: a.textContent?.trim() || "",
      href: (a as HTMLAnchorElement).href,
    }))
  );

  const buttons = await p.evaluate(() =>
    Array.from(document.querySelectorAll("button, a[role=button], [onclick]")).map((b, i) => {
      const el = b as HTMLElement;
      const text = el.textContent?.trim() || el.getAttribute("aria-label") || `button-${i}`;
      let selector: string;
      if (el.id) {
        selector = "#" + CSS.escape(el.id);
      } else if (el.getAttribute("data-testid")) {
        selector = `[data-testid="${el.getAttribute("data-testid")}"]`;
      } else if (el.tagName === "BUTTON" && el.textContent?.trim()) {
        selector = `button:has-text("${el.textContent.trim().slice(0, 30)}")`;
      } else {
        selector = `button:nth-of-type(${i + 1})`;
      }
      return { text, selector };
    })
  );

  const forms = await p.evaluate(() =>
    Array.from(document.querySelectorAll("form")).map((f) => ({
      action: (f as HTMLFormElement).action || "",
      method: (f as HTMLFormElement).method || "get",
      fields: Array.from(f.querySelectorAll("input, textarea, select")).map((el) => ({
        name: (el as HTMLInputElement).name || "",
        type: (el as HTMLInputElement).type || "text",
        placeholder: (el as HTMLInputElement).placeholder,
      })),
    }))
  );

  return { title, url, headings, links, buttons, forms };
}

export async function screenshot(): Promise<string> {
  await ensureBrowser();
  const p = getPage();
  const buf = await p.screenshot({ type: "png", fullPage: false });
  return btoa(String.fromCharCode(...new Uint8Array(buf)));
}

export async function click(selector: string): Promise<{ ok: boolean; newUrl?: string; error?: string }> {
  await ensureBrowser();
  const p = getPage();
  try {
    await p.waitForSelector(selector, { timeout: 5_000 });
    await p.click(selector, { delay: 50 });
    await sleep(300);
    return { ok: true, newUrl: p.url() };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

export async function typeText(selector: string, text: string): Promise<{ ok: boolean; error?: string }> {
  await ensureBrowser();
  const p = getPage();
  try {
    await p.waitForSelector(selector, { timeout: 5_000 });
    await p.click(selector);
    await p.type(selector, text, { delay: 20 });
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

export async function highlight(selector: string): Promise<{ ok: boolean; error?: string }> {
  await ensureBrowser();
  const p = getPage();
  try {
    await p.evaluate((sel: string) => {
      const el = document.querySelector(sel) as HTMLElement | null;
      if (!el) throw new Error(`Element not found: ${sel}`);
      el.style.outline = "3px solid #ff4500";
      el.style.outlineOffset = "2px";
      el.style.backgroundColor = "rgba(255, 69, 0, 0.1)";
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }, selector);
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

/** Close the browser and free resources. */
export async function close(): Promise<void> {
  if (browser && browser.connected) {
    await browser.close();
  }
  browser = null;
  page = null;
}
