import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const distRoot = resolve(__dirname, "dist");
const apiTarget = process.env.API_GATEWAY_URL || "http://127.0.0.1:8000";
const port = Number(process.env.PORT || 8501);
const host = process.env.HOST || "0.0.0.0";

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
};

function contentTypeFor(pathname) {
  const match = pathname.match(/\.[^.]+$/);
  return match ? contentTypes[match[0].toLowerCase()] || "application/octet-stream" : "application/octet-stream";
}

function sendStatic(response, pathname) {
  const safePath = normalize(pathname).replace(/^(\.\.[/\\])+/, "");
  let filePath = join(distRoot, safePath === "/" ? "index.html" : safePath);
  if (!filePath.startsWith(distRoot)) {
    response.writeHead(403);
    response.end("Forbidden");
    return;
  }

  try {
    const stat = statSync(filePath);
    if (stat.isDirectory()) {
      filePath = join(filePath, "index.html");
    }
  } catch {
    filePath = join(distRoot, "index.html");
  }

  response.writeHead(200, {
    "Content-Type": contentTypeFor(filePath),
    "Cache-Control": filePath.includes("/assets/") ? "public, max-age=31536000, immutable" : "no-cache",
    "X-Content-Type-Options": "nosniff",
  });
  createReadStream(filePath).pipe(response);
}

async function proxyToGateway(request, response) {
  const targetUrl = new URL(request.url || "/", apiTarget);
  const headers = new Headers();
  for (const [name, value] of Object.entries(request.headers)) {
    if (!value || ["host", "connection", "content-length"].includes(name.toLowerCase())) {
      continue;
    }
    headers.set(name, Array.isArray(value) ? value.join(", ") : value);
  }

  try {
    const upstream = await fetch(targetUrl, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method || "GET") ? undefined : request,
      duplex: "half",
    });
    response.writeHead(upstream.status, Object.fromEntries(upstream.headers.entries()));
    if (upstream.body) {
      for await (const chunk of upstream.body) {
        response.write(chunk);
      }
    }
    response.end();
  } catch (error) {
    response.writeHead(502, { "Content-Type": "application/json; charset=utf-8" });
    response.end(JSON.stringify({ detail: "API gateway proxy failed", error: String(error?.message || error) }));
  }
}

createServer((request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || "localhost"}`);
  if (url.pathname === "/health" || url.pathname.startsWith("/api/")) {
    void proxyToGateway(request, response);
    return;
  }
  sendStatic(response, url.pathname);
}).listen(port, host, () => {
  console.log(`Sitcheck Command Center listening on http://${host}:${port}`);
});
