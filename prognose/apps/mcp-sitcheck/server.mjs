import http from "node:http";
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { buildToolRuntime } from "./tools.mjs";

const runtime = buildToolRuntime();

const mcpServer = new Server(
  { name: "mcp-sitcheck", version: "0.1.0" },
  { capabilities: { tools: {} } },
);

mcpServer.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: Object.entries(runtime.tools).map(([name, spec]) => ({
    name,
    description: spec.description,
    inputSchema: spec.inputSchema,
  })),
}));

mcpServer.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const result = await runtime.callTool(name, args || {});
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(result, null, 2),
      },
    ],
  };
});

const healthPort = Number(process.env.MCP_HEALTH_PORT || 8081);
const healthServer = http.createServer((req, res) => {
  if (req.url === "/health") {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ status: "ok", service: "mcp-sitcheck" }));
    return;
  }
  res.writeHead(404);
  res.end();
});

healthServer.listen(healthPort, () => {
  console.log(`mcp-sitcheck health server listening on :${healthPort}`);
});

const transport = new StdioServerTransport();
await mcpServer.connect(transport);
