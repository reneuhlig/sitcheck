import { buildToolRuntime } from "../../apps/mcp-sitcheck/tools.mjs";

const runtime = buildToolRuntime({
  apiBaseUrl: process.env.API_BASE_URL || "http://localhost:8000",
  ollamaEnabled: false,
});

const zoneId = process.env.MCP_TEST_ZONE_ID || "default-zone";
const now = new Date();
const from = new Date(now.getTime() - 60 * 60 * 1000).toISOString();
const to = now.toISOString();

async function run() {
  const r1 = await runtime.callTool("get_history", {
    zone_id: zoneId,
    from,
    to,
    granularity: "1m",
  });
  const r2 = await runtime.callTool("get_forecast", { zone_id: zoneId, horizon: 60 });
  const r3 = await runtime.callTool("recommend_actions", { zone_id: zoneId, horizon: 60 });

  console.log("mcp smoke results");
  console.log(JSON.stringify({
    history_points: (r1.points || []).length,
    forecast_points: (r2.points || []).length,
    actions: (r3.actions || []).length,
  }, null, 2));
}

run().catch((err) => {
  console.error("mcp smoke test failed", err);
  process.exit(1);
});
