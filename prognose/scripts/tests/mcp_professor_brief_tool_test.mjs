import { buildToolRuntime } from "../../apps/mcp-sitcheck/tools.mjs";

const runtime = buildToolRuntime({ apiBaseUrl: "http://127.0.0.1:65535" });

if (!runtime?.tools?.generate_professor_brief) {
  throw new Error("generate_professor_brief tool missing");
}

const schema = runtime.tools.generate_professor_brief.inputSchema || {};
const required = Array.isArray(schema.required) ? schema.required : [];
if (!required.includes("zone_id") || !required.includes("horizon")) {
  throw new Error("generate_professor_brief input schema invalid");
}

console.log("mcp professor brief tool test passed");
