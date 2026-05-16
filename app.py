"""
Tethys Greece — FastAPI Backend
================================
Serves the HTML UI and handles /ask requests by running the
Groq + MCP pipeline from server.py.

Run:  python app.py
Then open:  http://localhost:7860
"""

import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_KEY   = os.environ.get("GROQ_KEY")
if not GROQ_KEY:
    raise RuntimeError("GROQ_KEY environment variable not set")
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are Tethys — an environmental intelligence assistant specialising in
Greek climate, heat events, water resources, and marine conditions.
You have access to live ERA5 reanalysis data through seven scientific tools:

  • get_heat_events          — heatwave detection, heat-day count (hourly ERA5)
  • get_temperature_anomaly  — 2m T vs 1991–2020 baseline
  • get_water_stress         — evaporation / precipitation ratio (WSI)
  • get_drought_index        — SPI-like standardised precipitation index
  • get_wind_speed_anomaly   — 10m wind speed + Beaufort vs baseline
  • get_solar_radiation      — SSRD in J/m²/day (MJ/m²/day) vs baseline
  • get_sea_surface_temp     — SST anomaly for Aegean + Ionian seas

RULES:
1. Never invent or estimate numerical climate values. Always call a tool first.
2. Always state the data source (ERA5, Copernicus CDS) in your answer.
3. Report anomalies with sign and unit: e.g. "+2.1 °C above the 1991–2020 baseline".
4. If multiple tools are relevant, call them all before writing your answer.
5. Format answers for a scientific/policy audience:
   - Lead with the key finding in one sentence.
   - Follow with supporting numbers.
   - End with a brief implication for water management, public health, energy, or marine ecosystems.
6. Use markdown formatting: **bold** for key numbers, bullet points for lists.
7. Note if a repeated query was served from local cache (instant response).
""".strip()

app = FastAPI()


# ── REQUEST MODEL ─────────────────────────────────────────────────────────────

class Question(BaseModel):
    text: str


# ── GROQ + MCP PIPELINE (streaming) ──────────────────────────────────────────

async def run_Tethys(question: str):
    """
    Generator that yields server-sent events (SSE):
      - tool_call   : while a tool is being invoked
      - tool_result : once the tool returns data
      - answer      : the final LLM-grounded response
      - [DONE]      : stream terminator
    """
    client = Groq(api_key=GROQ_KEY)

    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=os.environ.copy(),   # inherit CDS_KEY, GROQ_KEY, ERA5_CACHE_DIR, etc.
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_result = await session.list_tools()
            groq_tools = [
                {
                    "type": "function",
                    "function": {
                        "name":        t.name,
                        "description": t.description,
                        "parameters":  t.inputSchema,
                    },
                }
                for t in mcp_result.tools
            ]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ]

            tool_call_count = 0

            while True:
                response = client.chat.completions.create(
                    model       = GROQ_MODEL,
                    messages    = messages,
                    tools       = groq_tools,
                    tool_choice = "auto",
                    max_tokens  = 2048,
                    temperature = 0.1,
                )

                msg = response.choices[0].message
                messages.append({
                    "role":       "assistant",
                    "content":    msg.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (msg.tool_calls or [])
                    ] or None,
                })

                if not msg.tool_calls:
                    break

                for tc in msg.tool_calls:
                    tool_call_count += 1
                    tool_name  = tc.function.name
                    tool_input = json.loads(tc.function.arguments)

                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'input': tool_input, 'n': tool_call_count})}\n\n"

                    result      = await session.call_tool(tool_name, tool_input)
                    result_txt  = result.content[0].text
                    result_data = json.loads(result_txt)

                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'data': result_data})}\n\n"

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_txt,
                    })

            final = msg.content or "No answer generated."
            yield f"data: {json.dumps({'type': 'answer', 'text': final, 'tool_calls': tool_call_count})}\n\n"
            yield "data: [DONE]\n\n"


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(q: Question):
    return StreamingResponse(
        run_Tethys(q.text),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("ui.html", "r") as f:
        return f.read()


# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
