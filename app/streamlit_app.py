import json
import os
import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from openai import DefaultHttpxClient, OpenAI
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from calculator.CompositeIndex import build_composite_index
except ModuleNotFoundError:
    from Calculator.CompositeIndex import build_composite_index


DATA_PATH = Path("data/processed/urban_districts_scored.geojson")
MAP_CENTER = [22.32, 114.17]
MAP_ZOOM = 11
DEEPSEEK_SYSTEM_PROMPT = """You are an expert Urban Informatics assistant for the Hong Kong Urban Renewal Index Calculator.
You specialize in:

1. Urban Organism Theory - cities as living organisms:
   - Skeleton: building stock density, age, structural framework
   - Metabolism: POI accessibility (supermarkets, green space, services)
   - Circulatory System: road network density, transit coverage

2. Hong Kong Urban Renewal:
   - URA (Urban Renewal Authority) methodology and key projects
   - "Dual aging" challenge: aging population + aging buildings (50yr+)
   - High-density compact city characteristics unique to HK
   - Priority districts: Sham Shui Po, Kowloon City, Wan Chai, Yau Ma Tei

3. Interpreting the renewal priority scores:
   - Score 0.67-1.0 -> High priority: weak urban health, urgent renewal
   - Score 0.33-0.67 -> Medium: moderate intervention recommended
   - Score 0.0-0.33 -> Low: healthy urban fabric

4. Academic methodology: min-max normalisation, weighted composite index,
   AHP weight assignment, spatial analysis with GeoPandas.

Respond in the same language the user uses (Chinese or English).
Be concise (under 250 words) unless asked for more detail.
For Chinese queries, respond in fluent Mandarin Chinese.
"""


@st.cache_data
def load_scored_geojson(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def get_district_id_column(gdf: gpd.GeoDataFrame) -> str:
    candidates = [
        "DCCODE",
        "DC_CODE",
        "district",
        "District",
        "id",
        "ID",
        "name",
        "Name",
        "ENAME",
        "index",
    ]
    matching = [column for column in gdf.columns if column in candidates]
    return matching[0] if matching else gdf.columns[0]


def prepare_tooltip_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    tooltip_gdf = gdf.copy()
    for column in [
        "renewal_priority",
        "skeleton_score",
        "metabolism_score",
        "circulatory_score",
    ]:
        tooltip_gdf[f"{column}_display"] = tooltip_gdf[column].map(lambda value: f"{value:.2f}")
    return tooltip_gdf


def build_priority_map(gdf: gpd.GeoDataFrame, id_col: str) -> folium.Map:
    map_gdf = prepare_tooltip_columns(gdf)
    folium_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=map_gdf.to_json(),
        data=map_gdf[[id_col, "renewal_priority"]],
        columns=[id_col, "renewal_priority"],
        key_on=f"feature.properties.{id_col}",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        line_opacity=0.6,
        line_color="black",
        legend_name="Renewal Priority Score",
        highlight=True,
    ).add_to(folium_map)

    folium.GeoJson(
        map_gdf.to_json(),
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 0.7,
            "fillOpacity": 0.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                id_col,
                "renewal_priority_display",
                "priority_level",
                "skeleton_score_display",
                "metabolism_score_display",
                "circulatory_score_display",
            ],
            aliases=[
                f"{id_col}:",
                "Renewal priority:",
                "Priority level:",
                "Skeleton score:",
                "Metabolism score:",
                "Circulatory score:",
            ],
            localize=True,
            sticky=False,
            labels=True,
        ),
    ).add_to(folium_map)

    return folium_map


def get_deepseek_response(messages: list) -> str:
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        http_client=DefaultHttpxClient(trust_env=False),
    )
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": DEEPSEEK_SYSTEM_PROMPT}] + messages,
            max_tokens=800,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def render_chat_component(chat_history_json: str) -> None:
    float_chat_html = f"""
<script>
  const historyData = {chat_history_json};
  const parentDoc = window.parent.document;
  const rootId = 'deepseek-chat-root';

  const existingRoot = parentDoc.getElementById(rootId);
  if (existingRoot) existingRoot.remove();

  const root = parentDoc.createElement('div');
  root.id = rootId;
  root.innerHTML = `
    <style>
      #deepseek-chat-root #chat-bubble {{
        position: fixed;
        bottom: 32px;
        right: 32px;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: #1a56db;
        color: white;
        border: none;
        cursor: pointer;
        font-size: 26px;
        box-shadow: 0 4px 16px rgba(26,86,219,0.4);
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s;
      }}
      #deepseek-chat-root #chat-bubble:hover {{ transform: scale(1.08); }}
      #deepseek-chat-root #chat-window {{
        position: fixed;
        bottom: 100px;
        right: 32px;
        width: 370px;
        height: 520px;
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.18);
        display: none;
        flex-direction: column;
        z-index: 999998;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }}
      #deepseek-chat-root #chat-window.open {{ display: flex; }}
      #deepseek-chat-root #chat-header {{
        background: #1a56db;
        color: white;
        padding: 14px 18px;
        font-weight: 600;
        font-size: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
      }}
      #deepseek-chat-root #chat-header span {{
        font-size: 12px;
        opacity: 0.85;
        font-weight: 400;
      }}
      #deepseek-chat-root #close-btn {{
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
        line-height: 1;
      }}
      #deepseek-chat-root #chat-messages {{
        flex: 1;
        overflow-y: auto;
        padding: 14px;
        display: flex;
        flex-direction: column;
        gap: 10px;
        background: #f8f9fb;
      }}
      #deepseek-chat-root .msg-user {{
        align-self: flex-end;
        background: #1a56db;
        color: white;
        padding: 9px 13px;
        border-radius: 14px 14px 3px 14px;
        max-width: 82%;
        font-size: 13.5px;
        line-height: 1.5;
        word-break: break-word;
      }}
      #deepseek-chat-root .msg-bot {{
        align-self: flex-start;
        background: #ffffff;
        color: #1a1a2e;
        padding: 9px 13px;
        border-radius: 14px 14px 14px 3px;
        max-width: 88%;
        font-size: 13.5px;
        line-height: 1.6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        word-break: break-word;
        white-space: pre-wrap;
      }}
      #deepseek-chat-root .msg-thinking {{
        align-self: flex-start;
        color: #888;
        font-size: 12.5px;
        font-style: italic;
        padding: 4px 8px;
      }}
      #deepseek-chat-root #suggestions {{
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        padding: 10px 14px 0;
        background: #f8f9fb;
        flex-shrink: 0;
      }}
      #deepseek-chat-root .suggestion-btn {{
        background: #e8f0fe;
        color: #1a56db;
        border: none;
        border-radius: 20px;
        padding: 5px 11px;
        font-size: 12px;
        cursor: pointer;
        transition: background 0.15s;
        white-space: nowrap;
      }}
      #deepseek-chat-root .suggestion-btn:hover {{ background: #c7d9fc; }}
      #deepseek-chat-root #chat-input-area {{
        display: flex;
        padding: 10px 12px;
        gap: 8px;
        border-top: 1px solid #e5e7eb;
        background: white;
        flex-shrink: 0;
      }}
      #deepseek-chat-root #chat-input {{
        flex: 1;
        border: 1px solid #d1d5db;
        border-radius: 22px;
        padding: 8px 14px;
        font-size: 13.5px;
        outline: none;
        resize: none;
        font-family: inherit;
        line-height: 1.4;
      }}
      #deepseek-chat-root #chat-input:focus {{ border-color: #1a56db; }}
      #deepseek-chat-root #send-btn {{
        background: #1a56db;
        color: white;
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        font-size: 16px;
        cursor: pointer;
        flex-shrink: 0;
        align-self: flex-end;
        transition: background 0.15s;
      }}
      #deepseek-chat-root #send-btn:hover {{ background: #1344b0; }}
    </style>
    <button id="chat-bubble" title="Urban Renewal AI">&#x1F4AC;</button>
    <div id="chat-window">
      <div id="chat-header">
        <div>
          Urban Renewal AI
          <br><span>Powered by DeepSeek</span>
        </div>
        <button id="close-btn">&#x2715;</button>
      </div>
      <div id="suggestions">
        <button class="suggestion-btn" data-text="Explain Urban Organism Theory">Urban Organism</button>
        <button class="suggestion-btn" data-text="Why is Sham Shui Po high priority?">Sham Shui Po</button>
        <button class="suggestion-btn" data-text="How should I interpret metabolism scores?">Metabolism</button>
        <button class="suggestion-btn" data-text="What is URA methodology?">URA Methodology</button>
      </div>
      <div id="chat-messages"></div>
      <div id="chat-input-area">
        <textarea id="chat-input" rows="1" placeholder="Ask about scores, districts, or urban renewal..."></textarea>
        <button id="send-btn">&#x27A4;</button>
      </div>
    </div>
  `;
  parentDoc.body.appendChild(root);

  const messagesDiv = root.querySelector('#chat-messages');
  const chatWindow = root.querySelector('#chat-window');
  const chatBubble = root.querySelector('#chat-bubble');
  const closeBtn = root.querySelector('#close-btn');
  const chatInput = root.querySelector('#chat-input');
  const sendBtn = root.querySelector('#send-btn');

  function appendMessage(role, text) {{
    const div = parentDoc.createElement('div');
    div.className = role === 'user' ? 'msg-user' : 'msg-bot';
    div.textContent = text;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }}

  function showThinking() {{
    const div = parentDoc.createElement('div');
    div.className = 'msg-thinking';
    div.id = 'thinking-indicator';
    div.textContent = 'AI is thinking...';
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
  }}

  function removeThinking() {{
    const el = messagesDiv.querySelector('#thinking-indicator');
    if (el) el.remove();
  }}

  function toggleChat() {{
    chatWindow.classList.toggle('open');
  }}

  async function sendMessage(textOverride = null) {{
    const text = (textOverride ?? chatInput.value).trim();
    if (!text) return;
    chatInput.value = '';
    appendMessage('user', text);
    showThinking();
    try {{
      const response = await fetch('http://127.0.0.1:8502/api/chat', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ message: text }})
      }});
      removeThinking();
      if (response.ok) {{
        const data = await response.json();
        appendMessage('bot', data.reply);
      }} else {{
        appendMessage('bot', 'Sorry, an error occurred. Please try again.');
      }}
    }} catch (err) {{
      removeThinking();
      appendMessage('bot', 'Connection error: ' + err.message);
    }}
  }}

  if (historyData.length === 0) {{
    appendMessage('bot', 'Hello! I can help interpret the renewal map, explain Urban Organism Theory, and answer Hong Kong urban renewal questions.');
  }} else {{
    historyData.forEach(msg => {{
      appendMessage(msg.role === 'user' ? 'user' : 'bot', msg.content);
    }});
  }}

  chatBubble.addEventListener('click', toggleChat);
  closeBtn.addEventListener('click', toggleChat);
  sendBtn.addEventListener('click', () => sendMessage());
  chatInput.addEventListener('keydown', (e) => {{
    if (e.key === 'Enter' && !e.shiftKey) {{
      e.preventDefault();
      sendMessage();
    }}
  }});
  root.querySelectorAll('.suggestion-btn').forEach(btn => {{
    btn.addEventListener('click', () => sendMessage(btn.dataset.text));
  }});
</script>
"""
    components.html(float_chat_html, height=1, scrolling=False)


def main() -> None:
    st.set_page_config(layout="wide", page_title="HK Urban Renewal Index")

    st.sidebar.title("Urban Organism Weights")
    st.sidebar.subheader("Adjust dimension weights (must sum to 1.0)")

    w_skeleton = st.sidebar.slider("Skeleton (Building)", 0.0, 1.0, 0.4, 0.05)
    w_metabolism = st.sidebar.slider("Metabolism (Services)", 0.0, 1.0, 0.3, 0.05)
    w_circulatory = st.sidebar.slider("Circulatory (Roads)", 0.0, 1.0, 0.3, 0.05)

    total = w_skeleton + w_metabolism + w_circulatory
    if abs(total - 1.0) > 0.01:
        st.sidebar.warning(f"Weights sum to {total:.2f}, should be 1.0")
    else:
        st.sidebar.success("Weights valid")

    st.sidebar.divider()
    st.sidebar.caption("Urban Organism Theory")
    st.sidebar.caption("Skeleton = building density & floor area")
    st.sidebar.caption("Metabolism = access to supermarkets & green space")
    st.sidebar.caption("Circulatory = road network & transit access")

    st.title("HK Urban Renewal Index Calculator")
    st.markdown(
        "An interactive tool based on the **Urban Organism** theory. "
        "Adjust dimension weights in the sidebar to see how renewal "
        "priorities shift across Hong Kong's urban districts."
    )
    st.divider()

    gdf = load_scored_geojson(str(DATA_PATH))
    scored_gdf = build_composite_index(gdf, w_skeleton, w_metabolism, w_circulatory)
    id_col = get_district_id_column(scored_gdf)

    if "pending_chat_msg" in st.session_state and st.session_state.pending_chat_msg:
        user_msg = st.session_state.pending_chat_msg
        st.session_state.pending_chat_msg = None
        if "deepseek_history" not in st.session_state:
            st.session_state.deepseek_history = []
        st.session_state.deepseek_history.append({"role": "user", "content": user_msg})
        reply = get_deepseek_response(st.session_state.deepseek_history)
        st.session_state.deepseek_history.append({"role": "assistant", "content": reply})

    chat_history_json = json.dumps(
        st.session_state.get("deepseek_history", []),
        ensure_ascii=False,
    )

    left_col, right_col = st.columns([6, 4])

    with left_col:
        st.subheader("Interactive Priority Map")
        folium_map = build_priority_map(scored_gdf, id_col)
        st_folium(folium_map, width=700, height=500)

    with right_col:
        st.subheader("Urban Renewal Summary")
        high_count = int((scored_gdf["priority_level"] == "High").sum())
        mean_priority = float(scored_gdf["renewal_priority"].mean())
        top_district = scored_gdf.sort_values("renewal_priority", ascending=False).iloc[0]

        metric_cols = st.columns(3)
        metric_cols[0].metric("High priority districts", high_count)
        metric_cols[1].metric("Mean priority score", f"{mean_priority:.3f}")
        metric_cols[2].metric("Top district ID", str(top_district[id_col]))

        st.markdown("**Average Dimension Scores**")
        avg_scores = (
            scored_gdf[["skeleton_score", "metabolism_score", "circulatory_score"]]
            .mean()
            .rename(
                {
                    "skeleton_score": "Skeleton",
                    "metabolism_score": "Metabolism",
                    "circulatory_score": "Circulatory",
                }
            )
            .sort_values()
        )
        st.bar_chart(avg_scores, horizontal=True)

        st.markdown("**Top 10 Districts**")
        top_ten = scored_gdf.sort_values("renewal_priority", ascending=False).head(10).copy()
        top10_df = top_ten[
            [
                id_col,
                "renewal_priority",
                "priority_level",
                "skeleton_score",
                "metabolism_score",
                "circulatory_score",
            ]
        ].copy()
        st.dataframe(
            top10_df.style.format(
                {
                    "renewal_priority": "{:.3f}",
                    "skeleton_score": "{:.3f}",
                    "metabolism_score": "{:.3f}",
                    "circulatory_score": "{:.3f}",
                }
            ),
            use_container_width=True,
        )
        csv = top10_df.to_csv(index=False)
        st.download_button(
            label="Download top 10 as CSV",
            data=csv,
            file_name="hk_renewal_priority_top10.csv",
            mime="text/csv",
        )

    st.caption(
        "Data sources: OpenStreetMap contributors · "
        "HK Government Open Data · "
        "MSc Urban Informatics, PolyU 2025"
    )

    render_chat_component(chat_history_json)


if __name__ == "__main__":
    main()
