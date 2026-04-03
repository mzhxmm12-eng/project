import json
import os
import sys
from pathlib import Path

import folium
import geopandas as gpd
import streamlit as st
import streamlit.components.v1 as components
from openai import DefaultHttpxClient, OpenAI
from streamlit_folium import st_folium

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from calculator.CompositeIndex import build_composite_index
except ModuleNotFoundError:
    from Calculator.CompositeIndex import build_composite_index


DATA_PATH = Path("data/processed/urban_districts_with_indicators.geojson")
MAP_CENTER = [22.32, 114.17]
MAP_ZOOM = 11
TPU_NAMES = {
    "100": "Kennedy Town",
    "110": "Sai Ying Pun",
    "111": "Sai Ying Pun & Sheung Wan",
    "112": "Central",
    "113": "Admiralty",
    "114": "Wan Chai",
    "115": "Causeway Bay",
    "116": "North Point",
    "117": "Quarry Bay",
    "118": "Sai Wan Ho",
    "119": "Shau Kei Wan",
    "120": "Chai Wan",
    "121": "Stanley",
    "122": "Aberdeen",
    "123": "Ap Lei Chau",
    "124": "Wong Chuk Hang",
    "125": "Pok Fu Lam",
    "126": "The Peak",
    "130": "Kennedy Town & Mount Davis",
    "131": "Sai Ying Pun North",
    "132": "Pok Fu Lam South",
    "140": "Tin Hau",
    "150": "Tai Hang",
    "158": "Jardine's Lookout",
    "160": "Happy Valley",
    "170": "Mid-Levels",
    "180": "Repulse Bay",
    "190": "Deep Water Bay",
    "194": "Tai Tam",
    "210": "Tsim Sha Tsui",
    "211": "Tsim Sha Tsui East",
    "212": "Yau Ma Tei",
    "213": "Mong Kok",
    "214": "Sham Shui Po",
    "215": "Cheung Sha Wan",
    "216": "Lai Chi Kok",
    "220": "Ho Man Tin",
    "221": "Kowloon Tong",
    "222": "Kowloon City",
    "223": "Hung Hom",
    "224": "To Kwa Wan",
    "225": "Ma Tau Wai",
    "230": "Wong Tai Sin",
    "231": "Diamond Hill",
    "232": "Tsz Wan Shan",
    "233": "San Po Kong",
    "240": "Kwun Tong",
    "241": "Ngau Tau Kok",
    "242": "Lam Tin",
    "243": "Yau Tong",
    "244": "Tseung Kwan O North",
    "250": "Kowloon Bay",
    "255": "Kai Tak",
    "260": "Jordan",
    "270": "Prince Edward",
    "280": "Shek Kip Mei",
    "289": "Beacon Hill",
    "310": "Kwai Chung",
    "311": "Kwai Fong",
    "312": "Tsing Yi",
    "313": "Kwai Hing",
    "320": "Tsuen Wan",
    "321": "Tsuen Wan West",
    "325": "Sham Tseng",
    "330": "Tuen Mun",
    "331": "Tuen Mun South",
    "340": "Yuen Long",
    "341": "Yuen Long South",
    "342": "Tin Shui Wai",
    "350": "Sheung Shui",
    "351": "Fanling",
    "360": "Tai Po",
    "361": "Tai Po Hui",
    "370": "Sha Tin",
    "371": "Sha Tin Town Centre",
    "372": "Ma On Shan",
    "380": "Sai Kung",
    "381": "Tseung Kwan O South",
    "390": "Tung Chung",
    "391": "Lantau North",
    "700": "Kowloon (Extended)",
    "710": "Hung Hom Bay",
    "720": "Whampoa",
    "730": "Ho Man Tin South",
    "733": "Jordan Valley",
    "740": "Kwun Tong Waterfront",
    "750": "Kowloon East",
    "754": "Kowloon Bay Business Area",
    "755": "Kai Tak Development",
    "760": "New Kowloon",
    "761": "Diamond Hill North",
    "800": "New Territories (Extended)",
    "820": "Tuen Mun East",
    "824": "Tuen Mun North",
    "829": "Yuen Long East",
    "830": "Fanling North",
    "840": "Kwu Tung",
}
SCENARIOS = {
    "Balanced": {
        "w_skeleton": 0.4,
        "w_metabolism": 0.3,
        "w_circulatory": 0.3,
        "desc": "Equal emphasis · default setting",
    },
    "Building-led": {
        "w_skeleton": 0.6,
        "w_metabolism": 0.2,
        "w_circulatory": 0.2,
        "desc": "Focus on building density & FAR",
    },
    "Service-led": {
        "w_skeleton": 0.2,
        "w_metabolism": 0.5,
        "w_circulatory": 0.3,
        "desc": "Focus on services & green space",
    },
}
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
def load_data() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_PATH)
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
        "TPU_NUMBER",
    ]
    matching = [column for column in gdf.columns if column in candidates]
    return matching[0] if matching else gdf.columns[0]


def get_tpu_display_name(tpu_number) -> str:
    tpu_str = str(tpu_number).strip()
    if tpu_str in TPU_NAMES:
        return f"{tpu_str} · {TPU_NAMES[tpu_str]}"
    region_map = {
        "1": "HK Island",
        "2": "Kowloon",
        "3": "New Territories",
        "7": "Kowloon Ext.",
        "8": "NT Ext.",
        "9": "Outlying Islands",
    }
    region = region_map.get(tpu_str[0], "HK") if tpu_str else "Unknown"
    return f"{tpu_str} · {region}"


def build_priority_map(scored: gpd.GeoDataFrame, district_id_column: str) -> folium.Map:
    scored_json = scored.to_json()
    m = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=scored_json,
        data=scored[[district_id_column, "renewal_priority"]],
        columns=[district_id_column, "renewal_priority"],
        key_on=f"feature.properties.{district_id_column}",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        line_opacity=0.6,
        line_color="black",
        legend_name="Renewal Priority Score",
        highlight=True,
    ).add_to(m)

    folium.GeoJson(
        scored_json,
        name="TPU",
        style_function=lambda f: {
            "fillOpacity": 0,
            "weight": 0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "display_name",
                "renewal_priority",
                "priority_level",
                "skeleton_score",
                "metabolism_score",
                "circulatory_score",
            ],
            aliases=[
                "TPU:",
                "Priority Score:",
                "Level:",
                "Skeleton:",
                "Metabolism:",
                "Circulatory:",
            ],
            localize=True,
            sticky=True,
            labels=True,
            style="""
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
            """,
        ),
    ).add_to(m)

    return m


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

    if "scenario" not in st.session_state:
        st.session_state.scenario = "Balanced"

    st.sidebar.subheader("Assessment Scenario")
    for name, cfg in SCENARIOS.items():
        is_active = st.session_state.scenario == name
        if st.sidebar.button(
            name,
            key=f"btn_{name}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.scenario = name
            st.rerun()
        st.sidebar.caption(cfg["desc"])
        st.sidebar.write("")

    cfg = SCENARIOS[st.session_state.scenario]
    w_skeleton = cfg["w_skeleton"]
    w_metabolism = cfg["w_metabolism"]
    w_circulatory = cfg["w_circulatory"]

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**Weights:** Skeleton `{w_skeleton}` · "
        f"Metabolism `{w_metabolism}` · "
        f"Circulatory `{w_circulatory}`"
    )

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

    gdf = load_data()
    scored = build_composite_index(
        gdf.copy(),
        w_skeleton=w_skeleton,
        w_metabolism=w_metabolism,
        w_circulatory=w_circulatory,
    )
    scored["display_name"] = scored["TPU_NUMBER"].apply(get_tpu_display_name)
    id_col = get_district_id_column(scored)

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
        st.subheader(f"Interactive Priority Map - {st.session_state.scenario} Scenario")
        m = build_priority_map(scored, id_col)
        st_folium(m, width=700, height=500)

    with right_col:
        st.subheader("Urban Renewal Summary")

        high_count = int((scored["priority_level"] == "High").sum())
        mean_priority = float(scored["renewal_priority"].mean())
        top_idx = scored["renewal_priority"].idxmax()
        top_name = scored.loc[top_idx, "display_name"]

        metric_cols = st.columns(3)
        metric_cols[0].metric("High priority districts", high_count)
        metric_cols[1].metric("Mean priority score", f"{mean_priority:.3f}")
        metric_cols[2].metric("Top Priority TPU", top_name)

        st.markdown("**Average Dimension Scores**")
        avg_scores = (
            scored[["skeleton_score", "metabolism_score", "circulatory_score"]]
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
        top10_df = scored.nlargest(10, "renewal_priority")[
            [
                "display_name",
                "renewal_priority",
                "priority_level",
                "skeleton_score",
                "metabolism_score",
                "circulatory_score",
            ]
        ].copy()

        st.dataframe(
            top10_df.rename(
                columns={
                    "display_name": "TPU",
                    "renewal_priority": "Priority Score",
                    "priority_level": "Level",
                    "skeleton_score": "Skeleton",
                    "metabolism_score": "Metabolism",
                    "circulatory_score": "Circulatory",
                }
            )
            .reset_index(drop=True)
            .style.format(
                {
                    "Priority Score": "{:.3f}",
                    "Skeleton": "{:.3f}",
                    "Metabolism": "{:.3f}",
                    "Circulatory": "{:.3f}",
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
