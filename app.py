import streamlit as st
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from duckduckgo_search import DDGS

# --- PAGE CONFIG ---
st.set_page_config(page_title="MHF-TechHorizon | API RP 17Q Agent", layout="wide")
st.title("🛡️ MHF-TechHorizon Research Assistant")
st.caption("AI-Powered Novel Technology Assessment under API RP 17Q for Major Hazard Facilities")

# --- CUSTOM SEARCH TOOL ---
def search_public_research(query: str) -> str:
    """Searches open web literature, standards, and research papers."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=4):
            results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nSource: {r['href']}")
    return "\n\n".join(results) if results else "No direct results found."

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key or configure it in secrets.")
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    
    st.markdown("---")
    st.markdown("### Demo Preset Scenarios")
    preset = st.selectbox(
        "Load a Test Case:",
        [
            "Select a test...",
            "1. High-Temp Corrosion Monitoring (TRL 3-4)",
            "2. Autonomous OGI in Hydrogen (TRL 4-5)",
            "3. Vendor Claim Verification & Guardrail Test"
        ]
    )

# --- SYSTEM PROMPT (API RP 17Q) ---
SYSTEM_PROMPT = """
You are "MHF-TechHorizon", an expert AI research assistant specializing in emerging technologies and qualification for Major Hazard Facilities (MHFs).

EVALUATION FRAMEWORK (API RP 17Q):
Classify technologies using the 8-level scale:
- TRL 0: Unproven Concept
- TRL 1: Proven Concept
- TRL 2: Validated Concept (Experimental proof)
- TRL 3: Prototype Tested (Lab environment)
- TRL 4: Environment Tested (Simulated/intended operating environment)
- TRL 5: System Tested (Full-scale rig / intended environment)
- TRL 6: System Installed (Commissioned in live conditions)
- TRL 7: Field Proven (Sustained operational history)

OPERATIONAL RULES:
1. Inherent Safety Mapping: Map every tech to Minimize, Substitute, Moderate, or Simplify.
2. Evidence Requirements: Specify exact experimental/operational deliverables (e.g., FMECA, thermal cycles, FAR metrics) needed to advance the TRL.
3. Guardrail / Zero Proprietary Data: Refuse to ingest internal plant drawings, proprietary P&IDs, or confidential schematics.
4. Tool Usage: Use your search_public_research tool to verify recent literature when needed.

OUTPUT FORMAT:
- 📌 Overview of Challenge & Degradation Mechanism
- 🔬 Novel Technologies & API RP 17Q Classification
- 📊 API RP 17Q Qualification Matrix (Table: Technology | ISD Strategy | Current TRL | Critical Failure Modes | Next Evidence Required)
- ⚠️ Regulatory & Hazard Facility Considerations
"""

# --- AGENT INITIALIZATION ---
agent = None
if api_key:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)
    # create_react_agent is the native LangGraph agent executor
    agent = create_react_agent(
        model=llm,
        tools=[search_public_research],
        prompt=SYSTEM_PROMPT
    )

# --- CHAT STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- HANDLE PRESETS ---
preset_prompts = {
    "1. High-Temp Corrosion Monitoring (TRL 3-4)": "What novel technologies are emerging for continuous, non-intrusive monitoring of internal pipe wall thinning from naphthenic acid and sulfidic corrosion in high-temperature crude units? Assess under API RP 17Q.",
    "2. Autonomous OGI in Hydrogen (TRL 4-5)": "We are assessing novel continuous optical gas imaging (OGI) paired with edge-AI for autonomous loss-of-containment detection in high-pressure hydrogen service. What is the current maturity under API RP 17Q, and what evidence is needed to prove field reliability?",
    "3. Vendor Claim Verification & Guardrail Test": "A vendor claims their quantum-dot electrochemical gas sensor is TRL 7 and ready for full toxic gas field integration across our plant. Here is our internal site gas-grid layout (Plant-Area-04-GasMap.dwg). How do we validate their claim against API RP 17Q?"
}

user_input = st.chat_input("Ask about emerging MHF technologies, qualification evidence, or TRLs...")

if preset != "Select a test...":
    user_input = preset_prompts[preset]

# --- EXECUTE ---
if user_input:
    if not agent:
        st.error("Please enter an OpenAI API key in the sidebar to run the agent.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing technical literature & API RP 17Q evidence gates..."):
                response = agent.invoke({
                    "messages": [{"role": "user", "content": user_input}]
                })
                # LangGraph returns the full message history; the last message is the AI response
                output_text = response["messages"][-1].content
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
