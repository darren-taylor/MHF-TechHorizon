import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- PAGE CONFIG ---
st.set_page_config(page_title="MHF-TechHorizon | API RP 17Q Agent", layout="wide")
st.title("🛡️ MHF-TechHorizon Research Assistant")
st.caption("AI-Powered Novel Technology Assessment under API RP 17Q for Major Hazard Facilities")

# --- SIDEBAR API KEY & CONFIG ---
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
4. Tool Usage: Use your search tool to gather real, current academic and technical research when needed.

OUTPUT FORMAT:
- 📌 Overview of Challenge & Degradation Mechanism
- 🔬 Novel Technologies & API RP 17Q Classification
- 📊 API RP 17Q Qualification Matrix (Table: Technology | ISD Strategy | Current TRL | Critical Failure Modes | Next Evidence Required)
- ⚠️ Regulatory & Hazard Facility Considerations
"""

# --- AGENT INITIALIZATION ---
if api_key:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1, api_key=api_key)
    tools = [DuckDuckGoSearchResults(name="academic_search")]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
else:
    agent_executor = None

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
    if not agent_executor:
        st.error("Please enter an OpenAI API key in the sidebar to run the agent.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing technical literature & API RP 17Q evidence gates..."):
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": []
                })
                output_text = response["output"]
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
