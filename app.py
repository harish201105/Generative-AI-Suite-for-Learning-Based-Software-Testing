import streamlit as st
from api.models.gpt_model import GPTModel
from api.models.cohere_model import CohereModel
from api.models.mistral_model import MistralModel
from api.models.llama_model import LlamaModel
from api.models.qwen_model import QwenModel
from api.models.phi_model import PhiModel
from api.models.claude_model import ClaudeModel
from api.models.gemini_model import GeminiModel
from api.models.deepseek_v3_model import DeepSeekV3Model
from api.models.grok3_model import Grok3Model
from api.analysis.complexity_analyzer import ComplexityAnalyzer
from api.analysis.performance_analyzer import PerformanceAnalyzer
import os
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

# Enhanced page configuration
st.set_page_config(
    page_title="🧠 AI Code Analyzer Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main app styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.7rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Error styling */
    .stSuccess {
        border-left: 5px solid #28a745;
        border-radius: 5px;
    }
    
    .stError {
        border-left: 5px solid #dc3545;
        border-radius: 5px;
    }
    
    .stWarning {
        border-left: 5px solid #ffc107;
        border-radius: 5px;
    }
    
    .stInfo {
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        color: white;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1>🧠 Advanced AI Code Analyzer Pro</h1>
    <p class="subtitle">Intelligent Python Code Analysis & Multi-Model AI Benchmarking Platform</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    
    # Quick stats
    st.markdown("### 📊 Session Stats")
    if 'model_calls' not in st.session_state:
        st.session_state.model_calls = 0
    if 'total_tests_generated' not in st.session_state:
        st.session_state.total_tests_generated = 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🔄 API Calls", st.session_state.model_calls)
    with col2:
        st.metric("🧪 Tests Generated", st.session_state.total_tests_generated)
    
    st.markdown("---")
    
    # Model status indicators
    st.markdown("### 🤖 Model Status")
    
    model_status = {
        "GPT-5": "🟢 Ready",
        "Claude 4": "🟢 Ready",
        "Gemini 2.5": "🟢 Ready",
        "Mistral": "🟢 Ready",
        "Llama-4": "🟢 Ready",
        "Phi 4": "🟢 Ready",
        "Qwen3": "🟢 Ready",
        "DeepSeek-V3": "🟢 Ready",
        "Grok 3": "🟢 Ready",
        "Cohere": "🟢 Ready"
    }
    
    for model, status in model_status.items():
        st.markdown(f"**{model}:** {status}")
    
    st.markdown("---")
    
    # Quick settings
    st.markdown("### ⚙️ Settings")
    
    # Analysis settings
    enable_complexity_analysis = st.checkbox("🧮 Enable Complexity Analysis", value=True)
    enable_performance_tracking = st.checkbox("📊 Enable Performance Tracking", value=True)
    show_debug_info = st.checkbox("🔍 Show Debug Information", value=False)
    
    # Display settings
    st.markdown("### 🎨 Display Options")
    theme_choice = st.selectbox("🎨 Theme", ["Professional", "Dark", "Light"], index=0)
    result_layout = st.selectbox("📋 Result Layout", ["Expandable", "Tabs", "Cards"], index=0)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.session_state.model_calls = 0
        st.session_state.total_tests_generated = 0
        st.success("Cache cleared!")
    
    if st.button("📥 Export Session", use_container_width=True):
        session_data = {
            "model_calls": st.session_state.model_calls,
            "tests_generated": st.session_state.total_tests_generated,
            "timestamp": time.time()
        }
        st.download_button(
            "📄 Download Session Data",
            data=json.dumps(session_data, indent=2),
            file_name="session_data.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Help and info
    st.markdown("### ❓ Help & Info")
    
    with st.expander("🔧 Setup Guide"):
        st.markdown("""
        **API Keys Required:**
        - OpenAI API Key
        - Anthropic API Key  
        - Google AI API Key
        - Mistral API Key
        - Other provider keys
        
        **Environment Setup:**
        1. Create `.env` file
        2. Add API keys
        3. Restart application
        """)
    
    with st.expander("📖 User Guide"):
        st.markdown("""
        **How to Use:**
        1. Paste Python code
        2. Select AI models
        3. Generate test cases
        4. Compare results
        5. Download outputs
        
        **Best Practices:**
        - Use clear function names
        - Include docstrings
        - Test edge cases
        """)
    
    with st.expander("🚀 Pro Tips"):
        st.markdown("""
        **Performance Tips:**
        - Start with 2-3 models
        - Use Quick Select buttons
        - Monitor response times
        - Enable caching
        
        **Quality Tips:**
        - Write clear code
        - Add type hints
        - Include error handling
        """)
    
    st.markdown("---")
    st.markdown("**Version:** 2.0.0 | **Updated:** Sept 2025")

# Initialize models and analyzers with error handling
try:
    gpt_model = GPTModel()
    cohere_model = CohereModel()
    mistral_model = MistralModel()
    llama_model = LlamaModel()
    qwen_model = QwenModel()
    phi_model = PhiModel()
    claude_model = ClaudeModel()
    gemini_model = GeminiModel()
    deepseek_v3_model = DeepSeekV3Model()
    grok3_model = Grok3Model()
    complexity_analyzer = ComplexityAnalyzer()  # Now uses the advanced analyzer
    
    # Initialize performance analyzer with session state
    if 'performance_analyzer' not in st.session_state:
        st.session_state.performance_analyzer = PerformanceAnalyzer()
    performance_analyzer = st.session_state.performance_analyzer
    
    models_available = True
except ValueError as e:
    st.error(f"Configuration Error: {str(e)}")
    st.info("Please check your .env file and ensure you have the required API keys configured.")
    models_available = False
except Exception as e:
    st.error(f"Unexpected error initializing models: {str(e)}")
    models_available = False

if not models_available:
    st.warning("⚠️ No AI models are currently available.")
    st.markdown("""
    **To fix this issue:**
    
    1. Create a `.env` file in the project root directory
    2. Add your GitHub token: `GITHUB_TOKEN=your_github_token_here`
    3. Make sure the token has AI inference permissions
    4. Restart the application
    
    **Alternative:** You can also configure other AI services like OpenAI or Google Gemini by adding their respective API keys to the `.env` file.
    """)
    st.stop()

# Enhanced code input section
st.markdown("### 📝 Code Input")
st.markdown("Enter your Python code below for comprehensive analysis and intelligent test case generation.")

# Code input with better styling
col1, col2 = st.columns([3, 1])

with col1:
    code_snippet = st.text_area(
        "Python Code",
        height=300,
        placeholder="""# Example: Paste your Python code here
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def calculate_sum(numbers):
    return sum(numbers)
    
# Your code will be analyzed for complexity and test cases will be generated
""",
        help="Paste your Python code here. The analyzer works best with functions and classes."
    )

with col2:
    st.markdown("#### 💡 Tips")
    st.info("""
    **Best Results:**
    • Include function definitions
    • Add docstrings if available
    • Use meaningful variable names
    • Include edge cases in comments
    
    **Supported:**
    • Functions & Classes
    • Complex algorithms
    • Data structures
    • Mathematical operations
    """)
    
    # Code stats
    if code_snippet:
        lines = len(code_snippet.split('\n'))
        chars = len(code_snippet)
        st.metric("Lines of Code", lines)
        st.metric("Characters", chars)

if models_available:
    # Enhanced model selection with provider grouping
    st.markdown("### 🤖 AI Model Selection")
    st.markdown("Choose from our premium collection of AI models from leading providers worldwide.")
    
    # Organize models by provider
    model_providers = {
        "🔮 OpenAI": {
            "GPT-5 (with GPT-4o fallback)": gpt_model,
        },
        "🧠 Anthropic": {
            "Claude Sonnet 4": claude_model,
        },
        "🌟 Google": {
            "Gemini 2.5 Flash": gemini_model,
        },
        "⚡ Mistral AI": {
            "Mistral Medium 2505": mistral_model,
        },
        "🦙 Meta": {
            "Llama-4-Maverick-17B-128E": llama_model,
        },
        "🔬 Microsoft": {
            "Phi 4": phi_model,
        },
        "🎯 Alibaba": {
            "Qwen3-32B": qwen_model,
        },
        "🚀 DeepSeek": {
            "DeepSeek-V3-0324": deepseek_v3_model,
        },
        "⚡ xAI": {
            "Grok 3": grok3_model,
        },
        "🌐 Cohere": {
            "Cohere Command A": cohere_model,
        }
    }
    
    # Flatten for the multiselect
    available_models = {}
    for provider, models in model_providers.items():
        for model_name, model_obj in models.items():
            available_models[model_name] = model_obj
    
    # Create expandable sections for each provider
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_models = st.multiselect(
            "Select Models to Compare",
            options=list(available_models.keys()),
            help="Choose one or more models to generate and compare test cases",
            default=[]
        )
    
    with col2:
        st.markdown("#### 🎯 Quick Select")
        if st.button("🔥 Best Performers", help="Select top 3 fastest models"):
            selected_models = ["GPT-5 (with GPT-4o fallback)", "Gemini 2.5 Flash", "Qwen3-32B"]
            st.rerun()
        
        if st.button("🌟 Premium Models", help="Select premium flagship models"):
            selected_models = ["GPT-5 (with GPT-4o fallback)", "Claude Sonnet 4", "Gemini 2.5 Flash"]
            st.rerun()
        
        if st.button("⚡ Speed Demons", help="Select fastest response models"):
            selected_models = ["Phi 4", "Qwen3-32B", "DeepSeek-V3-0324"]
            st.rerun()
    
    # Show provider breakdown
    if selected_models:
        st.markdown("#### 📊 Selected Models by Provider")
        provider_count = {}
        for model in selected_models:
            for provider, models in model_providers.items():
                if model in models:
                    provider_count[provider] = provider_count.get(provider, 0) + 1
                    break
        
        cols = st.columns(len(provider_count))
        for i, (provider, count) in enumerate(provider_count.items()):
            with cols[i]:
                st.metric(provider, f"{count} model{'s' if count > 1 else ''}")

# Enhanced generate button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button(
        "🚀 Generate Test Cases & Analysis",
        help="Start comprehensive code analysis and test case generation",
        use_container_width=True
    )

if generate_button:
    if not code_snippet:
        st.error("❌ Please provide code to generate test cases.")
    elif not selected_models:
        st.error("❌ Please select at least one model.")
    else:
        # Create enhanced tabs for results
        results_tab, comparison_tab, analytics_tab = st.tabs([
            "📋 Test Cases & Analysis", 
            "📊 Performance Comparison", 
            "📈 Advanced Analytics"
        ])
        
        with results_tab:
            st.markdown("### 🔬 AI Model Results")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_models = len(selected_models)
            
            for idx, model_name in enumerate(selected_models):
                # Update progress
                progress = (idx + 1) / total_models
                progress_bar.progress(progress)
                status_text.text(f"Processing with {model_name}... ({idx + 1}/{total_models})")
                
                # Create expandable section for each model
                with st.expander(f"🤖 {model_name} Results", expanded=True):
                    model = available_models[model_name]
                    start_time = time.time()
                    
                    with st.spinner(f"Generating test cases using {model_name}..."):
                        result = model.analyze_code(code_snippet)
                    
                    if "error" in result:
                        st.error(f"❌ Test case generation failed: {result['error']}")
                        if "raw_output" in result:
                            # Use a details container instead of nested expander
                            st.markdown("#### 🔍 Debug Information")
                            debug_tab1, debug_tab2, debug_tab3 = st.tabs(["Raw Output", "Cleaned Output", "Fixed Output"])
                            
                            with debug_tab1:
                                st.markdown("**Raw Model Output:**")
                                st.code(result["raw_output"])
                            
                            with debug_tab2:
                                if "cleaned_output" in result:
                                    st.markdown("**Cleaned Output:**")
                                    st.code(result["cleaned_output"])
                                else:
                                    st.info("No cleaned output available")
                            
                            with debug_tab3:
                                if "fixed_output" in result:
                                    st.markdown("**Fixed Output:**")
                                    st.code(result["fixed_output"])
                                else:
                                    st.info("No fixed output available")
                    else:
                        # Success - display results with enhanced formatting
                        performance = performance_analyzer.analyze_model_performance(
                            model_name, result, start_time
                        )
                        
                        # Success indicator
                        st.success(f"✅ Successfully generated test cases with {model_name}")
                        
                        try:
                            test_cases = result["test_cases"]
                            
                            # Enhanced info display
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🎯 Test Framework", result.get("test_framework", "pytest"))
                            with col2:
                                st.metric("📊 Test Cases", len(test_cases))
                            with col3:
                                st.metric("⏱️ Response Time", f"{time.time() - start_time:.2f}s")
                            with col4:
                                st.metric("🎯 Coverage Areas", len(result.get("coverage_areas", [])))
                            
                            # Coverage areas with badges
                            if result.get("coverage_areas"):
                                st.markdown("#### 📋 Coverage Areas")
                                coverage_html = ""
                                for area in result.get("coverage_areas", []):
                                    coverage_html += f'<span style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.3rem 0.8rem; margin: 0.2rem; border-radius: 15px; display: inline-block; font-size: 0.9rem;">{area}</span>'
                                st.markdown(coverage_html, unsafe_allow_html=True)
                            
                            # Enhanced test cases display
                            st.markdown("#### 🧪 Generated Test Cases")
                            
                            # Create tabs for test cases instead of nested expanders
                            if len(test_cases) > 0:
                                test_case_tabs = st.tabs([f"Test {i+1}: {test_case.get('name', 'Unnamed')}" for i, test_case in enumerate(test_cases)])
                                
                                for i, (test_case, tab) in enumerate(zip(test_cases, test_case_tabs)):
                                    with tab:
                                        # Test Description with icon
                                        st.markdown("**📝 Description:**")
                                        st.markdown(test_case.get("description", "No description provided"))
                                        
                                        # Input and Expected Output in columns
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("**📥 Input:**")
                                            st.code(test_case.get("input", "No input provided"), language="python")
                                        
                                        with col2:
                                            st.markdown("**📤 Expected Output:**")
                                            st.code(test_case.get("expected_output", "No expected output provided"), language="python")
                                        
                                        # Test Code with copy button
                                        st.markdown("**🔬 Test Code:**")
                                        test_code = test_case.get("test_code", "No test code provided")
                                        st.code(test_code, language="python")
                            else:
                                st.info("No test cases generated.")
                            
                            # Complexity analysis with enhanced display
                            complexity_results = complexity_analyzer.analyze_test_case(code_snippet)
                            
                            st.markdown("#### 🧮 Complexity Analysis")
                            col1, col2 = st.columns(2)
                            with col1:
                                time_complexity = complexity_results[0]
                                st.markdown("**⏰ Time Complexity:**")
                                complexity_color = "success" if "O(1)" in time_complexity or "O(log" in time_complexity else "warning" if "O(n)" in time_complexity else "error"
                                st.markdown(f'<div class="st{complexity_color.title()}">{time_complexity}</div>', unsafe_allow_html=True)
                            with col2:
                                space_complexity = complexity_results[1]
                                st.markdown("**💾 Space Complexity:**")
                                space_color = "success" if "O(1)" in space_complexity else "warning" if "O(n)" in space_complexity else "error"
                                st.markdown(f'<div class="st{space_color.title()}">{space_complexity}</div>', unsafe_allow_html=True)
                            
                            # Enhanced download button
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label=f"📥 Download {model_name} Test Cases",
                                    data=json.dumps(test_cases, indent=2),
                                    file_name=f"test_cases_{model_name.lower().replace(' ', '_')}.json",
                                    mime="application/json",
                                    use_container_width=True
                                )
                            with col2:
                                # Copy all test code button
                                all_test_code = "\n\n".join([tc.get("test_code", "") for tc in test_cases])
                                st.download_button(
                                    label=f"📋 Download Complete Test Suite",
                                    data=all_test_code,
                                    file_name=f"test_suite_{model_name.lower().replace(' ', '_')}.py",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                            
                        except json.JSONDecodeError:
                            st.error("❌ Failed to parse test cases. Raw output:")
                            st.code(result["test_cases"])
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            st.success(f"🎉 Completed analysis for all {total_models} selected models!")
        
        with comparison_tab:
            st.markdown("### 📊 Model Performance Comparison")
            
            # Get performance comparison data
            comparison_data = performance_analyzer.get_performance_comparison()
            
            # Check if there are any valid models to compare
            if comparison_data['rankings']['best_model'] == 'No models available':
                st.warning("⚠️ No models have successfully generated test cases yet.")
                st.info("Please ensure your API keys are properly configured and try generating test cases again.")
            else:
                # Enhanced model rankings with cards
                st.markdown("#### 🏆 Model Rankings")
                rankings = comparison_data['rankings']
                
                # Display best and worst models with enhanced styling
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);">
                        <h3>🥇 Best Model</h3>
                        <h4>{}</h4>
                        <p>Score: {:.2f}</p>
                    </div>
                    """.format(
                        rankings['best_model'],
                        rankings['rankings'].get(rankings['best_model'], 0)
                    ), unsafe_allow_html=True)
                
                with col2:
                    # Average performer
                    if len(rankings['rankings']) > 2:
                        sorted_models = sorted(rankings['rankings'].items(), key=lambda x: x[1], reverse=True)
                        middle_model = sorted_models[len(sorted_models)//2]
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #ffc107, #fd7e14); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3);">
                            <h3>🥈 Average Model</h3>
                            <h4>{}</h4>
                            <p>Score: {:.2f}</p>
                        </div>
                        """.format(middle_model[0], middle_model[1]), unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #dc3545, #fd1744); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(220, 53, 69, 0.3);">
                        <h3>🥉 Needs Improvement</h3>
                        <h4>{}</h4>
                        <p>Score: {:.2f}</p>
                    </div>
                    """.format(
                        rankings['worst_model'],
                        rankings['rankings'].get(rankings['worst_model'], 0)
                    ), unsafe_allow_html=True)
                
                # Enhanced detailed metrics
                st.markdown("#### 📈 Detailed Performance Metrics")
                for model, metrics in rankings['detailed_metrics'].items():
                    with st.expander(f"🤖 {model} - Detailed Analysis", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            response_time = metrics['response_time']
                            time_color = "#28a745" if response_time < 3 else "#ffc107" if response_time < 6 else "#dc3545"
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: {time_color}20; border-radius: 10px; border-left: 4px solid {time_color};">
                                <h4 style="margin: 0; color: {time_color};">⏱️ Response Time</h4>
                                <h3 style="margin: 0.5rem 0;">{response_time:.2f}s</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            test_count = int(metrics['test_case_count'])
                            count_color = "#28a745" if test_count >= 8 else "#ffc107" if test_count >= 5 else "#dc3545"
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: {count_color}20; border-radius: 10px; border-left: 4px solid {count_color};">
                                <h4 style="margin: 0; color: {count_color};">🧪 Test Cases</h4>
                                <h3 style="margin: 0.5rem 0;">{test_count}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            complexity_score = metrics['complexity_score']
                            complexity_color = "#28a745" if complexity_score >= 8 else "#ffc107" if complexity_score >= 6 else "#dc3545"
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: {complexity_color}20; border-radius: 10px; border-left: 4px solid {complexity_color};">
                                <h4 style="margin: 0; color: {complexity_color};">🧮 Complexity</h4>
                                <h3 style="margin: 0.5rem 0;">{complexity_score:.1f}/10</h3>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            overall_score = metrics['overall_score']
                            overall_color = "#28a745" if overall_score >= 8 else "#ffc107" if overall_score >= 6 else "#dc3545"
                            st.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: {overall_color}20; border-radius: 10px; border-left: 4px solid {overall_color};">
                                <h4 style="margin: 0; color: {overall_color};">🎯 Overall</h4>
                                <h3 style="margin: 0.5rem 0;">{overall_score:.2f}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Enhanced summary with progress bars
                st.markdown("#### 📊 Performance Summary")
                if comparison_data['summary']['response_time']:
                    avg_response_time = sum(comparison_data['summary']['response_time'].values()) / len(comparison_data['summary']['response_time'])
                    avg_test_cases = sum(comparison_data['summary']['test_case_count'].values()) / len(comparison_data['summary']['test_case_count'])
                    avg_complexity = sum(comparison_data['summary']['complexity_score'].values()) / len(comparison_data['summary']['complexity_score'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Avg Response Time", f"{avg_response_time:.2f}s")
                        st.progress(min(avg_response_time / 10, 1.0))  # Max 10s scale
                    with col2:
                        st.metric("🧪 Avg Test Cases", f"{avg_test_cases:.1f}")
                        st.progress(min(avg_test_cases / 15, 1.0))  # Max 15 tests scale
                    with col3:
                        st.metric("🧮 Avg Complexity Score", f"{avg_complexity:.1f}/10")
                        st.progress(avg_complexity / 10)
                
                # Enhanced performance plots
                st.markdown("#### 📈 Performance Visualizations")
                
                try:
                    plots = performance_analyzer.plot_performance_comparison()
                    
                    # Basic Performance Metrics with better layout
                    st.markdown("##### 🔍 Core Performance Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plots['response_time'])
                    with col2:
                        st.pyplot(plots['test_case_count'])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plots['complexity_score'])
                    with col2:
                        st.pyplot(plots['overall_score'])
                    
                    # Test Case Analysis
                    st.markdown("##### 🧪 Test Case Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plots['input_types'])
                    with col2:
                        st.pyplot(plots['operation_coverage'])
                    
                    st.markdown("##### 📊 Test Case Distribution")
                    st.pyplot(plots['test_case_types'])
                    
                    # Combined Performance
                    st.markdown("##### 🎯 Comprehensive Performance Matrix")
                    st.pyplot(plots['heatmap'])
                except Exception as e:
                    st.warning("⚠️ Unable to generate performance visualizations.")
                    st.info("This usually happens when there's insufficient data or when models haven't generated test cases yet.")
                
                # Enhanced raw data display
                with st.expander("📋 Raw Performance Data", expanded=False):
                    if not comparison_data['dataframe'].empty:
                        st.dataframe(
                            comparison_data['dataframe'],
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No performance data available yet.")
        
        with analytics_tab:
            st.markdown("### 📊 Advanced Analytics")
            st.info("🚧 Advanced analytics features coming soon! This will include deeper insights, trends analysis, and ML-powered recommendations.")
            
            # Placeholder for future analytics features
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                #### 🔮 Coming Soon:
                - **Trend Analysis**: Track performance over time
                - **Model Recommendations**: AI-powered model selection
                - **Code Quality Insights**: Advanced code metrics
                - **Optimization Suggestions**: Performance improvement tips
                """)
            
            with col2:
                st.markdown("""
                #### 📈 Future Features:
                - **Comparative Analysis**: Side-by-side detailed comparisons
                - **Custom Metrics**: Define your own evaluation criteria
                - **Export Reports**: Generate comprehensive PDF reports
                - **API Integration**: Connect with your CI/CD pipeline
                """)
            
            # Show some basic stats if available
            if comparison_data['rankings']['best_model'] != 'No models available':
                st.markdown("#### 📊 Session Statistics")
                total_models = len(comparison_data['rankings']['rankings'])
                total_tests = sum(comparison_data['summary']['test_case_count'].values()) if comparison_data['summary']['test_case_count'] else 0
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🤖 Models Used", total_models)
                with col2:
                    st.metric("🧪 Total Tests Generated", int(total_tests))
                with col3:
                    st.metric("⏱️ Total Processing Time", f"{sum(comparison_data['summary']['response_time'].values()):.2f}s" if comparison_data['summary']['response_time'] else "0s")
                with col4:
                    st.metric("🎯 Success Rate", "100%" if total_models > 0 else "0%")