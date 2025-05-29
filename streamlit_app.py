import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="Finance Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .query-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    .language-brief-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .brief-content {
        background: rgba(255,255,255,0.1);
        padding: 2rem;
        border-radius: 8px;
        border-left: 4px solid #00d4aa;
        font-family: 'Georgia', serif;
        line-height: 1.8;
        white-space: pre-wrap;
        font-size: 1.1rem;
        backdrop-filter: blur(5px);
    }
    
    .success-banner {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-banner {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .analysis-summary {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .no-brief-message {
        background: #fff3cd;
        color: #856404;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

class FinanceAssistantUI:
    def __init__(self):
        self.api_base_url = st.session_state.api_base_url
    
    def check_api_health(self) -> bool:
        """Check if FastAPI backend is accessible"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def send_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Send query to FastAPI backend"""
        try:
            payload = {"query": query}
            response = requests.post(
                f"{self.api_base_url}/query",
                json=payload,
                timeout=600
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code}")
                try:
                    error_response = response.json()
                    return {"error": f"HTTP {response.status_code}", "details": error_response}
                except:
                    return {"error": f"HTTP {response.status_code}", "details": response.text}
                
        except requests.exceptions.Timeout:
            st.error("Request timed out. The analysis is taking longer than expected.")
            return {"error": "Request timeout", "details": "The analysis took longer than 10 minutes"}
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
            return {"error": "Connection error", "details": str(e)}
    
    def display_header(self):
        """Display application header"""
        st.markdown('<h1 class="main-header">📈 AI Finance Assistant</h1>', unsafe_allow_html=True)
        st.markdown("*Intelligent financial analysis powered by multi-agent AI system*")
    
    def display_sidebar(self):
        """Display sidebar with controls and information"""
        with st.sidebar:
            st.header("🔧 Controls")
            
            # API Status
            is_healthy = self.check_api_health()
            status_color = "🟢" if is_healthy else "🔴"
            status_text = "Online" if is_healthy else "Offline"
            st.markdown(f"**API Status:** {status_color} {status_text}")
            
            if not is_healthy:
                st.warning("Backend API is not accessible. Please ensure the FastAPI server is running on port 8000.")
            
            st.divider()
            
            # Display Options
            st.header("📄 Display Options")
            
            display_mode = st.radio(
                "Primary Display:",
                ["AI Brief Only", "Brief + Summary", "Brief + Raw Data"]
            )
            
            show_metrics = st.checkbox("Show performance metrics", value=True)
            
            return {
                "display_mode": display_mode,
                "show_metrics": show_metrics
            }
    
    def display_query_interface(self):
        """Display query input interface"""
        st.markdown('<div class="query-container">', unsafe_allow_html=True)
        
        # Query input
        query = st.text_area(
            "Enter your financial analysis query:",
            height=100,
            placeholder="e.g., 'Comprehensive analysis for APPLE and SAMSUNG with recent performance data'"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            analyze_button = st.button("🚀 Get AI Analysis", type="primary", use_container_width=True)
        
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        with col3:
            if st.button("📚 History", use_container_width=True):
                st.session_state.show_history = not st.session_state.get('show_history', False)
        
        if clear_button:
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return query, analyze_button
    
    def find_language_brief(self, result: Dict[str, Any]) -> Optional[str]:
        """Enhanced method to find language brief in various possible locations"""
        # Safety check for None result
        if not result or not isinstance(result, dict):
            return None
        
        # Get analysis_result safely
        analysis_result = result.get('analysis_result', {})
        if not isinstance(analysis_result, dict):
            analysis_result = {}
        
        # Check if there are agent_results
        agent_results = analysis_result.get('agent_results', {})
        if isinstance(agent_results, dict):
            # Search through each agent's results
            for agent_name, agent_data in agent_results.items():
                if isinstance(agent_data, dict):
                    # Check for language brief in this agent
                    if agent_data.get('language_brief'):
                        brief = agent_data.get('language_brief')
                        return brief.strip()
                    
                    # Check for 'brief' field
                    if agent_data.get('brief'):
                        brief = agent_data.get('brief')
                        return brief.strip()
                    
                    # Check for 'result' field that might contain the brief
                    if agent_data.get('result') and isinstance(agent_data.get('result'), str):
                        brief = agent_data.get('result')
                        return brief.strip()
                    
                # Check if the entire agent_data is a string (sometimes the brief is stored directly)
                elif isinstance(agent_data, str) and len(agent_data) > 100:  # Likely a brief if it's a long string
                    return agent_data.strip()
        
        # Original search paths as fallback
        possible_paths = [
            analysis_result.get('language_brief'),
            result.get('language_brief'),
            analysis_result.get('brief'),
            result.get('brief'),
            result.get('analysis_result') if isinstance(result.get('analysis_result'), str) else None,
            analysis_result.get('language_brief', {}).get('brief') if isinstance(analysis_result.get('language_brief'), dict) else None,
        ]
        
        # Find first non-empty brief
        for brief in possible_paths:
            if brief and isinstance(brief, str) and brief.strip():
                return brief.strip()
        
        return None

    def display_language_brief(self, result: Dict[str, Any]):
        """Display the AI-generated language brief as the main content"""
        
        # Safety check for None result
        if not result:
            st.markdown("""
            <div class="no-brief-message">
                <h3>❌ No Response Data</h3>
                <p>The API response is empty or None. This could indicate:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>Backend server connection issues</li>
                    <li>API processing error</li>
                    <li>Request timeout</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            return False
        
        # Try to find the language brief
        language_brief = self.find_language_brief(result)
        
        if language_brief:
            st.markdown('<div class="language-brief-container">', unsafe_allow_html=True)
            
            # Header for the brief
            st.markdown("""
            <h2 style="margin: 0 0 1rem 0; color: white; display: flex; align-items: center;">
                🤖 AI Financial Analysis
                <span style="font-size: 0.6em; margin-left: 1rem; opacity: 0.8;">Generated by Multi-Agent System</span>
            </h2>
            """, unsafe_allow_html=True)
            
            # Display the brief content
            st.markdown(f'<div class="brief-content">{language_brief}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            return True
        else:
            # Show message when no brief is available
            st.markdown("""
            <div class="no-brief-message">
                <h3>🤖 No AI Brief Found</h3>
                <p>The language brief wasn't found in the expected locations. This might indicate:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>The response structure has changed</li>
                    <li>The language agent didn't generate a brief</li>
                    <li>The brief is stored in an unexpected location</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            return False
    
    def display_analysis_summary(self, result: Dict[str, Any]):
        """Display a summary of the analysis results"""
        if not result:
            st.warning("No result data available for summary")
            return
            
        analysis_result = result.get('analysis_result', {})
        processed_query = result.get('processed_query', {})
        
        # Ensure analysis_result is a dict
        if not isinstance(analysis_result, dict):
            analysis_result = {}
        if not isinstance(processed_query, dict):
            processed_query = {}
        
        st.markdown('<div class="analysis-summary">', unsafe_allow_html=True)
        st.subheader("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbols = processed_query.get('symbols', [])
            if not isinstance(symbols, list):
                symbols = []
            st.metric("Symbols Analyzed", len(symbols))
            if symbols:
                st.caption(", ".join(symbols[:3]) + ("..." if len(symbols) > 3 else ""))
        
        with col2:
            agents_used = len(analysis_result.get('agent_results', {})) if isinstance(analysis_result.get('agent_results'), dict) else 0
            st.metric("AI Agents Used", agents_used)
        
        with col3:
            processing_time = result.get('processing_time_seconds', 0)
            st.metric("Processing Time", f"{processing_time:.1f}s")
        
        with col4:
            success_status = result.get('success', False)
            status_text = "✅ Success" if success_status else "❌ Failed"
            st.metric("Status", status_text)
        
        # Show which agents were involved
        agent_results = analysis_result.get('agent_results', {})
        if isinstance(agent_results, dict) and agent_results:
            st.subheader("🤖 AI Agents Involved")
            agent_names = list(agent_results.keys())
            
            if agent_names:
                agent_columns = st.columns(min(len(agent_names), 4))  # Max 4 columns
                
                for i, agent_name in enumerate(agent_names):
                    col_idx = i % len(agent_columns)
                    with agent_columns[col_idx]:
                        agent_result = agent_results[agent_name]
                        if isinstance(agent_result, dict):
                            agent_success = agent_result.get('success', False)
                        else:
                            agent_success = True  # Assume success if not dict
                        icon = "✅" if agent_success else "❌"
                        st.write(f"{icon} **{agent_name.replace('_', ' ').title()}**")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_raw_data_section(self, result: Dict[str, Any]):
        """Display raw data in an expandable section"""
        if not result:
            st.warning("No raw data available to display")
            return
            
        with st.expander("🔍 View Raw Response Data", expanded=False):
            st.subheader("Raw FastAPI Response")
            
            # Display mode selector within the expander
            col1, col2 = st.columns(2)
            with col1:
                format_option = st.selectbox(
                    "Format:",
                    ["Pretty JSON", "Raw JSON", "Python Dict"],
                    key="raw_format"
                )
            
            if format_option == "Pretty JSON":
                json_str = json.dumps(result, indent=2, default=str, ensure_ascii=False)
                st.code(json_str, language='json')
            elif format_option == "Raw JSON":
                json_str = json.dumps(result, default=str, ensure_ascii=False)
                st.code(json_str, language='json')
            elif format_option == "Python Dict":
                st.code(repr(result), language='python')
            
            # Download button for raw data
            col1, col2 = st.columns(2)
            with col1:
                json_str = json.dumps(result, indent=2, default=str, ensure_ascii=False)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    def display_response(self, result: Dict[str, Any], settings: Dict[str, Any]):
        """Main response display method - focuses on language brief"""
        # Safety check for None result
        if result is None:
            st.error("❌ No response received from the API. Please check if the backend server is running.")
            return
        
        # Show performance metrics if enabled
        if settings.get('show_metrics', True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'processing_time_seconds' in result:
                    st.metric("⏱️ Processing Time", f"{result['processing_time_seconds']:.2f}s")
            
            with col2:
                if 'query_id' in result:
                    st.metric("🆔 Query ID", result['query_id'][:8] + "...")
            
            with col3:
                success_status = result.get('success', False)
                status_text = "✅ Success" if success_status else "❌ Failed"
                st.metric("📊 Status", status_text)
        
        # Display status banner
        if result.get('success', False):
            st.markdown('<div class="success-banner">✅ Analysis completed successfully</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-banner">❌ Analysis failed or returned error</div>', unsafe_allow_html=True)
            if result.get('error'):
                st.error(f"Error: {result['error']}")
        
        # Main content display based on settings
        display_mode = settings.get('display_mode', 'AI Brief Only')
        
        # Always try to display the language brief first
        brief_displayed = self.display_language_brief(result)
        
        # Additional content based on display mode
        if display_mode == "Brief + Summary":
            st.divider()
            self.display_analysis_summary(result)
            
        elif display_mode == "Brief + Raw Data":
            st.divider()
            self.display_analysis_summary(result)
            st.divider()
            self.display_raw_data_section(result)
    
    def display_query_history(self):
        """Display query history"""
        if st.session_state.get('show_history', False) and st.session_state.query_history:
            with st.expander("📚 Recent Query History", expanded=True):
                for i, (timestamp, query, success) in enumerate(reversed(st.session_state.query_history[-5:])):
                    status_icon = "✅" if success else "❌"
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{status_icon} **{timestamp}**: {query[:80]}{'...' if len(query) > 80 else ''}")
                    with col2:
                        if st.button("🔄 Rerun", key=f"rerun_{i}"):
                            st.session_state.example_query = query
                            st.rerun()
    
    def run(self):
        """Main application runner"""
        self.display_header()
        
        # Sidebar
        settings = self.display_sidebar()
        
        # Main interface
        query, analyze_button = self.display_query_interface()
        
        # Query history
        self.display_query_history()
        
        # Process query
        if analyze_button and query.strip():
            with st.spinner("🔄 Analyzing with AI agents... This may take a moment."):
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                # Simulate progress updates
                for i in range(0, 101, 20):
                    progress_bar.progress(i)
                    if i == 0:
                        progress_text.text("Initializing AI agents...")
                    elif i == 20:
                        progress_text.text("Processing your query...")
                    elif i == 40:
                        progress_text.text("Gathering financial data...")
                    elif i == 60:
                        progress_text.text("Running analysis...")
                    elif i == 80:
                        progress_text.text("Generating insights...")
                    elif i == 100:
                        progress_text.text("Finalizing response...")
                    
                    time.sleep(0.3)
                
                # Actual API call
                start_time = time.time()
                result = self.send_query(query.strip())
                end_time = time.time()
                
                # Clear progress indicators
                progress_bar.empty()
                progress_text.empty()
                
                # Store in history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                success = result is not None and result.get('success', False)
                st.session_state.query_history.append((timestamp, query.strip(), success))
                
                # Display results - now with proper None check
                if result is not None:
                    result['_client_request_time_seconds'] = end_time - start_time
                    self.display_response(result, settings)
                else:
                    st.error("❌ No response received from the API. Please check if the backend server is running.")
        
        elif analyze_button and not query.strip():
            st.warning("⚠️ Please enter a query before analyzing.")

def main():
    """Main entry point"""
    app = FinanceAssistantUI()
    app.run()

if __name__ == "__main__":
    main()