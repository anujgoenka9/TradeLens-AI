"""
This module implements a CrewAI-based stock market analysis system that combines
technical analysis, news sentiment, and market overview data to generate
comprehensive investment insights.
"""
import streamlit as st
import asyncio
import json
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from groq import Groq
import os
from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from business_intelligence_agent.crew import MarketAnalysisCrew
import nest_asyncio
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import Dict, Any, Optional
import yaml
import logging
from business_intelligence_agent.ticker_fetcher import TickerFetcher
import sys
import base64
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Terminal output
        logging.FileHandler('app.log'),  # File output
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Path to MCP server
MCP_SERVER_PATH = Path(__file__).parent / "data_fetcher.py"

# Set page config
st.set_page_config(
    page_title="Market Intelligence Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .market-stats {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e6f3ff;
    }
    .assistant-message {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_symbol" not in st.session_state:
        st.session_state.current_symbol = None
    if "market_data" not in st.session_state:
        st.session_state.market_data = None
    if "crew" not in st.session_state:
        st.session_state.crew = None
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)
    if "ticker_fetcher" not in st.session_state:
        try:
            logger.info("Initializing TickerFetcher and ChromaDB connection...")
            st.session_state.ticker_fetcher = TickerFetcher()
            
            # Test the ChromaDB connection and get collection info
            collection = st.session_state.ticker_fetcher.collection
            logger.info(f"ChromaDB collection name: {collection.name}")
            
            # Get a sample query to verify data exists
            test_results = collection.query(
                query_texts=["technology"],
                n_results=1
            )
            
            if test_results and test_results.get('metadatas') and test_results['metadatas'][0]:
                logger.info("ChromaDB test query successful")
            else:
                logger.warning("ChromaDB test query returned no results - database might be empty")
                st.warning("ChromaDB database appears to be empty. Please run the data loading script first: python ticker_fetcher.py --load")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}", exc_info=True)
            st.error("Error connecting to ChromaDB. Please make sure you have run the data loading script first: python ticker_fetcher.py --load")
            
    # Add new session state variables for persistent display
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "current_market_analysis" not in st.session_state:
        st.session_state.current_market_analysis = None
    if "current_news_analysis" not in st.session_state:
        st.session_state.current_news_analysis = None
    if "current_strategy" not in st.session_state:
        st.session_state.current_strategy = None
    if "current_recommendations" not in st.session_state:
        st.session_state.current_recommendations = None
    if "current_company_info" not in st.session_state:
        st.session_state.current_company_info = None
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Summary"

def run_async(coro):
    logger.info(f"[app.py] Running async coroutine: {coro}")
    try:
        result = st.session_state.loop.run_until_complete(coro)
        logger.info(f"[app.py] Coroutine result: {result}")
        return result
    except Exception as e:
        logger.error(f"[app.py] Error running async operation: {str(e)}", exc_info=True)
        return None

async def discover_tools():
    """Connect to the MCP server and discover available tools."""
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            
            tool_info = []
            for tool_type, tool_list in tools:
                if tool_type == "tools":
                    for tool in tool_list:
                        tool_info.append({
                            "name": tool.name,
                            "description": tool.description,
                            "schema": tool.inputSchema
                        })
            return tool_info

async def execute_tool(tool_name: str, arguments: dict):
    logger.info(f"[app.py] Executing MCP tool: {tool_name} with arguments: {arguments}")
    server_params = StdioServerParameters(
        command="python",
        args=[str(MCP_SERVER_PATH)],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            logger.info(f"[app.py] MCP tool '{tool_name}' result: {result}")
            return result

async def query_groq(prompt: str, tools, messages):
    """Process a query through Groq using available tools."""
    tool_descriptions = "\n\n".join([
        f"Tool: {tool['name']}\nDescription: {tool['description']}\nSchema: {json.dumps(tool['schema'], indent=2)}"
        for tool in tools
    ])
    
    system_prompt = f"""You are a stock market analysis assistant with access to specialized tools through MCP.
    
Available tools:
{tool_descriptions}

When you need to use multiple tools, you should:
1. Plan which tools you need in what order
2. Execute them one by one
3. Combine their results into a comprehensive response
4. Format the response with proper markdown and sections

Important formatting rules:
1. Always escape dollar signs in prices with a backslash, like this: \\$123.45
2. Use proper markdown formatting for all outputs
3. Make sure all numerical values are properly formatted with commas and decimals

When you need to use a tool, respond with a JSON object in the following format:
{{
    "tool": "tool_name",
    "arguments": {{
        "arg1": "value1",
        "arg2": "value2"
    }}
}}

Do not include any other text when using a tool, just the JSON object.
For regular responses, simply respond normally.
"""
    
    # Filter out system messages
    filtered_messages = [msg for msg in messages if msg["role"] != "system"]
    messages = filtered_messages.copy()
    messages.append({"role": "user", "content": prompt})
    
    all_results = []
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system_prompt}] + messages
    )
    groq_response = response.choices[0].message.content
    
    try:
        # Look for JSON pattern in the response
        import re
        json_match = re.search(r'(\{[\s\S]*\})', groq_response)
        while json_match:
            json_str = json_match.group(1)
            tool_request = json.loads(json_str)
            if "tool" in tool_request and "arguments" in tool_request:
                tool_name = tool_request["tool"]
                arguments = tool_request["arguments"]
                tool_result = await execute_tool(tool_name, arguments)
                if not isinstance(tool_result, str):
                    tool_result = str(tool_result)
                all_results.append({"tool": tool_name, "result": tool_result})
                messages.append({"role": "assistant", "content": groq_response})
                messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": system_prompt}] + messages
                )
                groq_response = response.choices[0].message.content
                json_match = re.search(r'(\{[\s\S]*\})', groq_response)
            else:
                break
        if all_results:
            messages.append({"role": "assistant", "content": groq_response})
            final_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system_prompt}] + messages
            )
            return final_response.choices[0].message.content, messages
    except Exception as e:
        st.error(f"Error processing tool request: {str(e)}")
    return groq_response, messages

def format_price(price: float) -> str:
    """Format price with escaped dollar sign to prevent LaTeX interpretation"""
    return f"\\${price:,.2f}"

def display_market_overview(market_data):
    """Display market overview in a nice format"""
    if not market_data:
        st.error("No market data available. Please try refreshing the data.")
        return
    
    try:
        # Convert CallToolResult to string if needed
        if hasattr(market_data, 'content') and isinstance(market_data.content, list):
            for content in market_data.content:
                if hasattr(content, 'text'):
                    market_data = content.text
                    break
        elif hasattr(market_data, 'value'):
            market_data = market_data.value
        
        # Ensure we have a string to parse
        if not isinstance(market_data, str):
            market_data = str(market_data)
        
        # Clean the response if needed
        market_data = market_data.strip()
        
        data = json.loads(market_data)
        
        # Check for errors
        if "error" in data:
            st.error(f"Error fetching market data: {data['error']}")
            if "details" in data:
                if isinstance(data['details'], list):
                    for detail in data['details']:
                        st.warning(detail)
                else:
                    st.warning(data['details'])
            return
        
        if "market_indices" in data:
            indices = data["market_indices"]
            
            # Display warnings if any
            if "warnings" in data and data["warnings"]:
                for warning in data["warnings"]:
                    st.warning(warning)
            
            # Display market indices
            cols = st.columns(len(indices))
            for col, (symbol, info) in zip(cols, indices.items()):
                with col:
                    st.metric(
                        label=info['name'],  # The name already includes the ETF symbol from data_fetcher.py
                        value=format_price(info['current_value']),
                        delta=f"{info['change_percent']:+.2f}%"
                    )
            
            # Display last updated time
            if "last_updated" in data:
                st.caption(f"Last updated: {data['last_updated']}")
        else:
            st.error("No market indices data available in the response")
            
    except json.JSONDecodeError as e:
        st.error(f"Error parsing market data: {str(e)}")
        st.error(f"Raw response: {market_data}")
    except Exception as e:
        st.error(f"Error displaying market overview: {str(e)}")
        st.error(f"Raw market data type: {type(market_data)}")
        if hasattr(market_data, 'content'):
            st.error(f"Content: {market_data.content}")

def search_company(query: str) -> Optional[Dict[str, Any]]:
    logger.info(f"[app.py] User submitted query: '{query}'")
    try:
        logger.info(f"[app.py] Starting company search with query: '{query}'")
        logger.info(f"[app.py] ChromaDB collection details: {st.session_state.ticker_fetcher.collection.name}")
        logger.info(f"[app.py] ChromaDB collection metadata: {st.session_state.ticker_fetcher.collection.metadata}")
        
        # Query ChromaDB
        logger.info(f"[app.py] Executing ChromaDB query: '{query}'")
        results = st.session_state.ticker_fetcher.collection.query(
            query_texts=[query],
            n_results=1
        )
        logger.info(f"[app.py] ChromaDB query results: {results}")
        
        # Log raw results
        logger.info("Raw ChromaDB results:")
        logger.info(f"Results structure: {json.dumps({k: type(v).__name__ for k, v in results.items()}, indent=2)}")
        logger.info(f"Distances: {results.get('distances', [])}")
        logger.info(f"Documents: {results.get('documents', [])}")
        logger.info(f"Metadatas: {json.dumps(results.get('metadatas', []), indent=2)}")
        logger.info(f"Ids: {results.get('ids', [])}")
        
        # Check if we got any results
        if results and results.get('metadatas') and results['metadatas'][0]:
            company_info = results['metadatas'][0][0]
            logger.info(f"[app.py] Found matching company: {company_info['name']} ({company_info['symbol']})")
            logger.info(f"[app.py] Match details:")
            logger.info(f"- Sector: {company_info.get('sector', 'N/A')}")
            logger.info(f"- Industry: {company_info.get('industry', 'N/A')}")
            logger.info(f"- Market Cap: {company_info.get('market_cap', 'N/A')}")
            
            if 'distances' in results and results['distances'][0]:
                logger.info(f"[app.py] Match distance score: {results['distances'][0][0]}")
            
            # Create a formatted info message
            company_desc = f"""Found company matching your query:
- Name: {company_info['name']}
- Symbol: {company_info['symbol']}
- Sector: {company_info['sector']}
- Industry: {company_info['industry']}
- Market Cap: {company_info['market_cap']}"""
            st.info(company_desc)
            
            return company_info
            
        logger.warning(f"[app.py] No company found matching query: '{query}'")
        logger.info("[app.py] Search returned empty results or invalid structure")
        st.warning(f"Could not find a company matching: '{query}'. Please try a different search term.")
        return None
        
    except Exception as e:
        logger.error(f"[app.py] ChromaDB search error: {str(e)}", exc_info=True)
        st.error(f"Error searching for company: {str(e)}")
        return None

def get_file_download_link(content, filename, text):
    """Generate a link to download a string file."""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href

def display_analysis_tabs(company_info=None):
    """Display persistent analysis tabs with content from session state."""
    if not st.session_state.current_analysis:
        return
    
    # Use company info from session state if not provided
    if not company_info and st.session_state.current_company_info:
        company_info = st.session_state.current_company_info
    
    if not company_info:
        return
    
    # Get data from session state
    response = st.session_state.current_analysis
    market_analysis = st.session_state.current_market_analysis
    news_analysis = st.session_state.current_news_analysis
    strategy_synthesis = st.session_state.current_strategy
    recommendations = st.session_state.current_recommendations
    
    # Create tabs for organized display
    tab_names = ["Summary", "Market Analysis", "News", "Strategy", "Charts", "Download"]
    tabs = st.tabs(tab_names)
    
    # Summary tab - brief overview
    with tabs[0]:
        st.header(f"Analysis for {company_info['name']} ({company_info['symbol']})")
        st.subheader("Company Information")
        st.markdown(f"""
        - **Sector**: {company_info['sector']}
        - **Industry**: {company_info['industry']}
        - **Market Cap**: {company_info['market_cap']}
        - **Current Price**: \\${response.get('current_price', 0):.2f}
        - **Change**: \\${response.get('price_change', 0):+.2f} ({response.get('change_percent', 0):+.2f}%)
        - **Volume**: {response.get('volume', 0):,}
        """)
        st.subheader("Key Insights")
        if recommendations and len(recommendations) > 20:
            st.markdown(recommendations[:500] + "...")
        else:
            st.markdown("No recommendations available.")
    
    # Market Analysis tab
    with tabs[1]:
        st.header("Market Analysis")
        st.markdown(market_analysis)
    
    # News Analysis tab
    with tabs[2]:
        st.header("News & Sentiment Analysis")
        st.markdown(news_analysis)
    
    # Strategy tab
    with tabs[3]:
        st.header("Investment Strategy")
        st.markdown(strategy_synthesis)
    
    # Charts tab
    with tabs[4]:
        st.header("Visualizations")
        charts = response.get('charts', {})
        if charts:
            st.success(f"Found {len(charts)} charts for {company_info['symbol']}!")
            # Display each chart
            for chart_type, chart_path in charts.items():
                chart_name = chart_type.replace("_", " ").title()
                st.subheader(chart_name)
                try:
                    st.image(chart_path, caption=f"{chart_name} for {company_info['symbol']}")
                except Exception as e:
                    st.error(f"Error loading chart: {e}")
                    logger.error(f"[display_analysis_tabs] Error loading chart {chart_path}: {e}")
        else:
            st.warning("No visualizations available for this analysis.")
            st.write("Charts would appear here when available.")
    
    # Download tab
    with tabs[5]:
        st.header("Download Analysis Report")
        
        # Create a downloadable markdown file
        markdown_content = f"""# Market Analysis Report for {company_info['name']} ({company_info['symbol']})

## Company Information
- **Sector**: {company_info['sector']}
- **Industry**: {company_info['industry']}
- **Market Cap**: {company_info['market_cap']}
- **Current Price**: ${response.get('current_price', 0):.2f}
- **Change**: ${response.get('price_change', 0):+.2f} ({response.get('change_percent', 0):+.2f}%)
- **Volume**: {response.get('volume', 0):,}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}

## Market Analysis
{market_analysis.replace('\\$', '$')}

## News & Sentiment Analysis
{news_analysis.replace('\\$', '$')}

## Investment Strategy
{strategy_synthesis.replace('\\$', '$')}

## Risk Assessment & Recommendations
{recommendations.replace('\\$', '$')}
"""
        # Create download buttons that won't trigger a full page reload
        col1, col2 = st.columns(2)
        with col1:
            download_button_key = f"md_download_{company_info['symbol']}_{datetime.now().strftime('%H%M%S')}"
            st.download_button(
                label="📥 Download Markdown Report",
                data=markdown_content,
                file_name=f"{company_info['symbol']}_analysis.md",
                mime="text/markdown",
                key=download_button_key,
                on_click=lambda: None  # Empty callback to prevent rerun
            )
        
        with col2:
            json_button_key = f"json_download_{company_info['symbol']}_{datetime.now().strftime('%H%M%S')}"
            st.download_button(
                label="📥 Download JSON Data",
                data=json.dumps(response, indent=2),
                file_name=f"{company_info['symbol']}_analysis.json",
                mime="application/json",
                key=json_button_key,
                on_click=lambda: None  # Empty callback to prevent rerun
            )
        
        st.info("The downloads include all analysis details and can be used for reference or further processing.")

def main():
    logger.info("[app.py] Starting main()")
    # Initialize session state
    initialize_session_state()
    logger.info("[app.py] Session state initialized")
    # Sidebar
    with st.sidebar:
        st.title("🤖 Market Intelligence")
        st.markdown("---")
        if st.session_state.current_symbol:
            st.info(f"Currently analyzing: {st.session_state.current_symbol}")
        period = st.selectbox(
            "Analysis Period",
            ["1mo", "3mo", "6mo", "1y"],  # Valid yfinance periods
            index=0
        )
        logger.info(f"[app.py] User selected period: {period}")
        if st.button("🔄 Refresh Data"):
            logger.info("[app.py] User clicked Refresh Data")
            st.session_state.last_refresh = datetime.now()
            st.session_state.market_data = None
            # Don't clear the crew and analysis data here
            st.rerun()
        st.markdown("---")
        st.markdown("### Analysis Tools")
        st.markdown("""
        - Market Overview
        - Technical Analysis
        - News Sentiment
        - Historical Data
        """)
        if st.button("🗑️ Clear Analysis"):
            logger.info("[app.py] User clicked Clear Analysis")
            st.session_state.messages = []
            st.session_state.current_symbol = None
            st.session_state.crew = None
            st.session_state.current_analysis = None
            st.session_state.current_market_analysis = None
            st.session_state.current_news_analysis = None
            st.session_state.current_strategy = None
            st.session_state.current_recommendations = None
            st.session_state.current_company_info = None
            st.rerun()
    
    st.title("Market Intelligence Assistant")
    
    # Display market overview
    st.markdown("### Market Overview")
    with st.container():
        if st.session_state.market_data is None:
            logger.info("[app.py] Fetching market overview via MCP tool")
            with st.spinner("Fetching market overview..."):
                result = run_async(execute_tool("get_market_overview", {}))
                logger.info(f"[app.py] Market overview result: {result}")
                st.session_state.market_data = result
        display_market_overview(st.session_state.market_data)
    
    # Display analysis tabs if we have existing analysis
    if st.session_state.current_analysis:
        st.markdown("### Analysis Results")
        display_analysis_tabs()
    
    # Chat interface
    st.markdown("### Ask Me Anything")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input handling
    if prompt := st.chat_input("Ask about any company..."):
        logger.info(f"[app.py] User submitted chat prompt: {prompt}")
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    logger.info(f"[app.py] Searching company for prompt: {prompt}")
                    company = search_company(prompt)
                    logger.info(f"[app.py] ChromaDB search result: {company}")
                    
                    if company:
                        # Store company info for persistence
                        st.session_state.current_company_info = company
                        st.session_state.current_symbol = company['symbol']
                        
                        # Process analysis with crew
                        logger.info(f"[app.py] Creating MarketAnalysisCrew for symbol={company['symbol']}, period={period}")
                        
                        # First, fetch necessary market data to avoid duplicate calls
                        logger.info(f"[app.py] Fetching stock data for {company['symbol']}")
                        stock_data_result = run_async(execute_tool("get_stock_data", {
                            "symbols": company['symbol'],
                            "interval": "1d",
                            "period": period,
                            "analysis_date": datetime.now().strftime('%Y-%m-%d')
                        }))
                        
                        # Extract and parse the actual value from CallToolResult
                        stock_data_str = None
                        if hasattr(stock_data_result, 'value'):
                            stock_data_str = stock_data_result.value
                        elif hasattr(stock_data_result, 'content') and isinstance(stock_data_result.content, list):
                            for content in stock_data_result.content:
                                if hasattr(content, 'text'):
                                    stock_data_str = content.text
                                    break
                        else:
                            stock_data_str = str(stock_data_result)
                        
                        # Parse stock data for crew initialization
                        stock_data_json = None
                        try:
                            logger.info(f"[app.py] Parsing stock data string: '{stock_data_str}'")
                            stock_data_json = json.loads(stock_data_str)
                            logger.info(f"[app.py] Successfully parsed stock data for {company['symbol']}")
                        except Exception as e:
                            logger.error(f"[app.py] Error parsing stock data: {e}, raw data: {stock_data_str}")
                            
                        # Also fetch news data to pass to the crew
                        logger.info(f"[app.py] Fetching news data for {company['symbol']}")
                        news_data_result = run_async(execute_tool("get_market_news", {
                            "symbol": company['symbol'],
                            "max_articles": 10,
                            "days": 7,
                            "analysis_date": datetime.now().strftime('%Y-%m-%d')
                        }))
                        
                        # Extract and parse the news data
                        news_data_str = None
                        if hasattr(news_data_result, 'value'):
                            news_data_str = news_data_result.value
                        elif hasattr(news_data_result, 'content') and isinstance(news_data_result.content, list):
                            for content in news_data_result.content:
                                if hasattr(content, 'text'):
                                    news_data_str = content.text
                                    break
                        else:
                            news_data_str = str(news_data_result)
                        
                        # Parse news data
                        news_data_json = None
                        try:
                            logger.info(f"[app.py] Parsing news data")
                            news_data_json = json.loads(news_data_str)
                            logger.info(f"[app.py] Successfully parsed news data for {company['symbol']}")
                        except Exception as e:
                            logger.error(f"[app.py] Error parsing news data: {e}")
                            
                        # Also get historical analysis data
                        logger.info(f"[app.py] Fetching historical analysis for {company['symbol']}")
                        hist_data_result = run_async(execute_tool("get_historical_analysis", {
                            "symbols": company['symbol'],
                            "window_size": 20,
                            "interval": "1d", 
                            "period": period,
                            "calculations": "MEAN,STDDEV,VARIANCE,CORRELATION",
                            "analysis_date": datetime.now().strftime('%Y-%m-%d')
                        }))
                        
                        # Extract and parse historical data
                        hist_data_str = None
                        if hasattr(hist_data_result, 'value'):
                            hist_data_str = hist_data_result.value
                        elif hasattr(hist_data_result, 'content') and isinstance(hist_data_result.content, list):
                            for content in hist_data_result.content:
                                if hasattr(content, 'text'):
                                    hist_data_str = content.text
                                    break
                        else:
                            hist_data_str = str(hist_data_result)
                        
                        # Parse historical data
                        hist_data_json = None
                        try:
                            logger.info(f"[app.py] Parsing historical data")
                            hist_data_json = json.loads(hist_data_str)
                            # Enhanced logging for historical data structure
                            if isinstance(hist_data_json, dict):
                                logger.info(f"[app.py] Historical data keys: {list(hist_data_json.keys())}")
                                if company['symbol'] in hist_data_json:
                                    symbol_data = hist_data_json[company['symbol']]
                                    logger.info(f"[app.py] Historical data structure for {company['symbol']}: {list(symbol_data.keys())}")
                                    
                                    # Additional logging for nested structures
                                    if "data" in symbol_data:
                                        logger.info(f"[app.py] Historical data 'data' keys: {list(symbol_data['data'].keys())}")
                                    if "metrics" in symbol_data:
                                        logger.info(f"[app.py] Historical data metrics: {list(symbol_data['metrics'].keys())}")
                            
                            logger.info(f"[app.py] Successfully parsed historical data for {company['symbol']}")
                        except Exception as e:
                            logger.error(f"[app.py] Error parsing historical data: {e}")
                            
                        # Get market overview data from session state or fetch it if not available
                        market_overview_data = None
                        if st.session_state.market_data:
                            # Extract the data from the session state
                            try:
                                if hasattr(st.session_state.market_data, 'value'):
                                    overview_str = st.session_state.market_data.value
                                elif hasattr(st.session_state.market_data, 'content') and isinstance(st.session_state.market_data.content, list):
                                    for content in st.session_state.market_data.content:
                                        if hasattr(content, 'text'):
                                            overview_str = content.text
                                            break
                                else:
                                    overview_str = str(st.session_state.market_data)
                                
                                market_overview_data = json.loads(overview_str)
                                logger.info("[app.py] Using market overview data from session state")
                            except Exception as e:
                                logger.error(f"[app.py] Error parsing market overview data from session: {e}")
                        
                        # Create crew with pre-fetched data
                        st.session_state.crew = MarketAnalysisCrew(
                            symbol=company['symbol'],
                            period=period,
                            analysis_date=datetime.now(),
                            prefetched_market_data=stock_data_json,
                            market_overview_data=market_overview_data,
                            prefetched_news_data=news_data_json,
                            prefetched_historical_data=hist_data_json
                        )
                        
                        # Run the crew
                        logger.info(f"[app.py] Running run_analysis for crew {st.session_state.crew}")
                        response = run_async(st.session_state.crew.run_analysis())
                        logger.info(f"[app.py] run_analysis response keys: {list(response.keys() if isinstance(response, dict) else [])}")
                        
                        # Debug historical data and charts
                        if isinstance(response, dict):
                            if 'charts' in response:
                                logger.info(f"[app.py] Charts in response: {response['charts']}")
                            else:
                                logger.warning("[app.py] No charts found in response")
                            
                            # Log historical data structure
                            if 'historical_data' in response:
                                logger.info(f"[app.py] Historical data structure: {json.dumps({k: type(v).__name__ for k, v in response['historical_data'].items()})}")
                            elif hist_data_json:
                                # Log structure of the hist_data_json that we passed to the crew
                                logger.info(f"[app.py] Original historical data structure: {json.dumps({k: type(v).__name__ for k, v in hist_data_json.items() if k == company['symbol']})}")
                                if company['symbol'] in hist_data_json:
                                    symbol_data = hist_data_json[company['symbol']]
                                    logger.info(f"[app.py] Historical data for {company['symbol']}: {json.dumps({k: type(v).__name__ for k, v in symbol_data.items()})}")
                            else:
                                logger.warning("[app.py] No historical data found")
                        
                        if response:
                            # Format the response text by escaping dollar signs to prevent markdown issues
                            def escape_dollar_signs(text):
                                if not text or not isinstance(text, str):
                                    return "No data available."
                                # Replace $ with \$ to escape dollar signs in markdown
                                return text.replace("$", "\\$")
                            
                            # Ensure all sections have content and format properly
                            market_analysis = escape_dollar_signs(response.get('market_analysis', 'No market analysis available.'))
                            news_analysis = escape_dollar_signs(response.get('news_analysis', 'No news analysis available.'))
                            strategy_synthesis = escape_dollar_signs(response.get('strategy_synthesis', 'No strategy synthesis available.'))
                            recommendations = escape_dollar_signs(response.get('recommendations', 'No recommendations available.'))
                            
                            # Store response and charts in session state so they persist across interactions
                            st.session_state.current_analysis = response
                            st.session_state.current_market_analysis = market_analysis
                            st.session_state.current_news_analysis = news_analysis
                            st.session_state.current_strategy = strategy_synthesis
                            st.session_state.current_recommendations = recommendations
                            
                            # Display the analysis using our dedicated function
                            display_analysis_tabs(company_info=company)
                            
                            # Construct message for chat history in a more compact form
                            formatted_response = f"""### Analysis for {company['name']} ({company['symbol']})

**Company Information**:
- Sector: {company['sector']}
- Industry: {company['industry']}
- Market Cap: {company['market_cap']}

**Market Analysis**:
{market_analysis[:500]}... *(see full analysis in tabs)*

**News & Sentiment Analysis**:
{news_analysis[:300]}... *(see full analysis in tabs)*

**Investment Strategy**:
{strategy_synthesis[:300]}... *(see full analysis in tabs)*

**Risk Assessment & Recommendations**:
{recommendations[:300]}... *(see full analysis in tabs)*
"""
                            # Add to chat history
                            st.session_state.messages.extend([
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": formatted_response}
                            ])
                        else:
                            logger.error("[app.py] Failed to generate analysis. No response from run_analysis.")
                            st.error("Failed to generate analysis. Please try again.")
                    else:
                        logger.warning("[app.py] No company found for user query.")
                        st.warning("I couldn't find a specific company matching your query. Could you please be more specific or try a different company name?")
                except Exception as e:
                    logger.error(f"[app.py] Error in chat processing: {str(e)}", exc_info=True)
                    st.error(f"An error occurred: {str(e)}")
    
    st.markdown("---")
    if st.session_state.last_refresh:
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("[app.py] End of main()")

if __name__ == "__main__":
    main() 