"""
This module implements a CrewAI-based stock market analysis system that combines
technical analysis, news sentiment, and market overview data to generate
comprehensive investment insights.
"""
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
import json
from datetime import datetime, timedelta
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import sys

# Import visualization module
try:
    from business_intelligence_agent.visualization import generate_all_charts
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization module not available. Charts will not be generated.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to your MCP server
MCP_SERVER_PATH = Path(__file__).parent / "data_fetcher.py"

def load_agents_config() -> dict:
    """Load agent configurations from YAML file."""
    try:
        config_path = Path(__file__).parent / "config" / "agents.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading agents configuration: {e}")
        raise

def load_tasks_config() -> dict:
    """Load task configurations from YAML file."""
    try:
        config_path = Path(__file__).parent / "config" / "tasks.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading tasks configuration: {e}")
        raise

@CrewBase
class MarketAnalysisCrew:
    """Crew for comprehensive stock market analysis"""
    
    def __init__(self, symbol: str = "AAPL", period: str = "1mo", 
                 analysis_date: Optional[datetime] = None,
                 prefetched_market_data: Optional[Dict] = None,
                 market_overview_data: Optional[Dict] = None,
                 prefetched_news_data: Optional[Dict] = None,
                 prefetched_historical_data: Optional[Dict] = None):
        """
        Initialize the market analysis crew.
        
        Args:
            symbol: Stock symbol to analyze (default: "AAPL")
            period: Analysis period (default: "1mo")
            analysis_date: Date to analyze from (default: today)
            prefetched_market_data: Pre-fetched stock data (to avoid duplicate API calls)
            market_overview_data: Pre-fetched market overview data
            prefetched_news_data: Pre-fetched news articles data
            prefetched_historical_data: Pre-fetched historical analysis data
        """
        load_dotenv()
        self.symbol = symbol
        self.period = period
        self.analysis_date = analysis_date or datetime.now()
        
        # Initialize market data variables
        self.current_price = 0.0
        self.previous_close = 0.0
        self.price_change = 0.0
        self.change_percent = 0.0
        self.volume = 0
        self.last_trading_day = None
        
        # Initialize data containers
        self.market_data = {}
        self.news_data = {}
        self.market_overview = {}
        self.historical_data = {}
        
        self.agents_config = load_agents_config()
        self.tasks_config = load_tasks_config()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Flags to track what data is prefetched
        self.has_prefetched_market_data = prefetched_market_data is not None
        self.has_prefetched_news_data = prefetched_news_data is not None
        self.has_prefetched_historical_data = prefetched_historical_data is not None
        self.has_prefetched_overview = market_overview_data is not None
        
        # Process prefetched data if available
        if prefetched_market_data and self.symbol in prefetched_market_data:
            logger.info(f"[MarketAnalysisCrew] Using prefetched market data for {self.symbol}")
            self._process_stock_data(prefetched_market_data)
        
        if prefetched_historical_data and self.symbol in prefetched_historical_data:
            logger.info(f"[MarketAnalysisCrew] Using prefetched historical data for {self.symbol}")
            self.historical_data = prefetched_historical_data
            # Add to market_data for consistency with fetch_market_data
            if self.symbol in prefetched_historical_data:
                self.market_data["historical_analysis"] = prefetched_historical_data[self.symbol]
        
        if prefetched_news_data:
            logger.info(f"[MarketAnalysisCrew] Using prefetched news data for {self.symbol}")
            self.news_data = prefetched_news_data
            
        if market_overview_data:
            logger.info(f"[MarketAnalysisCrew] Using prefetched market overview data")
            self.market_overview = market_overview_data
        
        # Fetch any missing data
        need_to_fetch = not (self.has_prefetched_market_data and 
                            self.has_prefetched_news_data and 
                            self.has_prefetched_historical_data and 
                            self.has_prefetched_overview)
                            
        if need_to_fetch:
            logger.info(f"[MarketAnalysisCrew] Some data not prefetched, fetching remaining data for {self.symbol}")
            asyncio.run(self.fetch_market_data())
        else:
            logger.info(f"[MarketAnalysisCrew] All data prefetched, skipping fetch_market_data")
        
        self._setup_agents()
        self._setup_tasks()
        self._setup_llm()

    def _setup_llm(self):
        """Initialize the LLM with Groq configuration"""
        try:
            self.llm = LLM(
                model="groq/llama-3.3-70b-versatile",
            )
            self.llm_model = "groq/llama-3.3-70b-versatile"
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            raise

    async def _execute_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Execute a tool through the MCP server"""
        server_params = StdioServerParameters(
            command="python",
            args=[str(MCP_SERVER_PATH)],
        )
        logger.info(f"[_execute_mcp_tool] Invoking MCP tool: {tool_name} with arguments: {arguments}")
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info(f"[_execute_mcp_tool] Initialized MCP session for tool: {tool_name}")
                    result = await session.call_tool(tool_name, arguments)
                    logger.info(f"[_execute_mcp_tool] Raw result from MCP tool {tool_name}: {result}")
                    # Debug logging
                    logger.debug(f"[_execute_mcp_tool] Result type: {type(result)}")
                    logger.debug(f"[_execute_mcp_tool] Result content: {result}")
                    # Extract text content from MCP response
                    if hasattr(result, 'content') and isinstance(result.content, list):
                        for content in result.content:
                            if hasattr(content, 'text'):
                                logger.info(f"[_execute_mcp_tool] Extracted text content from MCP result for {tool_name}")
                                return content.text
                    if hasattr(result, 'value'):
                        logger.info(f"[_execute_mcp_tool] Extracted value from MCP result for {tool_name}")
                        return result.value
                    logger.info(f"[_execute_mcp_tool] Returning stringified result for {tool_name}")
                    return str(result)
        except Exception as e:
            logger.error(f"[_execute_mcp_tool] Error executing MCP tool {tool_name}: {e}", exc_info=True)
            raise

    def _setup_agents(self):
        """Initialize agents from configuration"""
        try:
            with open(Path(__file__).parent / "config" / "agents.yaml", "r") as f:
                self.agents_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading agents configuration: {e}")
            raise
    
    def _setup_tasks(self):
        """Initialize tasks from configuration"""
        try:
            with open(Path(__file__).parent / "config" / "tasks.yaml", "r") as f:
                self.tasks_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading tasks configuration: {e}")
            raise
    
    def _log_agent_start(self, agent_name):
        logger.info(f"[CrewAI] Agent '{agent_name}' started.")
    def _log_agent_end(self, agent_name):
        logger.info(f"[CrewAI] Agent '{agent_name}' finished.")
    def _log_task_start(self, task_name):
        logger.info(f"[CrewAI] Task '{task_name}' started.")
    def _log_task_end(self, task_name):
        logger.info(f"[CrewAI] Task '{task_name}' finished.")

    @agent
    def market_analyst(self) -> Agent:
        self._log_agent_start('market_analyst')
        config = self.agents_config['market_analyst']
        agent_obj = Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.llm,
            verbose=config.get('verbose', True)
        )
        self._log_agent_end('market_analyst')
        return agent_obj

    @agent
    def news_analyst(self) -> Agent:
        self._log_agent_start('news_analyst')
        config = self.agents_config['news_analyst']
        agent_obj = Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.llm,
            verbose=config.get('verbose', True)
        )
        self._log_agent_end('news_analyst')
        return agent_obj

    @agent
    def strategy_synthesizer(self) -> Agent:
        self._log_agent_start('strategy_synthesizer')
        config = self.agents_config['strategy_synthesizer']
        agent_obj = Agent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            llm=self.llm,
            verbose=config.get('verbose', True)
        )
        self._log_agent_end('strategy_synthesizer')
        return agent_obj
    
    @task
    def market_analysis_task(self) -> Task:
        self._log_task_start('market_analysis_task')
        config = self.tasks_config['market_analysis_task']
        task_obj = Task(
            description=config['description'].format(
                symbol=self.symbol,
                period=self.period,
                analysis_date=self.analysis_date.strftime('%Y-%m-%d'),
                current_price=self.current_price,
                previous_close=self.previous_close,
                change=self.price_change,
                change_percent=self.change_percent,
                volume=self.volume
            ),
            expected_output=config['expected_output'],
            agent=self.market_analyst(),
            context=config.get('context', []),
            config=config.get('config', {})
        )
        self._log_task_end('market_analysis_task')
        return task_obj

    @task
    def news_analysis_task(self) -> Task:
        self._log_task_start('news_analysis_task')
        config = self.tasks_config['news_analysis_task']
        task_obj = Task(
            description=config['description'].format(
                symbol=self.symbol,
                analysis_date=self.analysis_date.strftime('%Y-%m-%d')
            ),
            expected_output=config['expected_output'],
            agent=self.news_analyst(),
            context=config.get('context', []),
            config=config.get('config', {})
        )
        self._log_task_end('news_analysis_task')
        return task_obj

    @task
    def strategy_synthesis_task(self) -> Task:
        self._log_task_start('strategy_synthesis_task')
        config = self.tasks_config['strategy_synthesis_task']
        
        task_obj = Task(
            description=config['description'].format(
                symbol=self.symbol,
                analysis_date=self.analysis_date.strftime('%Y-%m-%d')
            ),
            expected_output=config['expected_output'],
            agent=self.strategy_synthesizer(),
            context=[],  # Use empty context to avoid errors
            config=config.get('config', {}),  # Add back the config parameter
            output_file=config.get('output_file', 'output/market_research.md')
        )
        self._log_task_end('strategy_synthesis_task')
        return task_obj

    @crew
    def crew(self) -> Crew:
        """Create and configure the crew"""
        return Crew(
            agents=[
                self.market_analyst(),
                self.news_analyst(),
                self.strategy_synthesizer()
            ],
            tasks=[
                self.market_analysis_task(),
                self.news_analysis_task(),
                self.strategy_synthesis_task()
            ],
            process=Process.sequential,
            verbose=True,
            share_crew_history=True  # Ensure tasks can access outputs from previous tasks
        )
    
    async def fetch_market_data(self):
        """Fetch all required market data using MCP tools"""
        try:
            # Skip fetching stock data if we have prefetched data (it was already processed in __init__)
            if not self.has_prefetched_market_data:
                logger.info(f"[fetch_market_data] Fetching stock data for symbol: {self.symbol}, period: {self.period}, analysis_date: {self.analysis_date}")
                # Fetch current stock data
                stock_data_args = {
                    "symbols": self.symbol,
                    "interval": "1d",
                    "period": self.period,
                    "analysis_date": self.analysis_date.strftime('%Y-%m-%d') if self.analysis_date else None
                }
                logger.info(f"[fetch_market_data] Calling get_stock_data with args: {stock_data_args}")
                stock_data_response = await self._execute_mcp_tool(
                    "get_stock_data",
                    stock_data_args
                )
                logger.info(f"[fetch_market_data] get_stock_data response: {stock_data_response}")
                # Ensure we have a valid string to parse
                if not isinstance(stock_data_response, str):
                    stock_data_response = str(stock_data_response)
                stock_data_response = stock_data_response.strip()
                try:
                    stock_data = json.loads(stock_data_response)
                    logger.info(f"[fetch_market_data] Parsed stock_data: {stock_data}")
                    # Process the stock data
                    self._process_stock_data(stock_data)
                except json.JSONDecodeError as e:
                    logger.error(f"[fetch_market_data] Failed to parse stock data response: {e}")
                    logger.error(f"[fetch_market_data] Raw response: {stock_data_response}")
                    raise ValueError(f"Invalid JSON response from get_stock_data: {e}")
            else:
                logger.info(f"[fetch_market_data] Using prefetched market data for {self.symbol}")
            
            # Fetch historical analysis if not prefetched
            if not self.has_prefetched_historical_data:
                try:
                    historical_data_args = {
                        "symbols": self.symbol,
                        "window_size": 20,
                        "interval": "1d",
                        "period": self.period,
                        "calculations": "MEAN,STDDEV,VARIANCE,CORRELATION",
                        "analysis_date": self.analysis_date.strftime('%Y-%m-%d') if self.analysis_date else None
                    }
                    logger.info(f"[fetch_market_data] Calling get_historical_analysis with args: {historical_data_args}")
                    historical_data_response = await self._execute_mcp_tool(
                        "get_historical_analysis",
                        historical_data_args
                    )
                    logger.info(f"[fetch_market_data] get_historical_analysis response: {historical_data_response}")

                    try:
                        # Make sure we have a string to parse
                        if not isinstance(historical_data_response, str):
                            historical_data_response = str(historical_data_response)
                        historical_data_response = historical_data_response.strip()
                        
                        # Verify it's valid JSON before parsing
                        if not (historical_data_response.startswith('{') or historical_data_response.startswith('[')):
                            logger.warning(f"[fetch_market_data] Invalid JSON format in historical_data_response, skipping")
                            logger.debug(f"[fetch_market_data] Raw response: {historical_data_response[:100]}")
                        else:
                            # Parse the JSON
                            historical_data = json.loads(historical_data_response)
                            logger.info(f"[fetch_market_data] Parsed historical_data type: {type(historical_data)}")
                            
                            # Store the response but handle different formats
                            if isinstance(historical_data, dict):
                                self.historical_data = historical_data
                                
                                # Check if this is the structure we expect
                                if self.symbol in historical_data:
                                    symbol_hist = historical_data[self.symbol]
                                    logger.info(f"[fetch_market_data] Found symbol {self.symbol} in historical data")
                                    
                                    if isinstance(symbol_hist, dict) and "error" not in symbol_hist:
                                        self.market_data["historical_analysis"] = symbol_hist
                                        logger.info(f"[fetch_market_data] Successfully fetched historical analysis for {self.symbol}")
                                        
                                        # Verify metrics structure
                                        if "metrics" in symbol_hist and isinstance(symbol_hist["metrics"], dict):
                                            metrics = symbol_hist["metrics"]
                                            logger.info(f"[fetch_market_data] Historical analysis metrics: {list(metrics.keys())}")
                                        else:
                                            logger.warning(f"[fetch_market_data] No valid metrics found in historical data")
                                    else:
                                        error_msg = symbol_hist.get('error', 'Unknown error') if isinstance(symbol_hist, dict) else "Invalid data format"
                                        logger.warning(f"[fetch_market_data] Could not use historical analysis: {error_msg}")
                                else:
                                    logger.warning(f"[fetch_market_data] Symbol {self.symbol} not found in historical data")
                            else:
                                logger.warning(f"[fetch_market_data] Unexpected historical data format: {type(historical_data)}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"[fetch_market_data] Failed to parse historical analysis response: {e}")
                        logger.warning(f"[fetch_market_data] Raw historical analysis response (first 100 chars): '{historical_data_response[:100]}'")
                except Exception as e:
                    logger.warning(f"[fetch_market_data] Error processing historical analysis: {str(e)}")
                    logger.warning(f"[fetch_market_data] Historical analysis will be skipped", exc_info=True)
            else:
                logger.info(f"[fetch_market_data] Using prefetched historical data for {self.symbol}")
                
            # Fetch news data if not prefetched
            if not self.has_prefetched_news_data:
                try:
                    news_args = {
                        "symbol": self.symbol,
                        "max_articles": 10,
                        "days": 7,
                        "analysis_date": self.analysis_date.strftime('%Y-%m-%d') if self.analysis_date else None
                    }
                    logger.info(f"[fetch_market_data] Calling get_market_news with args: {news_args}")
                    news_response = await self._execute_mcp_tool(
                        "get_market_news",
                        news_args
                    )
                    logger.info(f"[fetch_market_data] get_market_news response: {news_response}")
                    news_data = json.loads(news_response)
                    if "error" not in news_data and "articles" in news_data:
                        self.news_data = news_data
                        num_articles = len(news_data["articles"])
                        logger.info(f"[fetch_market_data] Successfully fetched {num_articles} recent news articles for {self.symbol}")
                    else:
                        logger.warning(f"[fetch_market_data] Could not fetch news data: {news_data.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"[fetch_market_data] Error fetching news data: {e}")
            else:
                logger.info(f"[fetch_market_data] Using prefetched news data for {self.symbol}")
                
            # Fetch market overview if not prefetched
            if not self.has_prefetched_overview:
                try:
                    overview_args = {
                        "analysis_date": self.analysis_date.strftime('%Y-%m-%d') if self.analysis_date else None
                    }
                    logger.info(f"[fetch_market_data] Calling get_market_overview with args: {overview_args}")
                    overview_response = await self._execute_mcp_tool(
                        "get_market_overview",
                        overview_args
                    )
                    logger.info(f"[fetch_market_data] get_market_overview response: {overview_response}")
                    overview_data = json.loads(overview_response)
                    if "error" not in overview_data:
                        self.market_overview = overview_data
                        logger.info("[fetch_market_data] Successfully fetched current market overview")
                    else:
                        logger.warning(f"[fetch_market_data] Could not fetch market overview: {overview_data['error']}")
                except Exception as e:
                    logger.error(f"[fetch_market_data] Error fetching market overview: {e}")
            else:
                logger.info("[fetch_market_data] Using prefetched market overview data")
                
            # Verify we have necessary market data
            if not self.market_data or not self.market_data.get("current_price"):
                logger.error("[fetch_market_data] Failed to fetch minimum required market data")
                raise ValueError("Failed to fetch minimum required market data")
                
            # Log the data we have collected
            logger.info(f"[fetch_market_data] Data collection complete for {self.symbol}")
            logger.info(f"[fetch_market_data] - Market data: {len(self.market_data)} fields")
            logger.info(f"[fetch_market_data] - News data: {len(self.news_data.get('articles', [])) if isinstance(self.news_data, dict) else 0} articles")
            logger.info(f"[fetch_market_data] - Historical data: {bool(self.historical_data)}")
            logger.info(f"[fetch_market_data] - Market overview: {bool(self.market_overview)}")
            
        except Exception as e:
            logger.error(f"[fetch_market_data] Error fetching market data: {str(e)}", exc_info=True)
            raise
    
    async def run_analysis(self) -> Dict[str, Any]:
        """Run the complete market analysis"""
        try:
            logger.info("[run_analysis] Running crew analysis")
            
            # Generate visualization charts before starting the analysis
            charts = {}
            logger.info(f"[run_analysis] Market data keys: {list(self.market_data.keys())}")
            logger.info(f"[run_analysis] Historical data keys: {list(self.historical_data.keys())}")
            
            # For visualization, we need to make sure historical data is structured properly
            historical_data_for_charts = {}
            
            # First, check if we have historical data in the expected format
            if self.symbol in self.historical_data:
                logger.info(f"[run_analysis] Found {self.symbol} in historical_data")
                historical_data_for_charts = self.historical_data[self.symbol]
                # Log the structure of this data
                if isinstance(historical_data_for_charts, dict):
                    logger.info(f"[run_analysis] Historical data structure for {self.symbol}: {list(historical_data_for_charts.keys())}")
                    # Check for critical data sections
                    if "metrics" in historical_data_for_charts:
                        logger.info(f"[run_analysis] Metrics available: {list(historical_data_for_charts['metrics'].keys())}")
                    else:
                        logger.warning(f"[run_analysis] No metrics found in historical data for {self.symbol}")
            elif "historical_analysis" in self.market_data:
                logger.info(f"[run_analysis] Found historical_analysis in market_data")
                historical_data_for_charts = self.market_data["historical_analysis"]
                logger.info(f"[run_analysis] Historical analysis keys: {list(historical_data_for_charts.keys())}")
            
            # Try to generate charts if we have any data
            if self.market_data or historical_data_for_charts:
                try:
                    logger.info(f"[run_analysis] Calling generate_all_charts for {self.symbol}")
                    # Ensure we're passing a dict with the symbol's data to visualization
                    charts = generate_all_charts(self.symbol, self.market_data, historical_data_for_charts)
                    logger.info(f"[run_analysis] Generated {len(charts)} charts: {list(charts.keys())}")
                except Exception as e:
                    logger.error(f"[run_analysis] Error generating charts: {e}", exc_info=True)
            else:
                logger.warning(f"[run_analysis] No data available for visualization for {self.symbol}")
            
            crew_output = self.crew().kickoff()
            logger.info(f"[run_analysis] Raw result from crew().kickoff(): {crew_output}")
            
            # Extract results from CrewOutput 
            task_results = {}
            market_analysis_result = ""
            news_analysis_result = ""
            strategy_synthesis_result = ""
            
            # Handle different CrewOutput formats across CrewAI versions
            if hasattr(crew_output, 'tasks_outputs'):
                # Newer CrewAI format
                logger.info(f"[run_analysis] Found tasks_outputs attribute: {crew_output.tasks_outputs}")
                for task_output in crew_output.tasks_outputs:
                    task_name = task_output.task.name if hasattr(task_output.task, 'name') else "unknown"
                    task_results[task_name] = task_output.output
                    logger.info(f"[run_analysis] Task {task_name} output: {task_output.output}")
                    
                    # Map outputs to correct fields
                    if "market_analysis" in task_name.lower():
                        market_analysis_result = task_output.output
                    elif "news_analysis" in task_name.lower():
                        news_analysis_result = task_output.output
                    elif "strategy" in task_name.lower():
                        strategy_synthesis_result = task_output.output
            
            elif hasattr(crew_output, 'result'):
                # Direct result field in CrewOutput
                logger.info(f"[run_analysis] Found result attribute: {crew_output.result}")
                result = crew_output.result
                
                # Try to parse JSON if it's a string
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, dict):
                            task_results = parsed
                        else:
                            task_results["combined_result"] = result
                    except json.JSONDecodeError:
                        # Not JSON, treat as combined result
                        task_results["combined_result"] = result
                elif isinstance(result, dict):
                    task_results = result
                else:
                    task_results["combined_result"] = str(result)
            
            else:
                # Legacy or different format - treat as direct result
                logger.info(f"[run_analysis] No recognized output format, using raw output: {crew_output}")
                if isinstance(crew_output, dict):
                    task_results = crew_output
                else:
                    # Fallback: treat as string 
                    task_results["combined_result"] = str(crew_output)
            
            # Build structured analysis report
            analysis_report = {
                "symbol": self.symbol,
                "period": self.period,
                "analysis_date": self.analysis_date.strftime('%Y-%m-%d'),
                "current_price": self.current_price,
                "previous_close": self.previous_close,
                "price_change": self.price_change,
                "change_percent": self.change_percent,
                "volume": self.volume,
                "last_trading_day": self.last_trading_day,
                "timestamp": datetime.now().isoformat(),
                "charts": charts,  # Add generated charts to the report
                "historical_data": historical_data_for_charts  # Include historical data in the report
            }
            
            # Extract information correctly from task_results
            if "market_analysis_task" in task_results:
                analysis_report["market_analysis"] = task_results["market_analysis_task"]
            elif "market_analysis" in task_results:
                analysis_report["market_analysis"] = task_results["market_analysis"]
            elif market_analysis_result:
                analysis_report["market_analysis"] = market_analysis_result
            else:
                analysis_report["market_analysis"] = task_results.get("combined_result", "No market analysis available")
            
            # For news analysis, use the task output or create a summary from raw news data
            has_news_analysis = False
            if "news_analysis_task" in task_results:
                analysis_report["news_analysis"] = task_results["news_analysis_task"]
                has_news_analysis = True
            elif "news_analysis" in task_results:
                analysis_report["news_analysis"] = task_results["news_analysis"]
                has_news_analysis = True
            elif news_analysis_result:
                analysis_report["news_analysis"] = news_analysis_result
                has_news_analysis = True
            elif not has_news_analysis and isinstance(self.news_data, dict) and "articles" in self.news_data:
                # If no news analysis from agents but we have news data, create a simple summary
                articles = self.news_data["articles"]
                if articles:
                    news_summary = f"Recent news for {self.symbol} includes {len(articles)} articles:\n\n"
                    for i, article in enumerate(articles, 1):  # Show all articles instead of just top 5
                        sentiment = article.get("sentiment", {})
                        news_summary += f"{i}. {article.get('title', 'No title')} ({sentiment.get('label', 'neutral')} sentiment)\n"
                        news_summary += f"   Source: {article.get('source', 'Unknown')}\n"
                        news_summary += f"   Summary: {article.get('summary', 'No summary available')}\n\n"
                    news_summary += "Overall sentiment from news articles appears to be " + self._determine_overall_sentiment(articles)
                    analysis_report["news_analysis"] = news_summary
                    has_news_analysis = True
                
            if not has_news_analysis:
                analysis_report["news_analysis"] = "No news analysis available"
            
            # Extract strategy synthesis
            if "strategy_synthesis_task" in task_results:
                analysis_report["strategy_synthesis"] = task_results["strategy_synthesis_task"]
                logger.info(f"[run_analysis] Using strategy from strategy_synthesis_task")
            elif "strategy_synthesis" in task_results:
                analysis_report["strategy_synthesis"] = task_results["strategy_synthesis"]
                logger.info(f"[run_analysis] Using strategy from strategy_synthesis key")
            elif strategy_synthesis_result:
                analysis_report["strategy_synthesis"] = strategy_synthesis_result
                logger.info(f"[run_analysis] Using strategy from strategy_synthesis_result")
            else:
                # Create a more specific fallback strategy based on available data
                logger.info(f"[run_analysis] No strategy synthesis found in results, creating detailed fallback")
                
                # Find the available historical data for metrics
                metrics = {}
                if self.historical_data and self.symbol in self.historical_data:
                    symbol_data = self.historical_data[self.symbol]
                    if "metrics" in symbol_data:
                        metrics = symbol_data["metrics"]
                        logger.info(f"[run_analysis] Using metrics from historical_data: {list(metrics.keys())}")
                elif "historical_analysis" in self.market_data and "metrics" in self.market_data["historical_analysis"]:
                    metrics = self.market_data["historical_analysis"]["metrics"]
                    logger.info(f"[run_analysis] Using metrics from market_data.historical_analysis: {list(metrics.keys())}")
                
                # Create a more detailed strategy with whatever metrics we have
                strategy = f"""## Investment Thesis for {self.symbol}
Based on technical analysis and market data, {self.symbol} is currently trading at ${self.current_price:.2f} with a {self.price_change:+.2f} ({self.change_percent:+.2f}%) change from previous close. Trading volume is {self.volume:,}.

## Technical Analysis Summary
"""
                # Add technical indicators if available
                if "MEAN" in metrics and "STDDEV" in metrics:
                    mean = metrics["MEAN"]
                    stddev = metrics["STDDEV"]
                    
                    # Determine position relative to mean
                    if self.current_price > mean + stddev:
                        strategy += f"The current price (${self.current_price:.2f}) is trading above the mean (${mean:.2f}) by more than one standard deviation (${stddev:.2f}), suggesting potential overbought conditions.\n\n"
                    elif self.current_price < mean - stddev:
                        strategy += f"The current price (${self.current_price:.2f}) is trading below the mean (${mean:.2f}) by more than one standard deviation (${stddev:.2f}), suggesting potential oversold conditions.\n\n"
                    else:
                        strategy += f"The current price (${self.current_price:.2f}) is trading within one standard deviation (${stddev:.2f}) of the mean (${mean:.2f}), suggesting a potentially range-bound market.\n\n"
                else:
                    strategy += f"Limited technical indicators are available for comprehensive analysis.\n\n"
                
                # Include moving averages if available
                if "sma_20" in metrics and "sma_50" in metrics and metrics["sma_20"] is not None and metrics["sma_50"] is not None:
                    sma_20 = metrics["sma_20"] 
                    sma_50 = metrics["sma_50"]
                    
                    if sma_20 > sma_50:
                        strategy += f"The 20-day SMA (${sma_20:.2f}) is above the 50-day SMA (${sma_50:.2f}), suggesting a potential bullish trend.\n\n"
                    else:
                        strategy += f"The 20-day SMA (${sma_20:.2f}) is below the 50-day SMA (${sma_50:.2f}), suggesting a potential bearish trend.\n\n"
                
                # Add volatility metrics if available
                if "STDDEV" in metrics and "MEAN" in metrics:
                    volatility = metrics["STDDEV"] / metrics["MEAN"] * 100
                    if volatility > 5:
                        strategy += f"Volatility is high at {volatility:.2f}% of mean price, suggesting increased risk and potential for larger price swings.\n\n"
                    elif volatility > 2:
                        strategy += f"Volatility is moderate at {volatility:.2f}% of mean price, suggesting normal market conditions.\n\n"
                    else:
                        strategy += f"Volatility is low at {volatility:.2f}% of mean price, suggesting stable trading conditions.\n\n"
                
                # Add price-volume analysis if available
                if "price_volume_correlation" in metrics:
                    corr = metrics["price_volume_correlation"]
                    if corr > 0.5:
                        strategy += f"Strong positive correlation ({corr:.2f}) between price and volume suggests buying strength when price increases.\n\n"
                    elif corr < -0.5:
                        strategy += f"Strong negative correlation ({corr:.2f}) between price and volume suggests potential distribution patterns.\n\n"
                    else:
                        strategy += f"Weak correlation ({corr:.2f}) between price and volume shows no clear relationship between buying/selling pressure and price movement.\n\n"
                
                # Add risk assessment
                strategy += f"""## Risk Assessment
- Market volatility could impact {self.symbol}'s performance unexpectedly
- Technical indicators may not account for unexpected news events
- Price at ${self.current_price:.2f} shows {self.price_change:+.2f} ({self.change_percent:+.2f}%) change, indicating {'positive' if self.price_change > 0 else 'negative'} recent momentum
- Recent trading volume of {self.volume:,} is a factor to consider for liquidity risk
"""

                # Add basic recommendation based on price action and technical indicators
                recommendation = "HOLD"
                reason = "based on mixed technical signals"
                
                if "sma_20" in metrics and "sma_50" in metrics and metrics["sma_20"] is not None and metrics["sma_50"] is not None:
                    if metrics["sma_20"] > metrics["sma_50"] and self.price_change > 0:
                        recommendation = "BUY"
                        reason = "based on positive technical indicators and price momentum"
                    elif metrics["sma_20"] < metrics["sma_50"] and self.price_change < 0:
                        recommendation = "SELL"
                        reason = "based on negative technical indicators and price momentum"
                
                strategy += f"""## Strategy Recommendation
{recommendation} - {reason}. Further analysis of fundamentals and market context is recommended.

## Price Targets
- Entry point: ${self.current_price:.2f} to ${(self.current_price * 1.02):.2f}
- Stop-loss: ${(self.current_price * 0.95):.2f} (5% below current price)
- Profit target: ${(self.current_price * 1.10):.2f} (10% above current price)
- Risk/reward ratio: 1:2

## Timeframe
Medium-term outlook (1-3 months) is recommended to allow technical patterns to develop fully. Monitor key support/resistance levels and be prepared to adjust strategy based on emerging patterns.
"""
                analysis_report["strategy_synthesis"] = strategy
                logger.info(f"[run_analysis] Created detailed fallback strategy ({len(strategy)} chars)")
            
            # Default empty recommendations if not found
            if "recommendations" in task_results:
                analysis_report["recommendations"] = task_results["recommendations"]
            else:
                # Create a simple recommendation based on price movement
                if self.price_change > 0:
                    analysis_report["recommendations"] = f"{self.symbol} has shown positive price movement recently. Consider further research on fundamentals before making investment decisions."
                elif self.price_change < 0:
                    analysis_report["recommendations"] = f"{self.symbol} has shown negative price movement recently. Monitor for potential entry points but exercise caution."
                else:
                    analysis_report["recommendations"] = f"No clear price trend for {self.symbol}. Additional research recommended before making investment decisions."
            
            logger.info(f"[run_analysis] Final analysis_report structure: {list(analysis_report.keys())}")
            for key, value in analysis_report.items():
                if key in ["market_analysis", "news_analysis", "strategy_synthesis", "recommendations"]:
                    value_type = type(value).__name__
                    value_length = len(str(value)) if value else 0
                    logger.info(f"[run_analysis] {key}: {value_type}, length: {value_length}")
            
            # Create the output file properly using with statement
            output_file = self.output_dir / "market_research.md"
            try:
                with open(output_file, 'w') as f:
                    f.write(f"# Market Analysis Report for {self.symbol}\n\n")
                    f.write(f"Analysis Period: {self.period} (as of {self.analysis_date.strftime('%Y-%m-%d')})\n")
                    f.write(f"Current Price: ${self.current_price:.2f}\n")
                    f.write(f"Change: ${self.price_change:+.2f} ({self.change_percent:+.2f}%)\n")
                    f.write(f"Volume: {self.volume:,}\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                    
                    # Add chart references if available
                    if charts:
                        f.write("## Charts\n\n")
                        for chart_type, chart_path in charts.items():
                            chart_name = chart_type.replace("_", " ").title()
                            f.write(f"### {chart_name}\n")
                            f.write(f"![{chart_name}]({chart_path})\n\n")
                    
                    f.write("## Analysis Results\n\n")
                    f.write(json.dumps(analysis_report, indent=2))
                logger.info(f"[run_analysis] Analysis complete. Results saved to {output_file}")
            except Exception as e:
                logger.error(f"[run_analysis] Error writing output file: {e}", exc_info=True)
                
            return analysis_report
        except Exception as e:
            logger.error(f"[run_analysis] Error running analysis: {e}", exc_info=True)
            return {
                "symbol": self.symbol,
                "error": str(e),
                "market_analysis": "Error occurred during analysis",
                "news_analysis": "Error occurred during analysis",
                "strategy_synthesis": "Error occurred during analysis",
                "recommendations": "Error occurred during analysis"
            }
            
    def _determine_overall_sentiment(self, articles: list) -> str:
        """Calculate the overall sentiment from a list of news articles"""
        if not articles:
            return "neutral"
            
        sentiment_scores = []
        for article in articles:
            sentiment = article.get("sentiment", {})
            if isinstance(sentiment, dict) and "score" in sentiment:
                try:
                    score = float(sentiment["score"])
                    sentiment_scores.append(score)
                except (ValueError, TypeError):
                    pass
        
        if not sentiment_scores:
            return "neutral"
            
        avg_score = sum(sentiment_scores) / len(sentiment_scores)
        if avg_score > 0.1:
            return "positive"
        elif avg_score < -0.1:
            return "negative"
        else:
            return "neutral"

    def _process_stock_data(self, stock_data: Dict):
        """Process stock data whether prefetched or freshly fetched"""
        # Validate stock_data is a dictionary
        if not isinstance(stock_data, dict):
            logger.error(f"[_process_stock_data] Invalid stock data format: expected dict, got {type(stock_data)}")
            return
            
        # Check if symbol exists in the data
        if self.symbol not in stock_data:
            logger.error(f"[_process_stock_data] No data found for symbol: {self.symbol}")
            return
            
        # Get symbol data
        symbol_data = stock_data[self.symbol]
        
        # Check for error in symbol data
        if "error" in symbol_data:
            logger.error(f"[_process_stock_data] Error in stock data for {self.symbol}: {symbol_data['error']}")
            return
        
        # Ensure we're dealing with a dictionary    
        if not isinstance(symbol_data, dict):
            logger.error(f"[_process_stock_data] Symbol data is not a dict: {type(symbol_data)}")
            return
            
        # Process data
        self.market_data = symbol_data
        
        # Extract key metrics with safe fallbacks
        try:
            self.current_price = float(symbol_data.get("current_price", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"[_process_stock_data] Invalid current_price: {symbol_data.get('current_price')}, using 0.0")
            self.current_price = 0.0
            
        try:
            self.previous_close = float(symbol_data.get("previous_close", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"[_process_stock_data] Invalid previous_close: {symbol_data.get('previous_close')}, using 0.0")
            self.previous_close = 0.0
            
        try:
            self.price_change = float(symbol_data.get("change", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"[_process_stock_data] Invalid change: {symbol_data.get('change')}, using 0.0")
            self.price_change = 0.0
            
        try:
            self.change_percent = float(symbol_data.get("change_percent", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"[_process_stock_data] Invalid change_percent: {symbol_data.get('change_percent')}, using 0.0")
            self.change_percent = 0.0
            
        try:
            self.volume = int(symbol_data.get("volume", 0))
        except (ValueError, TypeError):
            logger.warning(f"[_process_stock_data] Invalid volume: {symbol_data.get('volume')}, using 0")
            self.volume = 0
            
        self.last_trading_day = symbol_data.get("last_trading_day", None)
        
        # Log the source of the data
        source = "prefetched" if self.has_prefetched_market_data else "freshly fetched"
        logger.info(f"[_process_stock_data] Successfully processed {source} stock data for {self.symbol} (price: ${self.current_price:.2f})")

async def main():
    """Main entry point for running market analysis"""
    crew = None
    try:
        # Fetch market data ahead of creating the crew to avoid duplicate calls
        logger.info("Fetching market data for AAPL before creating crew")
        stock_data_response = None
        market_overview_response = None
        
        # Set up MCP server parameters
        server_params = StdioServerParameters(
            command="python",
            args=[str(MCP_SERVER_PATH)],
        )
        
        # Fetch stock data
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    stock_data_response = await session.call_tool("get_stock_data", {
                        "symbols": "AAPL",
                        "interval": "1d",
                        "period": "1mo"
                    })
                    
                    # Also prefetch market overview
                    market_overview_response = await session.call_tool("get_market_overview", {})
        except Exception as e:
            logger.error(f"Error prefetching data: {e}")
        
        # Parse responses
        stock_data = None
        if stock_data_response:
            if hasattr(stock_data_response, 'value'):
                try:
                    stock_data = json.loads(stock_data_response.value)
                except Exception as e:
                    logger.error(f"Error parsing stock data: {e}")
            elif isinstance(stock_data_response, str):
                try:
                    stock_data = json.loads(stock_data_response)
                except Exception as e:
                    logger.error(f"Error parsing stock data string: {e}")
        
        market_overview = None
        if market_overview_response:
            if hasattr(market_overview_response, 'value'):
                try:
                    market_overview = json.loads(market_overview_response.value)
                except Exception as e:
                    logger.error(f"Error parsing market overview: {e}")
            elif isinstance(market_overview_response, str):
                try:
                    market_overview = json.loads(market_overview_response)
                except Exception as e:
                    logger.error(f"Error parsing market overview string: {e}")
        
        # Create crew with prefetched data if available
        crew = MarketAnalysisCrew(
            symbol="AAPL",
            period="1mo",
            analysis_date=datetime.now(),
            prefetched_market_data=stock_data,
            market_overview_data=market_overview
        )
        
        result = await crew.run_analysis()
        logger.info("Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return None
    finally:
        if crew:
            # Cleanup any resources
            for agent in crew.agents:
                if hasattr(agent, 'tools'):
                    for tool in agent.tools:
                        if hasattr(tool, 'close'):
                            await tool.close()

def run_crew():
    """Entry point for running the crew from command line"""
    try:
        result = asyncio.run(main())
        if result:
            print("Analysis completed. Results saved to output/market_research.md")
            return 0  # Successful exit
        else:
            print("Analysis completed with errors. Check the logs for details.")
            return 1  # Error exit
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130  # Standard exit code for interrupt
    except Exception as e:
        logger.error(f"An error occurred while running the crew: {e}")
        return 1  # Error exit

if __name__ == "__main__":
    sys.exit(run_crew())  # Use sys.exit to properly set exit code
