"""
This script provides a Market Data MCP server that fetches stock data,
market news, and market overview using Alpha Vantage APIs.
It uses the FastMCP library to create the server and handle requests.
"""
import requests
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import os
from mcp.server.fastmcp import FastMCP
import json
import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests as curl_requests
import yfinance_cookie_patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create our MCP server with a descriptive name
mcp = FastMCP("market_data_service")

yfinance_cookie_patch.patch_yfdata_cookie_basic()

# Create a curl_cffi session impersonating Chrome
session = curl_requests.Session(impersonate="chrome")

class MarketDataHelper:
    """Helper class for market data operations"""
    
    def __init__(self):
        """Initialize with API configurations"""
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not self.alpha_vantage_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
        
        self.base_url = "https://www.alphavantage.co/query"
    
    def _make_request(self, params: Dict[str, str]) -> Dict:
        """Make a request to Alpha Vantage API with error handling"""
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "Error Message" in data:
                raise ValueError(data["Error Message"])
            if "Note" in data and "API call frequency" in data["Note"]:
                raise ValueError("API rate limit reached. Please try again later.")
                
            return data
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError("Invalid response from API")

# Initialize the helper class
market_helper = MarketDataHelper()

@mcp.tool()
async def get_stock_data(
    symbols: str,  # Accept comma-separated string or single symbol
    interval: str = "1d",
    period: str = "1y",
    analysis_date: Optional[str] = None
) -> str:
    logger.info(f"[data_fetcher.py] get_stock_data called with symbols={symbols}, interval={interval}, period={period}, analysis_date={analysis_date}")
    try:
        # Parse symbols
        if isinstance(symbols, str):
            symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            symbol_list = list(symbols)
        
        results = {}
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol, session=session)
                info = ticker.info
                fast_info = getattr(ticker, 'fast_info', {})
                hist = ticker.history(period=period, interval=interval)
                # Convert index to string for JSON serialization
                if not hist.empty:
                    hist = hist.copy()
                    hist.index = hist.index.strftime('%Y-%m-%d')
                    hist_dict = hist.tail(5).to_dict()
                else:
                    hist_dict = {}
                current_price = fast_info.get('last_price') or info.get('regularMarketPrice')
                previous_close = info.get('regularMarketPreviousClose')
                change = None
                change_percent = None
                if current_price is not None and previous_close is not None:
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close else None
                volume = info.get('regularMarketVolume')
                last_trading_day = info.get('regularMarketTime')
                results[symbol] = {
                    "symbol": symbol,
                    "name": info.get('shortName', ''),
                    "market_cap": info.get('marketCap', ''),
                    "sector": info.get('sector', ''),
                    "industry": info.get('industry', ''),
                    "current_price": current_price,
                    "previous_close": previous_close,
                    "change": change,
                    "change_percent": change_percent,
                    "volume": volume,
                    "last_trading_day": last_trading_day,
                    "data_last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "history": hist_dict,
                }
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                results[symbol] = {"error": str(e)}
        logger.info(f"[data_fetcher.py] get_stock_data returning: {results}")
        return json.dumps(results, indent=2)
    except Exception as e:
        logger.error(f"[data_fetcher.py] get_stock_data error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "diagnostic": f"symbols={symbols}, interval={interval}, period={period}, analysis_date={analysis_date}"})

@mcp.tool()
async def get_market_news(
    symbol: str,
    max_articles: int = 10,
    days: int = 7,
    analysis_date: Optional[str] = None
) -> str:
    logger.info(f"[data_fetcher.py] get_market_news called with symbol={symbol}, max_articles={max_articles}, days={days}, analysis_date={analysis_date}")
    try:
        # Calculate time range
        end_date = datetime.now()
        if analysis_date:
            end_date = datetime.strptime(analysis_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        time_from = start_date.strftime('%Y%m%dT0000')
        time_to = end_date.strftime('%Y%m%dT2359')
        
        # Get news data
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": time_from,
            "time_to": time_to,
            "limit": max_articles,
            "sort": "RELEVANCE",
            "apikey": market_helper.alpha_vantage_key
        }
        
        news_data = market_helper._make_request(params)
        
        if "feed" not in news_data:
            return json.dumps({
                "analysis_date": analysis_date or datetime.now().strftime('%Y-%m-%d'),
                "articles": []
            })
        
        # Process and format articles
        articles = []
        for item in news_data["feed"]:
            article = {
                'title': item.get('title', ''),
                'summary': item.get('summary', ''),
                'source': item.get('source', ''),
                'url': item.get('url', ''),
                'published_at': item.get('time_published', ''),
                'sentiment': {
                    'score': float(item.get('overall_sentiment_score', 0)),
                    'label': item.get('overall_sentiment_label', 'neutral')
                },
                'relevance_score': float(item.get('relevance_score', 0)),
                'topics': item.get('topics', []),
                'ticker_sentiment': item.get('ticker_sentiment', [])
            }
            articles.append(article)
        
        logger.info(f"[data_fetcher.py] get_market_news returning: {articles[:max_articles]}")
        return json.dumps({
            "analysis_date": analysis_date or datetime.now().strftime('%Y-%m-%d'),
            "articles": articles[:max_articles]  # Ensure we don't exceed max_articles
        }, indent=2)
        
    except Exception as e:
        logger.error(f"[data_fetcher.py] get_market_news error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "diagnostic": f"symbol={symbol}, max_articles={max_articles}, days={days}, analysis_date={analysis_date}"})

@mcp.tool()
async def get_market_overview(symbols: str = "SPY,DIA,QQQ,VIXY", analysis_date: Optional[str] = None) -> str:
    logger.info(f"[data_fetcher.py] get_market_overview called with symbols={symbols}, analysis_date={analysis_date}")
    try:
        symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        market_data = {}
        errors = []
        for symbol in symbol_list:
            try:
                ticker = yf.Ticker(symbol, session=session)
                info = ticker.info
                fast_info = getattr(ticker, 'fast_info', {})
                current_price = fast_info.get('last_price') or info.get('regularMarketPrice')
                previous_close = info.get('regularMarketPreviousClose')
                change = None
                change_percent = None
                if current_price is not None and previous_close is not None:
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close else None
                volume = info.get('regularMarketVolume')
                market_data[symbol] = {
                    'name': info.get('shortName', ''),
                    'current_value': current_price,
                    'change': change,
                    'change_percent': change_percent,
                    'volume': volume
                }
            except Exception as e:
                errors.append(f"Error fetching data for {symbol}: {str(e)}")
        result = {
            "analysis_date": analysis_date or datetime.now().strftime('%Y-%m-%d'),
            "market_indices": market_data,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "warnings": errors if errors else None
        }
        if not market_data:
            result["error"] = "Failed to fetch market data"
            result["details"] = errors
        logger.info(f"[data_fetcher.py] get_market_overview returning: {result}")
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"[data_fetcher.py] get_market_overview error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "diagnostic": f"symbols={symbols}, analysis_date={analysis_date}"})

@mcp.tool()
async def get_historical_analysis(
    symbols: str,  # Accept comma-separated string or single symbol
    window_size: int = 20,
    interval: str = "1d",
    period: str = "2mo",
    calculations: str = "MEAN,STDDEV,VARIANCE,CORRELATION",
    analysis_date: Optional[str] = None
) -> str:
    logger.info(f"[data_fetcher.py] get_historical_analysis called with symbols={symbols}, window_size={window_size}, interval={interval}, period={period}, calculations={calculations}, analysis_date={analysis_date}")
    try:
        if isinstance(symbols, str):
            symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            symbol_list = list(symbols)
        
        results = {}
        for symbol in symbol_list:
            try:
                logger.info(f"[data_fetcher.py] Fetching historical data for {symbol}")
                ticker = yf.Ticker(symbol, session=session)
                hist = ticker.history(period=period, interval=interval)
                if hist.empty:
                    logger.warning(f"[data_fetcher.py] No historical data found for {symbol}")
                    results[symbol] = {"error": "No historical data found"}
                    continue
                    
                logger.info(f"[data_fetcher.py] Retrieved {len(hist)} data points for {symbol}")
                
                # Create explicit data section with prices and volumes
                data_section = {
                    "prices": [],
                    "volumes": []
                }
                
                # Convert DataFrame to list of records for prices and volumes
                for date, row in hist.iterrows():
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Add price data
                    data_section["prices"].append({
                        "date": date_str,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                    })
                    
                    # Add volume data
                    data_section["volumes"].append({
                        "date": date_str,
                        "volume": int(row["Volume"])
                    })
                
                # Calculate metrics
                metrics = {}
                calculations_list = [calc.strip() for calc in calculations.split(',')]
                
                # Add metrics with explicit MEAN, STDDEV keys for visualization
                metrics["MEAN"] = float(hist['Close'].mean())
                metrics["STDDEV"] = float(hist['Close'].std())
                
                if window_size > 0:
                    rolling = hist['Close'].rolling(window=window_size)
                    metrics['rolling_mean'] = float(rolling.mean().iloc[-1])
                    metrics['rolling_std'] = float(rolling.std().iloc[-1])
                    if 'STDDEV(annualized=True)' in calculations:
                        metrics['rolling_std_annualized'] = float(metrics['rolling_std'] * np.sqrt(252))
                
                for calc in calculations_list:
                    calc = calc.split('(')[0]
                    if calc == "MEAN":
                        metrics['mean'] = float(hist['Close'].mean())
                    elif calc == "STDDEV":
                        metrics['stddev'] = float(hist['Close'].std())
                    elif calc == "VARIANCE":
                        metrics['variance'] = float(hist['Close'].var())
                    elif calc == "CORRELATION":
                        metrics['price_volume_correlation'] = float(hist['Close'].corr(hist['Volume']))
                
                # Calculate SMAs
                if len(hist) >= 20:
                    metrics['sma_20'] = float(hist['Close'].rolling(window=20).mean().iloc[-1])
                else:
                    metrics['sma_20'] = None
                    
                if len(hist) >= 50:
                    metrics['sma_50'] = float(hist['Close'].rolling(window=50).mean().iloc[-1])
                else:
                    metrics['sma_50'] = None
                    
                metrics['current_price'] = float(hist['Close'].iloc[-1])
                
                if len(hist) > 1:
                    metrics['price_change'] = float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2])
                    metrics['price_change_percent'] = float((metrics['price_change'] / hist['Close'].iloc[-2]) * 100) if hist['Close'].iloc[-2] != 0 else None
                else:
                    metrics['price_change'] = None
                    metrics['price_change_percent'] = None
                    
                results[symbol] = {
                    "symbol": symbol,
                    "analysis_date": analysis_date or datetime.now().strftime('%Y-%m-%d'),
                    "window_size": window_size,
                    "interval": interval,
                    "period": period,
                    "calculations_performed": calculations_list,
                    "metrics": metrics,
                    "data": data_section,  # Add explicit data section
                    "data_points": len(hist),
                    "latest_date": hist.index[-1].strftime('%Y-%m-%d'),
                    "earliest_date": hist.index[0].strftime('%Y-%m-%d')
                }
                logger.info(f"[data_fetcher.py] Successfully calculated metrics for {symbol}")
            except Exception as e:
                logger.error(f"[data_fetcher.py] Error analyzing {symbol}: {str(e)}", exc_info=True)
                results[symbol] = {"error": str(e)}
                
        if results:
            logger.info(f"[data_fetcher.py] get_historical_analysis returning data for {len(results)} symbols")
            return json.dumps(results, indent=2)
        else:
            error_msg = {"error": "No historical data could be retrieved for any symbols"}
            logger.error(f"[data_fetcher.py] {error_msg['error']}")
            return json.dumps(error_msg, indent=2)
    except Exception as e:
        logger.error(f"[data_fetcher.py] get_historical_analysis error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "diagnostic": f"symbols={symbols}, window_size={window_size}, interval={interval}, period={period}, calculations={calculations}, analysis_date={analysis_date}"})

if __name__ == "__main__":
    mcp.run() 