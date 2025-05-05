"""
This module provides data visualization capabilities for the Business Intelligence Agent.
It generates various charts for stock market analysis.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import os
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import calendar
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def ensure_output_dirs():
    """Create output directories if they don't exist."""
    charts_dir = Path("output/charts")
    charts_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir

def plot_price_history(symbol: str, history_data: Dict, output_dir: Optional[Path] = None) -> str:
    """
    Plot price history for a stock.
    
    Args:
        symbol: Stock symbol
        history_data: Dictionary containing historical price data
        output_dir: Directory to save the plot (optional)
        
    Returns:
        Path to the saved chart
    """
    try:
        # Ensure output directory exists
        if output_dir is None:
            output_dir = ensure_output_dirs()
        
        logger.info(f"[plot_price_history] Plotting price history for {symbol}")
        logger.info(f"[plot_price_history] History data keys: {list(history_data.keys() if isinstance(history_data, dict) else [])}")
        
        # Extract price and date data
        dates = []
        closes = []
        
        # Determine data structure and extract accordingly
        if "history" in history_data:
            # Structure from get_stock_data
            logger.info(f"[plot_price_history] Found 'history' key, using get_stock_data format")
            history = history_data["history"]
            logger.info(f"[plot_price_history] History keys: {list(history.keys() if isinstance(history, dict) else [])}")
            
            if "Close" in history:
                date_keys = sorted(history["Close"].keys())
                dates = [pd.to_datetime(date_key) for date_key in date_keys]
                closes = [history["Close"][date_key] for date_key in date_keys]
                logger.info(f"[plot_price_history] Extracted {len(dates)} data points from 'Close' key")
        elif "data" in history_data and "prices" in history_data["data"]:
            # Structure from historical_analysis
            logger.info(f"[plot_price_history] Found 'data.prices' keys, using historical_analysis format")
            for price_point in history_data["data"]["prices"]:
                dates.append(pd.to_datetime(price_point["date"]))
                closes.append(price_point["close"])
            logger.info(f"[plot_price_history] Extracted {len(dates)} data points from 'data.prices'")
        elif "historical_analysis" in history_data:
            # Structure where historical_analysis is a nested key
            logger.info(f"[plot_price_history] Found 'historical_analysis' key")
            historical = history_data["historical_analysis"]
            
            if isinstance(historical, dict) and "data" in historical and "prices" in historical["data"]:
                logger.info(f"[plot_price_history] Using data from nested historical_analysis.data.prices")
                for price_point in historical["data"]["prices"]:
                    dates.append(pd.to_datetime(price_point["date"]))
                    closes.append(price_point["close"])
                logger.info(f"[plot_price_history] Extracted {len(dates)} data points from nested structure")
        
        if not dates or not closes:
            logger.error(f"[plot_price_history] No valid price data found for {symbol}")
            return ""
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(dates, closes, 'b-', linewidth=2)
        
        # Add title and labels
        plt.title(f'{symbol} Price History', fontsize=16)
        plt.ylabel('Price ($)', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        
        # Format the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        
        # Add grid lines
        plt.grid(True, alpha=0.3)
        
        # Add current price annotation
        if closes:
            current_price = closes[-1]
            plt.text(dates[-1], current_price, f'${current_price:.2f}', 
                    fontsize=12, ha='right', va='bottom')
        
        # Save the chart
        filename = f"{symbol}_price_history.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[plot_price_history] Chart saved to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"[plot_price_history] Error generating price history chart: {e}", exc_info=True)
        return ""

def plot_volume_history(symbol: str, history_data: Dict, output_dir: Optional[Path] = None) -> str:
    """
    Plot volume history for a stock.
    
    Args:
        symbol: Stock symbol
        history_data: Dictionary containing historical volume data
        output_dir: Directory to save the plot (optional)
        
    Returns:
        Path to the saved chart
    """
    try:
        # Ensure output directory exists
        if output_dir is None:
            output_dir = ensure_output_dirs()
        
        logger.info(f"[plot_volume_history] Plotting volume history for {symbol}")
        logger.info(f"[plot_volume_history] History data keys: {list(history_data.keys() if isinstance(history_data, dict) else [])}")
        
        # Extract volume and date data
        dates = []
        volumes = []
        
        # Determine data structure and extract accordingly
        if "history" in history_data:
            # Structure from get_stock_data
            logger.info(f"[plot_volume_history] Found 'history' key, using get_stock_data format")
            history = history_data["history"]
            logger.info(f"[plot_volume_history] History keys: {list(history.keys() if isinstance(history, dict) else [])}")
            
            if "Volume" in history:
                date_keys = sorted(history["Volume"].keys())
                dates = [pd.to_datetime(date_key) for date_key in date_keys]
                volumes = [history["Volume"][date_key] for date_key in date_keys]
                logger.info(f"[plot_volume_history] Extracted {len(dates)} data points from 'Volume' key")
        elif "data" in history_data and "volumes" in history_data["data"]:
            # Structure from historical_analysis
            logger.info(f"[plot_volume_history] Found 'data.volumes' keys, using historical_analysis format")
            for volume_point in history_data["data"]["volumes"]:
                dates.append(pd.to_datetime(volume_point["date"]))
                volumes.append(volume_point["volume"])
            logger.info(f"[plot_volume_history] Extracted {len(dates)} data points from 'data.volumes'")
        elif "historical_analysis" in history_data:
            # Structure where historical_analysis is a nested key
            logger.info(f"[plot_volume_history] Found 'historical_analysis' key")
            historical = history_data["historical_analysis"]
            
            if isinstance(historical, dict) and "data" in historical and "volumes" in historical["data"]:
                logger.info(f"[plot_volume_history] Using data from nested historical_analysis.data.volumes")
                for volume_point in historical["data"]["volumes"]:
                    dates.append(pd.to_datetime(volume_point["date"]))
                    volumes.append(volume_point["volume"])
                logger.info(f"[plot_volume_history] Extracted {len(dates)} data points from nested structure")
        
        if not dates or not volumes:
            logger.error(f"[plot_volume_history] No valid volume data found for {symbol}")
            return ""
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.bar(dates, volumes, color='teal', alpha=0.7)
        
        # Add title and labels
        plt.title(f'{symbol} Trading Volume', fontsize=16)
        plt.ylabel('Volume', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        
        # Format the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        
        # Format y-axis with commas for thousands
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x):,}"))
        
        # Add grid lines
        plt.grid(True, alpha=0.3)
        
        # Save the chart
        filename = f"{symbol}_volume_history.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[plot_volume_history] Chart saved to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"[plot_volume_history] Error generating volume history chart: {e}", exc_info=True)
        return ""

def plot_technical_indicators(symbol: str, historical_data: Dict, output_dir: Optional[Path] = None) -> str:
    """
    Plot technical indicators (e.g., Moving Averages) for a stock.
    
    Args:
        symbol: Stock symbol
        historical_data: Dictionary containing historical analysis data
        output_dir: Directory to save the plot (optional)
        
    Returns:
        Path to the saved chart
    """
    try:
        # Ensure output directory exists
        if output_dir is None:
            output_dir = ensure_output_dirs()
        
        logger.info(f"[plot_technical_indicators] Plotting technical indicators for {symbol}")
        logger.info(f"[plot_technical_indicators] Historical data keys: {list(historical_data.keys() if isinstance(historical_data, dict) else [])}")
        
        metrics = {}
        data_prices = []
        
        # Check for various data structures
        if "metrics" in historical_data:
            logger.info(f"[plot_technical_indicators] Found 'metrics' key directly")
            metrics = historical_data["metrics"]
        elif "historical_analysis" in historical_data and isinstance(historical_data["historical_analysis"], dict):
            logger.info(f"[plot_technical_indicators] Looking for metrics in 'historical_analysis' key")
            hist_analysis = historical_data["historical_analysis"]
            if "metrics" in hist_analysis:
                logger.info(f"[plot_technical_indicators] Found 'metrics' in historical_analysis")
                metrics = hist_analysis["metrics"]
            
        # Log metrics found
        logger.info(f"[plot_technical_indicators] Available metrics: {list(metrics.keys() if isinstance(metrics, dict) else [])}")
        
        # Extract price data from various possible structures
        if "data" in historical_data and "prices" in historical_data["data"]:
            logger.info(f"[plot_technical_indicators] Found price data in 'data.prices'")
            data_prices = historical_data["data"]["prices"]
        elif "historical_analysis" in historical_data:
            hist_analysis = historical_data["historical_analysis"]
            if isinstance(hist_analysis, dict) and "data" in hist_analysis and "prices" in hist_analysis["data"]:
                logger.info(f"[plot_technical_indicators] Found price data in 'historical_analysis.data.prices'")
                data_prices = hist_analysis["data"]["prices"]
        
        # Extract price data for plotting
        dates = []
        closes = []
        
        # Get price data from the detected structure
        if data_prices:
            logger.info(f"[plot_technical_indicators] Processing {len(data_prices)} price data points")
            for price_point in data_prices:
                dates.append(pd.to_datetime(price_point["date"]))
                closes.append(price_point["close"])
        
        if not dates or not closes:
            logger.error(f"[plot_technical_indicators] No valid price data found for {symbol}")
            return ""
        
        # Create dataframe from price data
        df = pd.DataFrame({
            'date': dates,
            'close': closes
        })
        logger.info(f"[plot_technical_indicators] Created dataframe with {len(df)} rows")
        
        # Calculate moving averages (if not already provided)
        mean_value = None
        if isinstance(metrics, dict) and "MEAN" in metrics:
            mean_value = metrics["MEAN"]
            logger.info(f"[plot_technical_indicators] Using MEAN from metrics: {mean_value}")
        else:
            mean_value = df['close'].mean()
            logger.info(f"[plot_technical_indicators] Calculated mean value: {mean_value}")
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot close price
        plt.plot(df['date'], df['close'], 'b-', linewidth=2, label='Close Price')
        
        # Plot moving average as horizontal line
        plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: ${mean_value:.2f}')
        
        # If standard deviation is available, plot bands
        std_dev = None
        if isinstance(metrics, dict) and "STDDEV" in metrics:
            std_dev = metrics["STDDEV"]
            logger.info(f"[plot_technical_indicators] Using STDDEV from metrics: {std_dev}")
            plt.axhline(y=mean_value + std_dev, color='g', linestyle=':', label=f'+1 StdDev: ${(mean_value + std_dev):.2f}')
            plt.axhline(y=mean_value - std_dev, color='g', linestyle=':', label=f'-1 StdDev: ${(mean_value - std_dev):.2f}')
        
        # Add title and labels
        plt.title(f'{symbol} Technical Analysis', fontsize=16)
        plt.ylabel('Price ($)', fontsize=14)
        plt.xlabel('Date', fontsize=14)
        
        # Format the x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        
        # Add grid lines and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Save the chart
        filename = f"{symbol}_technical_analysis.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[plot_technical_indicators] Chart saved to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"[plot_technical_indicators] Error generating technical indicators chart: {e}", exc_info=True)
        return ""

def generate_all_charts(symbol: str, market_data: Dict, historical_data: Dict) -> Dict[str, str]:
    """
    Generate all available charts for a stock analysis.
    
    Args:
        symbol: Stock symbol
        market_data: Dictionary with current market data
        historical_data: Dictionary with historical analysis data
        
    Returns:
        Dictionary mapping chart types to file paths
    """
    logger.info(f"[generate_all_charts] Starting chart generation for {symbol}")
    logger.info(f"[generate_all_charts] Market data keys: {list(market_data.keys() if isinstance(market_data, dict) else [])}")
    logger.info(f"[generate_all_charts] Historical data keys: {list(historical_data.keys() if isinstance(historical_data, dict) else [])}")
    
    # Ensure we have valid data formats
    if not isinstance(market_data, dict):
        logger.error(f"[generate_all_charts] Market data is not a dictionary: {type(market_data)}")
        market_data = {}
    
    if not isinstance(historical_data, dict):
        logger.error(f"[generate_all_charts] Historical data is not a dictionary: {type(historical_data)}")
        historical_data = {}
    
    output_dir = ensure_output_dirs()
    charts = {}
    
    try:
        # Skip price history chart as it's redundant with technical indicators chart
        # Instead, focus on more specialized visualizations
        
        # Generate volume history chart
        logger.info(f"[generate_all_charts] Attempting to generate volume history chart")
        volume_chart_path = plot_volume_history(symbol, market_data, output_dir)
        if volume_chart_path:
            charts["volume_history"] = volume_chart_path
            logger.info(f"[generate_all_charts] Volume history chart generated: {volume_chart_path}")
        else:
            logger.warning(f"[generate_all_charts] Failed to generate volume history chart")
        
        # Generate technical indicators chart - this also shows price history with mean and stddev
        if historical_data:
            logger.info(f"[generate_all_charts] Attempting to generate technical indicators chart")
            tech_chart_path = plot_technical_indicators(symbol, historical_data, output_dir)
            if tech_chart_path:
                charts["technical_indicators"] = tech_chart_path
                logger.info(f"[generate_all_charts] Technical indicators chart generated: {tech_chart_path}")
            else:
                logger.warning(f"[generate_all_charts] Failed to generate technical indicators chart")
                
            # Generate candlestick chart - keep only this chart from the new visualizations
            logger.info(f"[generate_all_charts] Attempting to generate candlestick chart")
            candlestick_path = plot_candlestick(symbol, historical_data, output_dir)
            if candlestick_path:
                charts["candlestick"] = candlestick_path
                logger.info(f"[generate_all_charts] Candlestick chart generated: {candlestick_path}")
            else:
                logger.warning(f"[generate_all_charts] Failed to generate candlestick chart")
        else:
            logger.warning(f"[generate_all_charts] Skipping additional charts due to missing historical data")
        
        # If we couldn't generate any charts, create a fallback chart
        if not charts:
            logger.warning(f"[generate_all_charts] No charts were generated, creating a fallback chart")
            fallback_chart_path = create_fallback_chart(symbol, output_dir)
            if fallback_chart_path:
                charts["overview"] = fallback_chart_path
                logger.info(f"[generate_all_charts] Fallback chart generated: {fallback_chart_path}")
        
        logger.info(f"[generate_all_charts] Generated {len(charts)} charts for {symbol}: {list(charts.keys())}")
        return charts
    except Exception as e:
        logger.error(f"[generate_all_charts] Error generating charts: {e}", exc_info=True)
        return charts

def create_fallback_chart(symbol: str, output_dir: Optional[Path] = None) -> str:
    """Create a simple fallback chart when other charts can't be generated."""
    try:
        if output_dir is None:
            output_dir = ensure_output_dirs()
        
        # Create a simple chart with the symbol name
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Data visualization for {symbol}\nInsufficient data for detailed charts", 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=14, transform=plt.gca().transAxes)
        plt.axis('off')
        
        # Save the chart
        filename = f"{symbol}_overview.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[create_fallback_chart] Fallback chart saved to {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"[create_fallback_chart] Error creating fallback chart: {e}", exc_info=True)
        return ""

def plot_candlestick(symbol: str, history_data: Dict, output_dir: Optional[Path] = None) -> str:
    """
    Create a candlestick chart showing OHLC data.
    
    Args:
        symbol: Stock symbol
        history_data: Dictionary containing historical price data
        output_dir: Directory to save the plot (optional)
        
    Returns:
        Path to the saved chart
    """
    try:
        # Ensure output directory exists
        if output_dir is None:
            output_dir = ensure_output_dirs()
        
        logger.info(f"[plot_candlestick] Creating candlestick chart for {symbol}")
        
        # Extract OHLC data
        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        
        # Get price data
        data_prices = []
        if "data" in history_data and "prices" in history_data["data"]:
            data_prices = history_data["data"]["prices"]
        elif "historical_analysis" in history_data:
            hist_analysis = history_data["historical_analysis"]
            if isinstance(hist_analysis, dict) and "data" in hist_analysis and "prices" in hist_analysis["data"]:
                data_prices = hist_analysis["data"]["prices"]
        
        if not data_prices:
            logger.error(f"[plot_candlestick] No valid price data found for {symbol}")
            return ""
        
        logger.info(f"[plot_candlestick] Processing {len(data_prices)} OHLC data points")
        
        # Extract data
        for price_point in data_prices:
            dates.append(pd.to_datetime(price_point["date"]))
            opens.append(price_point["open"])
            highs.append(price_point["high"])
            lows.append(price_point["low"])
            closes.append(price_point["close"])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate width for candlesticks
        width = 0.6
        width_in_days = (dates[-1] - dates[0]).days / len(dates) * width

        # Create candlesticks
        for i in range(len(dates)):
            # Determine if up or down day
            if closes[i] >= opens[i]:
                color = 'green'
                body_color = 'white'  # hollow for up days
            else:
                color = 'red'
                body_color = 'red'    # filled for down days
            
            # Plot the candlestick
            # Central line (high to low)
            ax.plot([dates[i], dates[i]], [lows[i], highs[i]], color=color, linewidth=1.5)
            
            # Rectangle for the body
            rect = Rectangle(
                xy=(dates[i] - pd.Timedelta(days=width_in_days/2), opens[i]),
                width=pd.Timedelta(days=width_in_days),
                height=closes[i] - opens[i],
                fill=True,
                edgecolor=color,
                facecolor=body_color,
                linewidth=1.5
            )
            ax.add_patch(rect)
        
        # Format the chart
        ax.set_title(f'{symbol} Candlestick Chart', fontsize=16)
        ax.set_ylabel('Price ($)', fontsize=14)
        ax.set_xlabel('Date', fontsize=14)
        
        # Format axes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        green_patch = Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='green')
        red_patch = Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='red')
        ax.legend([green_patch, red_patch], ['Bullish', 'Bearish'], loc='upper left')
        
        # Save the chart
        filename = f"{symbol}_candlestick.png"
        filepath = output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[plot_candlestick] Chart saved to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"[plot_candlestick] Error generating candlestick chart: {e}", exc_info=True)
        return ""