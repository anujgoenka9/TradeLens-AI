"""
Script to fetch ticker symbols and company information and store in ChromaDB for semantic search.
"""
import requests
import logging
from typing import Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import time
from requests.exceptions import RequestException
import argparse
from pathlib import Path

import yfinance as yf
from curl_cffi import requests as curl_requests
import yfinance_cookie_patch

yfinance_cookie_patch.patch_yfdata_cookie_basic()

# Create a curl_cffi session impersonating Chrome
session = curl_requests.Session(impersonate="chrome")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TickerFetcher:
    """Class to fetch and manage ticker symbols using ChromaDB."""
    
    def __init__(self):
        """Initialize the TickerFetcher with ChromaDB client."""
        # Get the absolute path to the database directory
        db_path = Path(__file__).parent.parent.parent / "chroma_db"
        db_path.mkdir(exist_ok=True)
        
        # Create a persistent client to store data
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(name="stock_tickers")
            logger.info("Using existing stock_tickers collection")
        except Exception:
            # Create new collection if it doesn't exist
            logger.info("Creating new stock_tickers collection")
            self.collection = self.chroma_client.create_collection(
                name="stock_tickers",
                metadata={"description": "Stock ticker symbols and company information"}
            )

    def fetch_exchange_tickers(self, exchange: str) -> Dict[str, Dict[str, Any]]:
        """
        Fetch ticker symbols and company information from specified exchange.
        
        Args:
            exchange: Exchange to fetch from ('NASDAQ' or 'NYSE')
            
        Returns:
            Dictionary mapping ticker symbols to company information
        """
        try:
            # NASDAQ API endpoint for listed companies
            base_url = "https://api.nasdaq.com/api/screener/stocks"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.9",
            }
            
            params = {
                "tableonly": "true",
                "limit": 10000,
                "offset": 0,
                "download": "true",
                "exchange": exchange
            }

            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            tickers = {}
            skipped = 0
            
            # Process the response data
            if data and 'data' in data and 'rows' in data['data']:
                rows = data['data']['rows']
                logger.info(f"Processing {len(rows)} rows from {exchange}")
                
                for row in rows:
                    symbol = str(row.get('symbol', '')).strip()
                    if not symbol:
                        continue

                    # Skip if market cap is too small (less than $1M)
                    market_cap = str(row.get('marketCap', '')).strip()
                    if not market_cap or market_cap == 'N/A':
                        skipped += 1
                        continue
                        
                    try:
                        market_cap_value = float(market_cap.replace(',', ''))
                        if market_cap_value < 1000000:  # $1M
                            skipped += 1
                            continue
                    except ValueError:
                        skipped += 1
                        continue

                    tickers[symbol] = {
                        'name': str(row.get('name', '')).strip(),
                        'market_cap': market_cap,
                        'sector': str(row.get('sector', '')).strip(),
                        'industry': str(row.get('industry', '')).strip()
                    }

            logger.info(f"Found {len(tickers)} valid tickers from {exchange} (skipped {skipped})")
            return tickers

        except Exception as e:
            logger.error(f"Error fetching {exchange} tickers: {e}")
            return {}

    def fetch_all_tickers(self, sample_mode: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Fetch tickers from both NASDAQ and NYSE exchanges.
        
        Args:
            sample_mode: If True, only fetch top 100 tickers by market cap for testing
            
        Returns:
            Dictionary mapping ticker symbols to company information
        """
        all_tickers = {}
        # Fetch NASDAQ tickers
        logger.info("Fetching NASDAQ tickers...")
        nasdaq_tickers = self.fetch_exchange_tickers('NASDAQ')
        all_tickers.update(nasdaq_tickers)
        # Fetch NYSE tickers
        logger.info("Fetching NYSE tickers...")
        nyse_tickers = self.fetch_exchange_tickers('NYSE')
        all_tickers.update(nyse_tickers)
        # Deduplicate by company name, prefer 'Common Stock'
        name_to_ticker = {}
        for symbol, info in all_tickers.items():
            name = info['name']
            # Fix Berkshire Hathaway ticker for Yahoo
            yahoo_symbol = symbol
            if symbol == 'BRK/A':
                yahoo_symbol = 'BRK-A'
            elif symbol == 'BRK/B':
                yahoo_symbol = 'BRK-B'
            
            # Special handling for Berkshire Hathaway - only keep BRK-A
            if 'Berkshire Hathaway' in name:
                if yahoo_symbol == 'BRK-A':
                    name_to_ticker[name] = (yahoo_symbol, info)
                continue
            
            # Prefer 'Common Stock' if duplicate
            if name not in name_to_ticker:
                name_to_ticker[name] = (yahoo_symbol, info)
            else:
                prev_symbol, prev_info = name_to_ticker[name]
                if 'Common Stock' in info['name'] and 'Common Stock' not in prev_info['name']:
                    name_to_ticker[name] = (yahoo_symbol, info)
        deduped_tickers = {symbol: info for symbol, info in [v for v in name_to_ticker.values()]}
        if sample_mode:
            # Sort tickers by market cap and take top 100
            sorted_tickers = sorted(
                deduped_tickers.items(),
                key=lambda x: float(x[1]['market_cap'].replace(',', '')),
                reverse=True
            )
            sample_tickers = dict(sorted_tickers[:100])
            logger.info(f"Sample mode: Using top {len(sample_tickers)} tickers by market cap")
            return sample_tickers
        logger.info(f"Total unique tickers fetched: {len(deduped_tickers)}")
        return deduped_tickers

    def store_in_chromadb(self, tickers: Dict[str, Dict[str, Any]]) -> None:
        """
        Store ticker information in ChromaDB.
        
        Args:
            tickers: Dictionary of ticker symbols and their information
        """
        try:
            # Prepare data for ChromaDB
            documents = []  # Company descriptions for embedding
            metadatas = []  # Additional metadata
            ids = []       # Unique IDs for each entry
            
            logger.info(f"Preparing {len(tickers)} tickers for ChromaDB storage...")
            
            for symbol, info in tickers.items():
                # Fix Berkshire Hathaway ticker for Yahoo
                yf_symbol = symbol
                if symbol == 'BRK/A':
                    yf_symbol = 'BRK-A'
                elif symbol == 'BRK/B':
                    yf_symbol = 'BRK-B'
                # Fetch longBusinessSummary from yfinance
                try:
                    yf_ticker = yf.Ticker(yf_symbol, session=session)
                    summary = yf_ticker.info.get('longBusinessSummary', '')
                except Exception as e:
                    logger.warning(f"Could not fetch longBusinessSummary for {yf_symbol}: {e}")
                    summary = ''

                # Create a rich description for better semantic search
                description = (
                    f"Company {info['name']} (Symbol: {symbol}). "
                    f"A {info['industry']} company in the {info['sector']} sector "
                    f"with market cap of {info['market_cap']}. "
                    f"Business Summary: {summary}"
                )
                
                # Store full information in metadata
                metadata = {
                    'symbol': symbol,
                    'name': info['name'],
                    'market_cap': info['market_cap'],
                    'sector': info['sector'],
                    'industry': info['industry'],
                    'longBusinessSummary': summary
                }
                
                documents.append(description)
                metadatas.append(metadata)
                ids.append(symbol)
            
            # Add data in smaller batches with progress tracking
            batch_size = 5
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            logger.info(f"Starting to store data in batches of {batch_size}")
            logger.info(f"Total batches to process: {total_batches}")
            
            start_time = time.time()
            
            for i in range(0, len(documents), batch_size):
                batch_start_time = time.time()
                end_idx = min(i + batch_size, len(documents))
                current_batch = i // batch_size + 1
                
                logger.info(f"Processing batch {current_batch}/{total_batches} (items {i+1}-{end_idx})...")
                
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                
                batch_duration = time.time() - batch_start_time
                logger.info(f"Batch {current_batch} completed in {batch_duration:.2f} seconds")
                
                # Estimate remaining time
                avg_time_per_batch = (time.time() - start_time) / current_batch
                remaining_batches = total_batches - current_batch
                est_remaining_time = avg_time_per_batch * remaining_batches
                
                logger.info(f"Estimated remaining time: {est_remaining_time:.2f} seconds")
            
            total_duration = time.time() - start_time
            logger.info(f"Successfully stored {len(tickers)} tickers in ChromaDB")
            logger.info(f"Total processing time: {total_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error storing tickers in ChromaDB: {e}")

    def query_by_company(self, query: str, n_results: int = 5) -> None:
        """
        Query the database for companies matching the search term.
        
        Args:
            query: Search query (company name, sector, etc.)
            n_results: Number of results to return (default: 5)
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Print results in a readable format
            print(f"\nSearch results for: {query}")
            print("-" * 50)
            
            if results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    print(f"Symbol: {metadata['symbol']}")
                    print(f"Name: {metadata['name']}")
                    print(f"Market Cap: {metadata['market_cap']}")
                    print(f"Sector: {metadata['sector']}")
                    print(f"Industry: {metadata['industry']}")
                    if 'longBusinessSummary' in metadata and metadata['longBusinessSummary']:
                        print(f"Summary: {metadata['longBusinessSummary']}")
                    print("-" * 30)
            else:
                print("No results found.")
                
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")

def main():
    """Main function to fetch tickers and store in ChromaDB."""
    parser = argparse.ArgumentParser(description='Fetch and query stock ticker data')
    parser.add_argument('--load', action='store_true', help='Fetch and load new data into ChromaDB')
    parser.add_argument('--query', type=str, help='Query to search for companies')
    parser.add_argument('--results', type=int, default=5, help='Number of results to return')
    args = parser.parse_args()

    fetcher = TickerFetcher()
    
    if args.load:
        # Fetch and store new data
        logger.info("Fetching and storing new data...")
        tickers = fetcher.fetch_all_tickers(sample_mode=True)
        fetcher.store_in_chromadb(tickers)
        logger.info("Data loading complete!")
    
    if args.query:
        # Just query the existing data
        fetcher.query_by_company(args.query, args.results)
    
    if not args.load and not args.query:
        # Default behavior: show example queries
        print("\nExample queries:")
        fetcher.query_by_company("technology companies in artificial intelligence")
        fetcher.query_by_company("pharmaceutical companies")
        fetcher.query_by_company("Large cap companies")

if __name__ == "__main__":
    main() 