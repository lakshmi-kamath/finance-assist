import os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

def get_knowledge_base_config() -> Dict:
    """Get knowledge base configuration with enhanced foreign exchange support"""
    return {
        'alphavantage_api_key': os.getenv('ALPHAVANTAGE_API_KEY', 'demo'),
        'fred_api_key': os.getenv('FRED_API_KEY', 'demo'),
        'quandl_api_key': os.getenv('QUANDL_API_KEY', 'demo'),
        'polygon_api_key': os.getenv('POLYGON_API_KEY', 'demo'),
        'finnhub_api_key': os.getenv('FINNHUB_API_KEY', 'demo'),
        
        'vector_store': {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'index_path': 'knowledge_base/vector_store/faiss_index',
            'metadata_path': 'knowledge_base/vector_store/metadata.json',
            'max_documents': 15000,  # Increased for foreign exchange filings
            'similarity_threshold': 0.5
        },
        'fred_config': {
            'rate_limit_seconds': 0.1,
            'max_retries': 3,
            'default_lookback_days': 30,
            'comprehensive_lookback_days': 90,
            'high_priority_indicators': ['INTEREST_RATE', 'VIX', 'USD_INDEX']
        },
        'data_collection': {
            'news_sources': [
                'reuters_finance',
                'marketwatch', 
                'yahoo_finance_news',
                'nikkei_asia',  # Added for Asian market coverage
                'korea_herald_business'
            ],
            'asia_tech_symbols': [
                'TSM',        # Taiwan Semiconductor (NYSE ADR)
                '005930.KS',  # Samsung Electronics (Korea)
                'BABA',       # Alibaba (NYSE ADR) 
                'TCEHY',      # Tencent (OTC ADR)
                '6758.T',     # Sony Group (Tokyo)
                'ASML',       # ASML (NASDAQ ADR)
                '9984.T',     # SoftBank Group (Tokyo)
                '0700.HK'     # Tencent Holdings (Hong Kong)
            ],
            'update_frequency_hours': 4,
            'news_lookback_hours': 24,
            'filing_lookback_days': 90
        },
        
        'foreign_exchanges': {
            'KSE': {
                'name': 'Korea Stock Exchange',
                'enabled': True,
                'rate_limit_seconds': 2.0,
                'max_retries': 2,
                'supported_symbols': ['005930.KS', '000660.KS', '035420.KS'],
                'filing_types': ['annual_report', 'quarterly_report', 'disclosure'],
                'base_urls': {
                    'kind': 'https://kind.krx.co.kr/eng',
                    'dart': 'https://dart.fss.or.kr'
                }
            },
            'TSE': {
                'name': 'Tokyo Stock Exchange',
                'enabled': True,
                'rate_limit_seconds': 2.0,
                'max_retries': 2,
                'supported_symbols': ['6758.T', '9984.T', '7203.T', '6861.T'],
                'filing_types': ['securities_report', 'quarterly_report', 'timely_disclosure'],
                'base_urls': {
                    'jpx': 'https://www.jpx.co.jp/english',
                    'tdnet': 'https://www.release.tdnet.info'
                }
            },
            'SEHK': {
                'name': 'Hong Kong Stock Exchange',
                'enabled': True,
                'rate_limit_seconds': 1.5,
                'max_retries': 2,
                'supported_symbols': ['0700.HK', '0941.HK', '1299.HK'],
                'filing_types': ['annual_report', 'interim_report', 'announcement'],
                'base_urls': {
                    'hkex': 'https://www.hkexnews.hk'
                }
            },
            'SEC': {
                'name': 'Securities and Exchange Commission',
                'enabled': True,
                'rate_limit_seconds': 1.0,
                'max_retries': 3,
                'supported_symbols': ['TSM', 'BABA', 'TCEHY', 'ASML'],
                'filing_types': ['10-K', '10-Q', '8-K', '20-F'],
                'base_urls': {
                    'edgar': 'https://www.sec.gov'
                }
            }
        },
        
        'processing': {
            'chunk_size': 512,
            'chunk_overlap': 50,
            'batch_size': 100,
            'max_content_length': 5000,  # For filing content
            'enable_content_extraction': True
        },
        
        'fallback_options': {
            'enable_fallback_scraping': True,
            'fallback_sources': [
                'company_investor_relations',
                'financial_portals',
                'regulatory_announcements'
            ],
            'fallback_timeout_seconds': 30
        },
        
        'company_mappings': {
            # Korean companies
            '005930.KS': {
                'name': 'Samsung Electronics Co Ltd',
                'english_name': 'Samsung Electronics',
                'sector': 'Technology Hardware',
                'country': 'South Korea',
                'currency': 'KRW'
            },
            '000660.KS': {
                'name': 'SK Hynix Inc',
                'english_name': 'SK Hynix',
                'sector': 'Semiconductors',
                'country': 'South Korea',
                'currency': 'KRW'
            },
            '035420.KS': {
                'name': 'NAVER Corporation',
                'english_name': 'NAVER',
                'sector': 'Internet Services',
                'country': 'South Korea',
                'currency': 'KRW'
            },
            
            # Japanese companies
            '6758.T': {
                'name': 'Sony Group Corporation',
                'english_name': 'Sony Group',
                'sector': 'Consumer Electronics',
                'country': 'Japan',
                'currency': 'JPY'
            },
            '9984.T': {
                'name': 'SoftBank Group Corp',
                'english_name': 'SoftBank Group',
                'sector': 'Telecommunications',
                'country': 'Japan',
                'currency': 'JPY'
            },
            '7203.T': {
                'name': 'Toyota Motor Corporation',
                'english_name': 'Toyota Motor',
                'sector': 'Automotive',
                'country': 'Japan',
                'currency': 'JPY'
            },
            
            # Hong Kong companies
            '0700.HK': {
                'name': 'Tencent Holdings Limited',
                'english_name': 'Tencent Holdings',
                'sector': 'Internet Services',
                'country': 'Hong Kong',
                'currency': 'HKD'
            },
            
            # US-listed ADRs
            'TSM': {
                'name': 'Taiwan Semiconductor Manufacturing Company Limited',
                'english_name': 'TSMC',
                'sector': 'Semiconductors',
                'country': 'Taiwan',
                'currency': 'USD',
                'listing_type': 'ADR'
            },
            'BABA': {
                'name': 'Alibaba Group Holding Limited',
                'english_name': 'Alibaba Group',
                'sector': 'E-commerce',
                'country': 'China',
                'currency': 'USD',
                'listing_type': 'ADR'
            },
            'TCEHY': {
                'name': 'Tencent Holdings Limited',
                'english_name': 'Tencent Holdings',
                'sector': 'Internet Services',
                'country': 'China',
                'currency': 'USD',
                'listing_type': 'ADR'
            },
            'ASML': {
                'name': 'ASML Holding N.V.',
                'english_name': 'ASML Holding',
                'sector': 'Semiconductor Equipment',
                'country': 'Netherlands',
                'currency': 'USD',
                'listing_type': 'ADR'
            }
        },
        
        'monitoring': {
            'enable_pipeline_monitoring': True,
            'log_level': 'INFO',
            'max_log_size_mb': 100,
            'alert_on_failures': True,
            'success_rate_threshold': 0.8
        },
        
        'scheduling': {
            'market_data_frequency': '4h',
            'news_frequency': '6h',
            'filings_frequency': '24h',
            'comprehensive_update_frequency': '7d',
            'timezone': 'UTC'
        }
    }