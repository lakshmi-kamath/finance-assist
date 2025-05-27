import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

def get_config() -> Dict:
    """Get knowledge base configuration."""
    return {
        'vector_store': {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'index_path': 'knowledge_base/vector_store/faiss.index',
            'metadata_path': 'knowledge_base/vector_store/metadata.pkl',
            'index_type': 'flat',
            'dimension': 384,
            'max_documents': 15000,
            'similarity_threshold': 0.7,
            'default_k': 5
        },
        'processing': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'batch_size': 100,
            'max_content_length': 5000,
            'enable_content_extraction': True
        },
        'data_collection': {
            'news_sources': [
                'reuters_finance',
                'marketwatch',
                'yahoo_finance_news',
                'nikkei_asia',
                'korea_herald_business'
            ],
            'asia_tech_symbols': [
                'TSM', '005930.KS', 'BABA', 'TCEHY', '6758.T', 'ASML', '9984.T', '0700.HK'
            ],
            'update_frequency_hours': 4,
            'news_lookback_hours': 24,
            'filing_lookback_days': 90
        },
        'api_keys': {
            'alphavantage': os.getenv('ALPHAVANTAGE_API_KEY', 'demo'),
            'fred': os.getenv('FRED_API_KEY', 'demo'),
            'quandl': os.getenv('QUANDL_API_KEY', 'demo'),
            'polygon': os.getenv('POLYGON_API_KEY', 'demo'),
            'finnhub': os.getenv('FINNHUB_API_KEY', 'demo')
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
        'company_mappings': {
            '005930.KS': {
                'name': 'Samsung Electronics Co Ltd',
                'english_name': 'Samsung Electronics',
                'sector': 'Technology Hardware',
                'country': 'South Korea',
                'currency': 'KRW'
            },
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
            '0700.HK': {
                'name': 'Tencent Holdings Limited',
                'english_name': 'Tencent Holdings',
                'sector': 'Internet Services',
                'country': 'Hong Kong',
                'currency': 'HKD'
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
        }
    }