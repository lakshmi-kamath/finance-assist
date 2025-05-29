import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

from .models import ProcessedQuery, QueryContext, AnalysisType

class QueryProcessor:
    """Processes natural language queries into structured format for orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Symbol mappings for Asian tech stocks and common names
        self.symbol_mappings = {
            # US Tech
            'apple': 'AAPL',
            'microsoft': 'MSFT', 
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'amazon': 'AMZN',
            'meta': 'META',
            'facebook': 'META',
            'tesla': 'TSLA',
            'netflix': 'NFLX',
            'nvidia': 'NVDA',
            
            # Asian Tech Stocks
            'samsung': '005930.KS',  # Samsung Electronics (Korea)
            'tsmc': 'TSM',           # Taiwan Semiconductor (US-listed ADR)
            'taiwan semiconductor': 'TSM',
            'alibaba': 'BABA',       # Alibaba (US-listed)
            'tencent': '0700.HK',    # Tencent (Hong Kong)
            'baidu': 'BIDU',         # Baidu (US-listed)
            'jd': 'JD',              # JD.com (US-listed)
            'netease': 'NTES',       # NetEase (US-listed)
            'sony': 'SONY',          # Sony (US-listed ADR)
            'nintendo': 'NTDOY',     # Nintendo (US-listed ADR)
            'softbank': 'SFTBY',     # SoftBank (US-listed ADR)
            'line': 'LN',            # Line Corp
            'rakuten': 'RKUNY',      # Rakuten (US-listed ADR)
            'kddi': 'KDDIY',         # KDDI (US-listed ADR)
            'ntt': 'NTTYY',          # NTT (US-listed ADR)
            'sk hynix': '000660.KS', # SK Hynix (Korea)
            'lg': '066570.KS',       # LG Electronics (Korea)
            'hyundai': '005380.KS',  # Hyundai Motor (Korea)
            'kia': '000270.KS',      # Kia Motors (Korea)
            'posco': '005490.KS',    # POSCO (Korea)
            
            # Chinese Tech (US-listed)
            'nio': 'NIO',
            'xpeng': 'XPEV', 
            'li auto': 'LI',
            'pinduoduo': 'PDD',
            'bilibili': 'BILI',
            'didi': 'DIDI',
            'weibo': 'WB',
            
            # Indian Tech (US-listed ADRs)
            'infosys': 'INFY',
            'wipro': 'WIT',
            'tata': 'TTM',
            'hdfc': 'HDB',
            'icici': 'IBN'
        }
        
        # Time window mappings (in hours)
        self.time_windows = {
            'today': 24,
            'recent': 168,      # 7 days
            'this week': 168,
            'past week': 168, 
            'weekly': 168,
            'this month': 720,  # 30 days
            'monthly': 720,
            'past month': 720,
            'quarterly': 2160,  # 90 days
            'this quarter': 2160,
            'past quarter': 2160,
            'ytd': 8760,        # 365 days
            'year to date': 8760,
            'annual': 8760,
            'yearly': 8760,
            'long term': 17520, # 2 years
            'historical': 26280 # 3 years
        }
        
        # Analysis type keywords
        self.analysis_keywords = {
            AnalysisType.COMPREHENSIVE: [
                'comprehensive', 'complete', 'full analysis', 'detailed analysis',
                'in-depth', 'thorough', 'overall', 'general analysis'
            ],
            AnalysisType.PORTFOLIO: [
                'portfolio', 'holdings', 'allocation', 'diversification',
                'portfolio analysis', 'investment portfolio', 'asset allocation'
            ],
            AnalysisType.EARNINGS: [
                'earnings', 'financial results', 'quarterly results', 'revenue',
                'profit', 'income statement', 'financial performance', 'eps'
            ],
            AnalysisType.MARKET_SENTIMENT: [
                'sentiment', 'market sentiment', 'investor sentiment', 'mood',
                'outlook', 'opinion', 'perception', 'market opinion'
            ],
            AnalysisType.RISK_ASSESSMENT: [
                'risk', 'volatility', 'beta', 'risk assessment', 'downside',
                'risk analysis', 'market risk', 'investment risk', 'safety'
            ],
            AnalysisType.COMPANY_INSIGHTS: [
                'company', 'business', 'corporate', 'company analysis',
                'business insights', 'company profile', 'corporate analysis'
            ],
            AnalysisType.PEER_COMPARISON: [
                'comparison', 'vs', 'versus', 'compare', 'peer', 'competitor',
                'competitive analysis', 'peer comparison', 'benchmark'
            ],
            AnalysisType.SECTOR_ANALYSIS: [
                'sector', 'industry', 'sector analysis', 'industry analysis',
                'sector trends', 'industry trends', 'vertical'
            ]
        }
    
    async def process_query(self, raw_query: str) -> ProcessedQuery:
        """Main method to process raw query into structured format"""
        try:
            # Extract symbols
            symbols = self._extract_symbols(raw_query)
            
            # Determine analysis type
            analysis_type = self._determine_analysis_type(raw_query)
            
            # Calculate time window
            time_window = self._calculate_time_window(raw_query)
            
            # Build enhanced query
            enhanced_query = self._enhance_query(raw_query, symbols, analysis_type)
            
            # Build context
            context = self._build_context(
                symbols=symbols,
                analysis_type=analysis_type,
                time_window=time_window,
                raw_query=raw_query
            )
            
            processed_query = ProcessedQuery(
                query=enhanced_query,
                symbols=symbols,
                context=context
            )
            
            self.logger.info(f"Processed query: {len(symbols)} symbols, {analysis_type} analysis")
            return processed_query
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            # Return basic processed query on error
            return ProcessedQuery(
                query=raw_query,
                symbols=[],
                context={'analysis_type': 'comprehensive', 'error': str(e)}
            )
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract ticker symbols from query text"""
        symbols = []
        query_lower = query.lower()
        
        # First, check for direct ticker symbol matches (e.g., AAPL, MSFT)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        direct_tickers = re.findall(ticker_pattern, query.upper())
        
        # Validate direct tickers (basic validation)
        for ticker in direct_tickers:
            if len(ticker) >= 2 and ticker not in ['USD', 'EUR', 'JPY', 'KRW', 'CNY']:
                symbols.append(ticker)
        
        # Check for company name mappings
        for name, symbol in self.symbol_mappings.items():
            if name in query_lower:
                if symbol not in symbols:
                    symbols.append(symbol)
                    self.logger.info(f"Mapped '{name}' to '{symbol}'")
        
        # Remove duplicates while preserving order
        unique_symbols = []
        for symbol in symbols:
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)
        
        return unique_symbols
    
    def _determine_analysis_type(self, query: str) -> AnalysisType:
        """Determine the type of analysis based on query keywords"""
        query_lower = query.lower()
        
        # Score each analysis type based on keyword matches
        type_scores = {}
        
        for analysis_type, keywords in self.analysis_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Weight longer, more specific keywords higher
                    score += len(keyword.split())
            type_scores[analysis_type] = score
        
        # Return the analysis type with highest score
        if type_scores and max(type_scores.values()) > 0:
            best_type = max(type_scores, key=type_scores.get)
            self.logger.info(f"Determined analysis type: {best_type} (score: {type_scores[best_type]})")
            return best_type
        
        # Default to comprehensive analysis
        return AnalysisType.COMPREHENSIVE
    
    def _calculate_time_window(self, query: str) -> int:
        """Calculate appropriate time window based on query terms"""
        query_lower = query.lower()
        
        # Check for explicit time references
        for time_term, hours in self.time_windows.items():
            if time_term in query_lower:
                self.logger.info(f"Found time reference '{time_term}': {hours} hours")
                return hours
        
        # Default time window based on analysis patterns
        if any(term in query_lower for term in ['earnings', 'quarterly', 'results']):
            return 2160  # 90 days for earnings-focused queries
        elif any(term in query_lower for term in ['news', 'recent', 'latest']):
            return 168   # 7 days for news-focused queries
        elif any(term in query_lower for term in ['trend', 'performance', 'analysis']):
            return 720   # 30 days for trend analysis
        
        # Default to 7 days
        return 168
    
    def _enhance_query(self, raw_query: str, symbols: List[str], analysis_type: AnalysisType) -> str:
        """Enhance the original query with additional context"""
        enhanced_parts = [raw_query]
        
        # Add symbol context if found
        if symbols:
            symbol_text = f"focusing on {', '.join(symbols[:3])}"  # Limit to first 3 symbols
            if symbol_text.lower() not in raw_query.lower():
                enhanced_parts.append(symbol_text)
        
        # Add analysis type context
        if analysis_type != AnalysisType.COMPREHENSIVE:
            type_context = f"with emphasis on {analysis_type.value.replace('_', ' ')}"
            enhanced_parts.append(type_context)
        
        # Add market context for Asian stocks
        asian_exchanges = ['.KS', '.HK', '.T', '.SS', '.SZ']
        has_asian_stocks = any(any(exchange in symbol for exchange in asian_exchanges) for symbol in symbols)
        
        if has_asian_stocks:
            enhanced_parts.append("including Asian market context and cross-regional analysis")
        
        return ' '.join(enhanced_parts)
    
    def _build_context(self, symbols: List[str], analysis_type: AnalysisType, 
                      time_window: int, raw_query: str) -> Dict[str, Any]:
        """Build comprehensive context dictionary"""
        query_lower = raw_query.lower()
        
        # Determine additional flags based on query content
        include_peer_analysis = any(term in query_lower for term in [
            'compare', 'comparison', 'vs', 'versus', 'peer', 'competitor'
        ])
        
        include_sector_trends = any(term in query_lower for term in [
            'sector', 'industry', 'market trends', 'industry trends'
        ])
        
        # Determine if this is portfolio-focused
        portfolio_focused = analysis_type == AnalysisType.PORTFOLIO or any(
            term in query_lower for term in ['portfolio', 'allocation', 'holdings']
        )
        
        # Set confidence threshold based on query complexity
        confidence_threshold = 0.8 if len(symbols) <= 2 else 0.7
        
        # Set max results based on query scope
        max_results = 15 if include_peer_analysis or include_sector_trends else 10
        
        context = {
            'tickers': symbols,
            'analysis_type': analysis_type.value,
            'include_historical': True,
            'include_peer_analysis': include_peer_analysis,
            'include_sector_trends': include_sector_trends,
            'time_window_hours': time_window,
            'confidence_threshold': confidence_threshold,
            'max_results': max_results,
            'portfolio_focused': portfolio_focused,
            'query_complexity': self._assess_query_complexity(raw_query, symbols),
            'asian_market_focus': self._has_asian_market_focus(symbols),
            'multi_region_analysis': self._requires_multi_region_analysis(symbols)
        }
        
        return context
    
    def _assess_query_complexity(self, query: str, symbols: List[str]) -> str:
        """Assess the complexity of the query"""
        complexity_indicators = 0
        
        # Check for complex analysis terms
        complex_terms = [
            'comprehensive', 'detailed', 'in-depth', 'thorough',
            'comparison', 'correlation', 'regression', 'valuation',
            'risk-adjusted', 'peer analysis', 'sector analysis'
        ]
        
        for term in complex_terms:
            if term in query.lower():
                complexity_indicators += 1
        
        # Factor in number of symbols
        if len(symbols) > 3:
            complexity_indicators += 1
        
        # Factor in query length
        if len(query.split()) > 15:
            complexity_indicators += 1
        
        if complexity_indicators >= 3:
            return 'high'
        elif complexity_indicators >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _has_asian_market_focus(self, symbols: List[str]) -> bool:
        """Check if query focuses on Asian markets"""
        asian_exchanges = ['.KS', '.HK', '.T', '.SS', '.SZ']
        asian_symbols = [s for s in symbols if any(ex in s for ex in asian_exchanges)]
        return len(asian_symbols) > len(symbols) / 2
    
    def _requires_multi_region_analysis(self, symbols: List[str]) -> bool:
        """Check if analysis requires multi-region perspective"""
        if len(symbols) < 2:
            return False
            
        # Check for mix of US and Asian symbols
        us_symbols = [s for s in symbols if '.' not in s or s.endswith('.US')]
        asian_exchanges = ['.KS', '.HK', '.T', '.SS', '.SZ']
        asian_symbols = [s for s in symbols if any(ex in s for ex in asian_exchanges)]
        
        return len(us_symbols) > 0 and len(asian_symbols) > 0

# Factory function
def create_query_processor() -> QueryProcessor:
    """Factory function to create query processor"""
    return QueryProcessor()