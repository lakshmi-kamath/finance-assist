import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
import json

from api.models import AnalysisResult, CompanyInsight, MarketDataPoint

class ResponseFormatter:
    """Formats orchestrator results into structured, user-friendly responses"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_orchestrator_result(self, orchestrator_data: Dict[str, Any], processing_time: float) -> Dict[str, Any]:
        """Format orchestrator result for API response"""
        # Extract language brief safely
        language_brief = self._extract_language_brief(orchestrator_data)
        
        # Extract data sources used
        data_sources_used = self._extract_data_sources(orchestrator_data)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(orchestrator_data, language_brief)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score_v2(orchestrator_data)
        
        # Count data points used
        data_points_used = self._count_data_points(orchestrator_data)
        
        return {
            'analysis_result': {
                'task_type': orchestrator_data.get('task_type', 'unknown'),
                'symbols_analyzed': orchestrator_data.get('symbols_analyzed', []),
                'language_brief': language_brief,
                'executive_summary': executive_summary,
                'confidence_score': confidence_score,
                'data_points_used': data_points_used,
                'api_data': orchestrator_data.get('api_data', {}),
                'scraping_data': orchestrator_data.get('scraping_data', {}),
                'retrieval_data': orchestrator_data.get('retrieval_data', {}),
                'retrieved_context': orchestrator_data.get('retrieved_context', {}),
                'insights': orchestrator_data.get('insights', {}),
                'summary': orchestrator_data.get('summary', {}),
                'agent_results': orchestrator_data.get('agent_results', {})
            },
            'metadata': {
                'query_complexity': orchestrator_data.get('task_type', 'unknown'),
                'data_sources_used': data_sources_used,
                'analysis_scope': orchestrator_data.get('summary', {}).get('analysis_depth', 'standard'),
                'timestamp': orchestrator_data.get('timestamp', datetime.now().isoformat()),
                'processing_time_seconds': processing_time,
                'language_brief_generated': bool(language_brief and language_brief.strip()),
                'brief_length': len(language_brief) if language_brief else 0,
                'total_agents_used': len(orchestrator_data.get('summary', {}).get('data_sources_used', [])),
                'successful_data_retrievals': self._count_successful_retrievals(orchestrator_data)
            }
        }

    def _extract_language_brief(self, orchestrator_data: Dict[str, Any]) -> Optional[str]:
        """Extract language brief from orchestrator data"""
        if not orchestrator_data or not isinstance(orchestrator_data, dict):
            return None
        
        # Try multiple possible locations for the language brief
        possible_paths = [
            # Direct paths
            orchestrator_data.get('language_brief'),
            orchestrator_data.get('brief'),
            orchestrator_data.get('final_response'),
            orchestrator_data.get('response'),
            
            # From summary
            orchestrator_data.get('summary', {}).get('language_brief'),
            orchestrator_data.get('summary', {}).get('brief'),
            
            # From agent results
            orchestrator_data.get('agent_results', {}).get('language_agent', {}).get('result'),
            orchestrator_data.get('agent_results', {}).get('language_agent', {}).get('brief'),
            orchestrator_data.get('agent_results', {}).get('language_agent', {}).get('response'),
            
            # From insights
            orchestrator_data.get('insights', {}).get('language_brief'),
            orchestrator_data.get('insights', {}).get('brief'),
        ]
        
        # Check each possible path
        for brief in possible_paths:
            if brief and isinstance(brief, str) and brief.strip():
                return brief.strip()
            elif brief and isinstance(brief, dict):
                # If it's a dict, try to extract text from common keys
                for key in ['content', 'text', 'brief', 'response', 'result']:
                    if key in brief and isinstance(brief[key], str) and brief[key].strip():
                        return brief[key].strip()
        
        # Fallback: try to find any agent result that looks like a language brief
        agent_results = orchestrator_data.get('agent_results', {})
        if isinstance(agent_results, dict):
            for agent_name, agent_data in agent_results.items():
                if isinstance(agent_data, dict):
                    # Look for language-related agent results
                    if 'language' in agent_name.lower() or 'brief' in agent_name.lower():
                        for key in ['result', 'response', 'brief', 'content', 'text']:
                            if key in agent_data and isinstance(agent_data[key], str) and agent_data[key].strip():
                                return agent_data[key].strip()
                    
                    # Also check for any substantial text content
                    for key in ['result', 'response', 'brief', 'content', 'text']:
                        if key in agent_data and isinstance(agent_data[key], str):
                            content = agent_data[key].strip()
                            if len(content) > 100:  # Substantial content
                                return content
        
        return None

    def _extract_data_sources(self, orchestrator_data: Dict[str, Any]) -> List[str]:
        """Extract data sources used from orchestrator data"""
        data_sources = []
        
        if orchestrator_data.get('api_data'):
            data_sources.append('market_data_api')
        
        if orchestrator_data.get('scraping_data'):
            data_sources.append('web_scraping')
        
        if orchestrator_data.get('retrieval_data') or orchestrator_data.get('retrieved_context'):
            data_sources.append('knowledge_base')
        
        # Check agent results for data sources
        agent_results = orchestrator_data.get('agent_results', {})
        if isinstance(agent_results, dict):
            for agent_name in agent_results.keys():
                if 'api' in agent_name.lower():
                    if 'market_data_api' not in data_sources:
                        data_sources.append('market_data_api')
                elif 'scraping' in agent_name.lower() or 'web' in agent_name.lower():
                    if 'web_scraping' not in data_sources:
                        data_sources.append('web_scraping')
                elif 'retrieval' in agent_name.lower() or 'knowledge' in agent_name.lower():
                    if 'knowledge_base' not in data_sources:
                        data_sources.append('knowledge_base')
        
        return data_sources

    def _generate_executive_summary(self, orchestrator_data: Dict[str, Any], language_brief: Optional[str]) -> str:
        """Generate executive summary from orchestrator data"""
        # If we have a language brief, extract summary from it
        if language_brief:
            # Try to extract first paragraph as summary
            paragraphs = language_brief.split('\n\n')
            if paragraphs and len(paragraphs[0]) > 50:
                return self._clean_text(paragraphs[0])
        
        # Fallback: build summary from available data
        summary_parts = []
        
        # Add symbol information
        symbols = orchestrator_data.get('symbols_analyzed', [])
        if symbols:
            summary_parts.append(f"Analysis completed for {', '.join(symbols[:3])}")
            if len(symbols) > 3:
                summary_parts[-1] += f" and {len(symbols) - 3} other securities"
        
        # Add data source information
        if orchestrator_data.get('api_data'):
            summary_parts.append("incorporating current market data")
        
        if orchestrator_data.get('scraping_data'):
            summary_parts.append("enhanced with recent news and sentiment analysis")
        
        if orchestrator_data.get('retrieved_context'):
            summary_parts.append("utilizing historical context and patterns")
        
        if summary_parts:
            return '. '.join(summary_parts) + '.'
        
        return "Comprehensive financial analysis completed with available data sources."

    def _calculate_confidence_score_v2(self, orchestrator_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on data availability and quality"""
        base_score = 0.3
        
        # Add points for different data sources
        if orchestrator_data.get('api_data'):
            base_score += 0.25
        
        if orchestrator_data.get('scraping_data'):
            base_score += 0.2
        
        if orchestrator_data.get('retrieved_context') or orchestrator_data.get('retrieval_data'):
            base_score += 0.15
        
        # Add points for successful agent results
        agent_results = orchestrator_data.get('agent_results', {})
        if isinstance(agent_results, dict):
            successful_agents = 0
            total_agents = len(agent_results)
            
            for agent_name, agent_data in agent_results.items():
                if isinstance(agent_data, dict):
                    # Check if agent was successful
                    if (agent_data.get('success', True) and 
                        (agent_data.get('result') or agent_data.get('response') or agent_data.get('data'))):
                        successful_agents += 1
            
            if total_agents > 0:
                success_rate = successful_agents / total_agents
                base_score += success_rate * 0.1
        
        # Cap at 1.0
        return min(base_score, 1.0)

    def _count_data_points(self, orchestrator_data: Dict[str, Any]) -> int:
        """Count total data points used in analysis"""
        data_points = 0
        
        # Count API data points
        api_data = orchestrator_data.get('api_data', {})
        if isinstance(api_data, dict):
            for symbol_data in api_data.values():
                if isinstance(symbol_data, dict):
                    data_points += len(symbol_data)
                else:
                    data_points += 1
        
        # Count scraping data points
        scraping_data = orchestrator_data.get('scraping_data', {})
        if isinstance(scraping_data, dict):
            for symbol_data in scraping_data.values():
                if isinstance(symbol_data, dict):
                    # Count news items, sentiment data, etc.
                    data_points += len(symbol_data.get('news', []))
                    if symbol_data.get('sentiment'):
                        data_points += 1
                else:
                    data_points += 1
        
        # Count retrieval data points
        retrieval_data = orchestrator_data.get('retrieval_data', {})
        if isinstance(retrieval_data, dict):
            data_points += len(retrieval_data)
        
        retrieved_context = orchestrator_data.get('retrieved_context', {})
        if isinstance(retrieved_context, dict):
            data_points += len(retrieved_context)
        
        return data_points

    def _count_successful_retrievals(self, orchestrator_data: Dict[str, Any]) -> int:
        """Count successful data retrievals"""
        successful_retrievals = 0
        
        # Count based on available data
        if orchestrator_data.get('api_data'):
            successful_retrievals += 1
        
        if orchestrator_data.get('scraping_data'):
            successful_retrievals += 1
        
        if orchestrator_data.get('retrieval_data') or orchestrator_data.get('retrieved_context'):
            successful_retrievals += 1
        
        # Count successful agent executions
        agent_results = orchestrator_data.get('agent_results', {})
        if isinstance(agent_results, dict):
            for agent_data in agent_results.values():
                if isinstance(agent_data, dict) and agent_data.get('success', True):
                    if agent_data.get('result') or agent_data.get('response') or agent_data.get('data'):
                        successful_retrievals += 1
        
        return successful_retrievals

    def _build_analysis_result(self, final_results: Dict[str, Any], 
                             language_brief: Dict[str, Any], 
                             metadata: Dict[str, Any]) -> AnalysisResult:
        """Build structured analysis result"""
        
        # Extract executive summary
        executive_summary = self._extract_executive_summary(language_brief, final_results)
        
        # Build market analysis
        market_analysis = self._build_market_analysis(final_results, language_brief)
        
        # Extract company insights
        company_insights = self._extract_company_insights(final_results, language_brief)
        
        # Extract recommendations
        recommendations = self._extract_recommendations(language_brief, final_results)
        
        # Build risk assessment
        risk_assessment = self._build_risk_assessment(final_results, language_brief)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(metadata, final_results)
        
        # Get data points used
        data_points_used = metadata.get('brief_data_points', 0) + len(final_results.get('api_results', [])) + len(final_results.get('scraping_results', []))
        
        # Check if portfolio focused
        portfolio_focused = metadata.get('brief_portfolio_focused', False) or 'portfolio' in metadata.get('query', '').lower()
        
        return AnalysisResult(
            executive_summary=executive_summary,
            market_analysis=market_analysis,
            company_insights=company_insights,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            confidence_score=confidence_score,
            data_points_used=data_points_used,
            portfolio_focused=portfolio_focused
        )
    
    def _extract_executive_summary(self, language_brief: Dict[str, Any], 
                                 final_results: Dict[str, Any]) -> str:
        """Extract and format executive summary"""
        
        # Try to get from language brief first
        if language_brief and 'summary' in language_brief:
            summary = language_brief['summary']
            if isinstance(summary, str) and len(summary) > 50:
                return self._clean_text(summary)
        
        # Try to get from language brief content
        if language_brief and 'content' in language_brief:
            content = language_brief['content']
            if isinstance(content, str):
                # Extract first paragraph as summary
                paragraphs = content.split('\n\n')
                if paragraphs and len(paragraphs[0]) > 50:
                    return self._clean_text(paragraphs[0])
        
        # Fallback: build summary from final results
        summary_parts = []
        
        # Add symbol information
        symbols = final_results.get('symbols_analyzed', [])
        if symbols:
            summary_parts.append(f"Analysis of {', '.join(symbols[:3])}")
            if len(symbols) > 3:
                summary_parts[-1] += f" and {len(symbols) - 3} other securities"
        
        # Add key findings
        api_results = final_results.get('api_data', {})
        if api_results:
            summary_parts.append("incorporating current market data and financial metrics")
        
        scraping_results = final_results.get('scraping_data', {})
        if scraping_results:
            summary_parts.append("enhanced with recent news and market sentiment")
        
        if summary_parts:
            return '. '.join(summary_parts) + '.'
        
        return "Comprehensive financial analysis completed with available data sources."
    
    def _build_market_analysis(self, final_results: Dict[str, Any], 
                             language_brief: Dict[str, Any]) -> Dict[str, Any]:
        """Build market analysis section"""
        market_analysis = {}
        
        # Extract market data from API results
        api_data = final_results.get('api_data', {})
        if api_data:
            market_analysis['current_data'] = self._extract_market_metrics(api_data)
        
        # Extract market trends from language brief
        if language_brief and 'market_trends' in language_brief:
            market_analysis['trends'] = language_brief['market_trends']
        
        # Extract sector information
        scraping_data = final_results.get('scraping_data', {})
        if scraping_data:
            market_analysis['sentiment'] = self._extract_sentiment_data(scraping_data)
        
        # Add retrieved context insights
        retrieved_context = final_results.get('retrieved_context', {})
        if retrieved_context:
            market_analysis['historical_context'] = self._extract_historical_insights(retrieved_context)
        
        return market_analysis
    
    def _extract_company_insights(self, final_results: Dict[str, Any], 
                                language_brief: Dict[str, Any]) -> List[CompanyInsight]:
        """Extract company-specific insights"""
        insights = []
        
        # Get symbols from results
        symbols = final_results.get('symbols_analyzed', [])
        api_data = final_results.get('api_data', {})
        
        for symbol in symbols:
            # Build company insight
            insight_data = {
                'symbol': symbol,
                'company_name': self._get_company_name(symbol, api_data),
                'key_metrics': {},
                'recent_news': [],
                'sentiment_score': None
            }
            
            # Extract metrics from API data
            if api_data and symbol in api_data:
                symbol_data = api_data[symbol]
                insight_data['key_metrics'] = self._extract_key_metrics(symbol_data)
                
                # Extract basic info
                if 'sector' in symbol_data:
                    insight_data['sector'] = symbol_data['sector']
                if 'market_cap' in symbol_data:
                    insight_data['market_cap'] = symbol_data['market_cap']
                if 'pe_ratio' in symbol_data:
                    insight_data['pe_ratio'] = symbol_data['pe_ratio']
            
            # Extract news and sentiment from scraping data
            scraping_data = final_results.get('scraping_data', {})
            if scraping_data and symbol in scraping_data:
                symbol_scraping = scraping_data[symbol]
                if 'news' in symbol_scraping:
                    insight_data['recent_news'] = symbol_scraping['news'][:5]  # Top 5 news items
                if 'sentiment' in symbol_scraping:
                    insight_data['sentiment_score'] = symbol_scraping['sentiment']
            
            insights.append(CompanyInsight(**insight_data))
        
        return insights
    
    def _extract_recommendations(self, language_brief: Dict[str, Any], 
                               final_results: Dict[str, Any]) -> List[str]:
        """Extract actionable recommendations"""
        recommendations = []
        
        # Try to get from language brief
        if language_brief:
            if 'recommendations' in language_brief:
                recs = language_brief['recommendations']
                if isinstance(recs, list):
                    recommendations.extend([self._clean_text(r) for r in recs])
                elif isinstance(recs, str):
                    # Parse recommendations from text
                    rec_lines = self._parse_recommendations_from_text(recs)
                    recommendations.extend(rec_lines)
            
            elif 'content' in language_brief:
                # Extract recommendations from content
                content = language_brief['content']
                rec_lines = self._parse_recommendations_from_text(content)
                recommendations.extend(rec_lines)
        
        # Fallback: generate basic recommendations from data
        if not recommendations:
            recommendations = self._generate_fallback_recommendations(final_results)
        
        return recommendations[:5]  # Limit to top 5
    
    def _build_risk_assessment(self, final_results: Dict[str, Any], 
                             language_brief: Dict[str, Any]) -> Dict[str, Any]:
        """Build risk assessment section"""
        risk_assessment = {}
        
        # Extract volatility data from API results
        api_data = final_results.get('api_data', {})
        if api_data:
            risk_metrics = {}
            for symbol, data in api_data.items():
                if isinstance(data, dict):
                    symbol_risk = {}
                    if 'volatility' in data:
                        symbol_risk['volatility'] = data['volatility']
                    if 'beta' in data:
                        symbol_risk['beta'] = data['beta']
                    if 'var' in data:
                        symbol_risk['value_at_risk'] = data['var']
                    
                    if symbol_risk:
                        risk_metrics[symbol] = symbol_risk
            
            if risk_metrics:
                risk_assessment['metrics'] = risk_metrics
        
        # Extract risk insights from language brief
        if language_brief and 'risk_factors' in language_brief:
            risk_assessment['qualitative_factors'] = language_brief['risk_factors']
        
        # Add overall risk level assessment
        risk_assessment['overall_risk_level'] = self._assess_overall_risk(api_data, language_brief)
        
        return risk_assessment
    
    def _calculate_confidence_score(self, metadata: Dict[str, Any], 
                                  final_results: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and completeness"""
        base_score = 0.5
        
        # Add points for data sources
        api_data = final_results.get('api_data', {})
        if api_data:
            base_score += 0.2
        
        scraping_data = final_results.get('scraping_data', {})
        if scraping_data:
            base_score += 0.15
        
        retrieved_context = final_results.get('retrieved_context', {})
        if retrieved_context:
            base_score += 0.1
        
        # Adjust based on metadata
        brief_confidence = metadata.get('brief_confidence', 0.5)
        base_score = (base_score + brief_confidence) / 2
        
        # Cap at 1.0
        return min(base_score, 1.0)
    
    # Helper methods for data extraction
    
    def _extract_market_metrics(self, api_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market metrics from API data"""
        metrics = {}
        
        for symbol, data in api_data.items():
            if isinstance(data, dict):
                symbol_metrics = {}
                
                # Common financial metrics
                for key in ['price', 'change', 'change_percent', 'volume', 'market_cap', 'pe_ratio']:
                    if key in data:
                        symbol_metrics[key] = data[key]
                
                if symbol_metrics:
                    metrics[symbol] = symbol_metrics
        
        return metrics
    
    def _extract_sentiment_data(self, scraping_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment data from scraping results"""
        sentiment_data = {}
        
        for symbol, data in scraping_data.items():
            if isinstance(data, dict) and 'sentiment' in data:
                sentiment_data[symbol] = {
                    'score': data['sentiment'],
                    'news_count': len(data.get('news', []))
                }
        
        return sentiment_data
    
    def _extract_historical_insights(self, retrieved_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract historical insights from retrieved context"""
        insights = {}
        
        if 'historical_patterns' in retrieved_context:
            insights['patterns'] = retrieved_context['historical_patterns']
        
        if 'seasonal_trends' in retrieved_context:
            insights['seasonal'] = retrieved_context['seasonal_trends']
        
        return insights
    
    def _get_company_name(self, symbol: str, api_data: Dict[str, Any]) -> str:
        """Get company name from symbol"""
        if api_data and symbol in api_data:
            data = api_data[symbol]
            if isinstance(data, dict) and 'company_name' in data:
                return data['company_name']
        
        # Fallback mappings for common symbols
        name_mappings = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'TSLA': 'Tesla Inc.',
            'TSM': 'Taiwan Semiconductor Manufacturing Company',
            'BABA': 'Alibaba Group Holding Limited',
            'SONY': 'Sony Corporation'
        }
        
        return name_mappings.get(symbol, symbol)
    
    def _extract_key_metrics(self, symbol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key financial metrics"""
        metrics = {}
        
        metric_keys = [
            'price', 'change', 'change_percent', 'volume', 'market_cap',
            'pe_ratio', 'eps', 'dividend_yield', 'beta', 'volatility'
        ]
        
        for key in metric_keys:
            if key in symbol_data:
                metrics[key] = symbol_data[key]
        
        return metrics
    
    def _parse_recommendations_from_text(self, text: str) -> List[str]:
        """Parse recommendations from text content"""
        recommendations = []
        
        # Look for recommendation patterns
        patterns = [
            r'recommend(?:ation)?s?[:\s]+(.+?)(?:\n|$)',
            r'suggest(?:ion)?s?[:\s]+(.+?)(?:\n|$)',
            r'(?:should|consider)[:\s]+(.+?)(?:\n|$)',
            r'action[:\s]+(.+?)(?:\n|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                clean_rec = self._clean_text(match.strip())
                if len(clean_rec) > 10:  # Filter out very short matches
                    recommendations.append(clean_rec)
        
        # Look for numbered/bulleted lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^[\d\-\*\•]\s*', line):
                clean_line = re.sub(r'^[\d\-\*\•]\s*', '', line).strip()
                if len(clean_line) > 10:
                    recommendations.append(self._clean_text(clean_line))
        
        return recommendations
    
    def _generate_fallback_recommendations(self, final_results: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations when none are found"""
        recommendations = []
        
        symbols = final_results.get('symbols_analyzed', [])
        api_data = final_results.get('api_data', {})
        
        if symbols and api_data:
            recommendations.append("Monitor current market positions and consider rebalancing based on recent performance metrics.")
            recommendations.append("Review risk exposure and ensure portfolio diversification aligns with investment objectives.")
            
            if len(symbols) > 1:
                recommendations.append("Consider correlation analysis between holdings to optimize portfolio risk-return profile.")
        
        if not recommendations:
            recommendations.append("Conduct regular portfolio review and maintain disciplined investment approach.")
        
        return recommendations
    
    def _assess_overall_risk(self, api_data: Dict[str, Any], 
                           language_brief: Dict[str, Any]) -> str:
        """Assess overall risk level"""
        if not api_data:
            return "moderate"
        
        high_risk_indicators = 0
        total_symbols = len(api_data)
        
        for symbol, data in api_data.items():
            if isinstance(data, dict):
                # Check volatility
                if data.get('volatility', 0) > 0.3:
                    high_risk_indicators += 1
                
                # Check beta
                if data.get('beta', 1.0) > 1.5:
                    high_risk_indicators += 1
        
        risk_ratio = high_risk_indicators / max(total_symbols * 2, 1)  # Max 2 indicators per symbol
        
        if risk_ratio > 0.6:
            return "high"
        elif risk_ratio > 0.3:
            return "moderate"
        else:
            return "low"
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text content"""
        if not isinstance(text, str):
            return str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove brackets
        text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses with content
        
        return text.strip()
    
    def _format_metadata(self, metadata: Dict[str, Any], 
                        processing_time: float = None) -> Dict[str, Any]:
        """Format metadata for response"""
        formatted_metadata = {
            'query_complexity': metadata.get('query_complexity', 'medium'),
            'data_sources_used': [],
            'analysis_scope': metadata.get('analysis_scope', 'comprehensive'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add data source information
        if metadata.get('api_data_available'):
            formatted_metadata['data_sources_used'].append('market_data_api')
        
        if metadata.get('scraping_data_available'):
            formatted_metadata['data_sources_used'].append('web_scraping')
        
        if metadata.get('knowledge_base_used'):
            formatted_metadata['data_sources_used'].append('knowledge_base')
        
        # Add processing information
        if processing_time:
            formatted_metadata['processing_time_seconds'] = processing_time
        
        return formatted_metadata
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response format"""
        return {
            'success': False,
            'error': error_message,
            'analysis_result': None,
            'metadata': {
                'error_timestamp': datetime.now().isoformat(),
                'error_type': 'formatting_error'
            },
            'timestamp': datetime.now().isoformat()
        }
# Factory function
def create_response_formatter() -> ResponseFormatter:
    """Factory function to create response formatter"""
    return ResponseFormatter()