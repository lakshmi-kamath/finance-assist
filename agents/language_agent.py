import asyncio
import json
import logging
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseType(Enum):
    BRIEF = "brief"
    DETAILED = "detailed"
    ALERT = "alert"
    ANALYSIS = "analysis"
    SUMMARY = "summary"

class ContextType(Enum):
    MORNING_BRIEF = "morning_brief"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ANALYSIS = "risk_analysis"
    MARKET_ANALYSIS = "market_analysis"
    EARNINGS_ANALYSIS = "earnings_analysis"
    GENERAL_QUERY = "general_query"

@dataclass
class DataSource:
    """Represents a single data source with metadata"""
    content: str
    source_type: str  # 'real_time', 'historical', 'news', 'earnings'
    timestamp: datetime
    relevance_score: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class AnalysisContext:
    """Context information for generating responses"""
    user_query: str
    context_type: ContextType
    response_type: ResponseType
    user_preferences: Dict[str, Any]
    portfolio_context: Optional[Dict[str, Any]] = None
    time_horizon: str = "short_term"  # short_term, medium_term, long_term

class LanguageAgent:
    """
    Language Agent responsible for synthesizing financial data using Gemini LLM
    with rate limiting and fallback capabilities
    """
    
    def __init__(self, 
                 gemini_api_key: str,
                 model_name: str = "gemini-1.5-flash",  # Use flash for lower rate limits
                 max_tokens: int = 1000,  # Reduced for free tier
                 temperature: float = 0.3,
                 enable_retries: bool = True,
                 max_retries: int = 3):
        """
        Initialize Language Agent
        
        Args:
            gemini_api_key: Google Gemini API key
            model_name: Gemini model to use (flash is better for free tier)
            max_tokens: Maximum tokens for response
            temperature: LLM temperature setting
            enable_retries: Enable automatic retries on rate limits
            max_retries: Maximum number of retries
        """
        genai.configure(api_key=gemini_api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enable_retries = enable_retries
        self.max_retries = max_retries
        
        # Rate limiting tracking
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Minimum 2 seconds between requests
        
        # Configure generation settings for free tier
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=0.8,  # Reduced for more focused responses
            top_k=32,   # Reduced for efficiency
            max_output_tokens=max_tokens,
        )
        
        # Safety settings for financial content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Model fallback hierarchy (from most to least capable)
        self.model_fallbacks = [
            "gemini-1.5-flash",
            "gemini-1.0-pro"
        ]
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        # Context-specific prompts
        self.system_prompts = {
            ContextType.MORNING_BRIEF: self._get_morning_brief_prompt(),
            ContextType.PORTFOLIO_UPDATE: self._get_portfolio_update_prompt(),
            ContextType.RISK_ANALYSIS: self._get_risk_analysis_prompt(),
            ContextType.MARKET_ANALYSIS: self._get_market_analysis_prompt(),
            ContextType.EARNINGS_ANALYSIS: self._get_earnings_analysis_prompt(),
            ContextType.GENERAL_QUERY: self._get_general_query_prompt()
        }
    
    async def synthesize_response(self, 
                                data_sources: List[DataSource],
                                context: AnalysisContext) -> Dict[str, Any]:
        """
        Main method to synthesize financial data into coherent response
        
        Args:
            data_sources: List of data sources with content and metadata
            context: Analysis context and user preferences
            
        Returns:
            Dict containing synthesized response and metadata
        """
        try:
            # Step 1: Prepare and prioritize data sources
            processed_sources = self._process_data_sources(data_sources, context)
            
            # Step 2: Build context-aware prompt
            prompt = self._build_prompt(processed_sources, context)
            
            # Step 3: Generate response using LLM
            response = await self._generate_llm_response(prompt, context)
            
            # Step 4: Post-process and enhance response
            final_response = self._post_process_response(response, processed_sources, context)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in synthesize_response: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _process_data_sources(self, 
                            data_sources: List[DataSource],
                            context: AnalysisContext) -> Dict[str, Any]:
        """Process and prioritize data sources based on context"""
        
        # Separate data by type and recency
        real_time_data = []
        historical_data = []
        news_data = []
        earnings_data = []
        
        current_time = datetime.now()
        
        for source in data_sources:
            # Calculate time decay factor
            time_diff = current_time - source.timestamp
            if time_diff.days == 0:
                freshness_weight = 1.0
            elif time_diff.days <= 7:
                freshness_weight = 0.8
            elif time_diff.days <= 30:
                freshness_weight = 0.6
            else:
                freshness_weight = 0.4
            
            # Adjust relevance based on freshness and confidence
            adjusted_score = source.relevance_score * freshness_weight * source.confidence
            
            source_data = {
                'content': source.content,
                'score': adjusted_score,
                'timestamp': source.timestamp,
                'metadata': source.metadata
            }
            
            if source.source_type == 'real_time':
                real_time_data.append(source_data)
            elif source.source_type == 'historical':
                historical_data.append(source_data)
            elif source.source_type == 'news':
                news_data.append(source_data)
            elif source.source_type == 'earnings':
                earnings_data.append(source_data)
        
        # Sort each category by adjusted score
        for data_list in [real_time_data, historical_data, news_data, earnings_data]:
            data_list.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'real_time': real_time_data,
            'historical': historical_data,
            'news': news_data,
            'earnings': earnings_data,
            'total_sources': len(data_sources)
        }
    
    def _build_prompt(self, 
                     processed_sources: Dict[str, Any],
                     context: AnalysisContext) -> str:
        """Build context-aware prompt for LLM - optimized for free tier"""
        
        # Get system prompt based on context type
        system_prompt = self.system_prompts.get(context.context_type, 
                                               self.system_prompts[ContextType.GENERAL_QUERY])
        
        # Build concise data context for free tier
        data_context = self._build_concise_data_context(processed_sources, context)
        
        # Build user context
        user_context = self._build_user_context(context)
        
        # More concise prompt for free tier limits
        prompt = f"""
{system_prompt}

CONTEXT: {user_context}

DATA: {data_context}

QUERY: {context.user_query}

Provide a {context.response_type.value} response with key insights and specific data points.
"""
        
        return prompt
    
    def _build_concise_data_context(self, 
                                  processed_sources: Dict[str, Any],
                                  context: AnalysisContext) -> str:
        """Build concise data context optimized for token limits"""
        
        data_parts = []
        
        # Prioritize real-time data (most important)
        if processed_sources['real_time']:
            rt_data = processed_sources['real_time'][0]['content']  # Just the top one
            data_parts.append(f"Live: {rt_data}")
        
        # Add news if relevant and space allows
        if processed_sources['news'] and len(data_parts) < 2:
            news_data = processed_sources['news'][0]['content']
            data_parts.append(f"News: {news_data}")
        
        # Add earnings if relevant to context
        if processed_sources['earnings'] and context.context_type == ContextType.EARNINGS_ANALYSIS:
            earnings_data = processed_sources['earnings'][0]['content']
            data_parts.append(f"Earnings: {earnings_data}")
        
        return " | ".join(data_parts) if data_parts else "No specific data available."
    
    def _build_data_context(self, 
                           processed_sources: Dict[str, Any],
                           context: AnalysisContext) -> str:
        """Build structured data context for the prompt"""
        
        data_sections = []
        
        # Real-time data (highest priority)
        if processed_sources['real_time']:
            real_time_section = "REAL-TIME MARKET DATA:\n"
            for i, source in enumerate(processed_sources['real_time'][:3]):  # Top 3
                real_time_section += f"• {source['content']}\n"
            data_sections.append(real_time_section)
        
        # Recent news and events
        if processed_sources['news']:
            news_section = "RECENT NEWS & EVENTS:\n"
            for i, source in enumerate(processed_sources['news'][:2]):  # Top 2
                news_section += f"• {source['content']}\n"
            data_sections.append(news_section)
        
        # Earnings data (if relevant)
        if processed_sources['earnings'] and context.context_type in [
            ContextType.EARNINGS_ANALYSIS, ContextType.PORTFOLIO_UPDATE
        ]:
            earnings_section = "EARNINGS DATA:\n"
            for i, source in enumerate(processed_sources['earnings'][:2]):  # Top 2
                earnings_section += f"• {source['content']}\n"
            data_sections.append(earnings_section)
        
        # Historical context (for perspective)
        if processed_sources['historical']:
            historical_section = "HISTORICAL CONTEXT:\n"
            for i, source in enumerate(processed_sources['historical'][:2]):  # Top 2
                historical_section += f"• {source['content']}\n"
            data_sections.append(historical_section)
        
        return "\n\n".join(data_sections)
    
    async def _rate_limit_wait(self):
        """Implement rate limiting to avoid quota issues"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    async def _generate_llm_response_with_retry(self, 
                                             prompt: str,
                                             context: AnalysisContext,
                                             model_name: str = None) -> str:
        """Generate response with retry logic and fallback models"""
        
        current_model = model_name or self.model_name
        
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                await self._rate_limit_wait()
                
                # Adjust parameters based on response type
                temp = self.temperature
                max_tokens = self.max_tokens
                
                if context.response_type == ResponseType.BRIEF:
                    max_tokens = min(300, self.max_tokens)  # Further reduced
                    temp = 0.2
                elif context.response_type == ResponseType.DETAILED:
                    max_tokens = min(800, self.max_tokens)  # Reduced for free tier
                    temp = 0.4
                elif context.response_type == ResponseType.ALERT:
                    max_tokens = min(200, self.max_tokens)
                    temp = 0.1
                
                # Create model instance for this attempt
                current_config = genai.types.GenerationConfig(
                    temperature=temp,
                    top_p=0.8,
                    top_k=32,
                    max_output_tokens=max_tokens,
                )
                
                model = genai.GenerativeModel(
                    model_name=current_model,
                    generation_config=current_config,
                    safety_settings=self.safety_settings
                )
                
                # Generate response
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: model.generate_content(prompt)
                )
                
                # Check if response was successful
                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
                else:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                        logger.warning(f"Response blocked: {response.prompt_feedback.block_reason}")
                        return "I'm unable to provide a response due to content safety restrictions. Please rephrase your query."
                    else:
                        logger.warning("Empty response from Gemini")
                        if attempt < self.max_retries:
                            continue
                        return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limit errors
                if "429" in error_str or "quota" in error_str.lower():
                    if attempt < self.max_retries:
                        # Extract retry delay if available
                        retry_delay = self._extract_retry_delay(error_str)
                        wait_time = retry_delay or (2 ** attempt) + random.uniform(1, 3)
                        
                        logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                        
                        # Try fallback models
                        if attempt >= 1 and len(self.model_fallbacks) > attempt - 1:
                            current_model = self.model_fallbacks[min(attempt - 1, len(self.model_fallbacks) - 1)]
                            logger.info(f"Falling back to model: {current_model}")
                        
                        continue
                    else:
                        logger.error(f"Max retries exceeded for rate limiting: {error_str}")
                        return self._generate_rate_limit_fallback_response(context)
                
                # Handle other errors
                elif attempt < self.max_retries:
                    wait_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning(f"Error on attempt {attempt + 1}: {error_str}. Retrying in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries exceeded: {error_str}")
                    raise
        
        return "I apologize, but I'm currently unable to process your request due to technical difficulties. Please try again later."
    
    def _extract_retry_delay(self, error_str: str) -> Optional[float]:
        """Extract retry delay from error message"""
        try:
            import re
            match = re.search(r'retry_delay.*?seconds:\s*(\d+)', error_str)
            if match:
                return float(match.group(1))
        except:
            pass
        return None
    
    def _generate_rate_limit_fallback_response(self, context: AnalysisContext) -> str:
        """Generate a fallback response when rate limits are exceeded"""
        
        fallback_responses = {
            ContextType.MORNING_BRIEF: "I'm currently experiencing high demand. For your morning brief, please check financial news sources like Bloomberg, CNBC, or Reuters, or use tools like Yahoo Finance for the latest market updates.",
            ContextType.PORTFOLIO_UPDATE: "I'm temporarily unavailable for detailed portfolio analysis. Please monitor your holdings directly, check with your broker, or consider using portfolio tracking apps or platforms for updates.",
            ContextType.RISK_ANALYSIS: "Risk analysis service is temporarily limited. Please review your positions using tools like Value-at-Risk (VaR) calculators, portfolio stress testing, or consult your financial advisor for tailored advice.",
            ContextType.MARKET_ANALYSIS: "Market analysis is currently limited due to high demand. Please check platforms like Bloomberg, TradingView, or Yahoo Finance for current market conditions.",
            ContextType.EARNINGS_ANALYSIS: "Earnings analysis is temporarily unavailable. Please check company investor relations pages for the latest earnings information or use financial data aggregators like Bloomberg or Yahoo Finance.",
            ContextType.GENERAL_QUERY: "I'm currently experiencing high usage. For general financial queries, consider visiting forums like Reddit's r/investing, Seeking Alpha, or Investopedia for reliable information."
        }
        
        return fallback_responses.get(context.context_type, fallback_responses[ContextType.GENERAL_QUERY])
    
    def _build_user_context(self, context: AnalysisContext) -> str:
        """Build user-specific context"""
            
        user_context_parts = []
            
        # Add portfolio context if available
        if context.portfolio_context:
            portfolio_info = f"Portfolio Holdings: {context.portfolio_context.get('holdings', 'N/A')}"
            risk_tolerance = f"Risk Tolerance: {context.portfolio_context.get('risk_tolerance', 'Moderate')}"
            user_context_parts.extend([portfolio_info, risk_tolerance])
            
        # Add user preferences
        if context.user_preferences:
            prefs = []
            for key, value in context.user_preferences.items():
                prefs.append(f"{key}: {value}")
            if prefs:
                user_context_parts.append(f"Preferences: {', '.join(prefs)}")
            
            # Add time horizon
        user_context_parts.append(f"Time Horizon: {context.time_horizon}")
            
        return "\n".join(user_context_parts) if user_context_parts else "No specific user context available."
    
    async def _generate_llm_response(self, 
                                   prompt: str,
                                   context: AnalysisContext) -> str:
        """Generate response using Gemini LLM"""
        
        try:
            # Adjust parameters based on response type
            temp = self.temperature
            max_tokens = self.max_tokens
            
            if context.response_type == ResponseType.BRIEF:
                max_tokens = min(500, self.max_tokens)
                temp = 0.2
            elif context.response_type == ResponseType.DETAILED:
                max_tokens = self.max_tokens
                temp = 0.4
            elif context.response_type == ResponseType.ALERT:
                max_tokens = min(300, self.max_tokens)
                temp = 0.1
            
            # Update generation config for this request
            current_config = genai.types.GenerationConfig(
                temperature=temp,
                top_p=0.95,
                top_k=64,
                max_output_tokens=max_tokens,
            )
            
            # Generate response using Gemini
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=current_config,
                    safety_settings=self.safety_settings
                )
            )
            
            # Check if response was blocked
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                # Handle blocked or empty responses
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                    logger.warning(f"Response blocked: {response.prompt_feedback.block_reason}")
                    return "I'm unable to provide a response due to content safety restrictions. Please rephrase your query."
                else:
                    logger.warning("Empty response from Gemini")
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            raise
    
    def _post_process_response(self, 
                             response: str,
                             processed_sources: Dict[str, Any],
                             context: AnalysisContext) -> Dict[str, Any]:
        """Post-process and enhance the LLM response"""
        
        # Extract key metrics mentioned in response
        key_metrics = self._extract_key_metrics(response)
        
        # Generate confidence score based on data quality
        confidence_score = self._calculate_confidence_score(processed_sources)
        
        # Add metadata
        response_metadata = {
            'generated_at': datetime.now().isoformat(),
            'context_type': context.context_type.value,
            'response_type': context.response_type.value,
            'data_sources_used': processed_sources['total_sources'],
            'confidence_score': confidence_score,
            'key_metrics': key_metrics
        }
        
        return {
            'response': response,
            'metadata': response_metadata,
            'confidence_score': confidence_score,
            'sources_summary': self._generate_sources_summary(processed_sources)
        }
    
    def _extract_key_metrics(self, response: str) -> List[str]:
        """Extract key financial metrics mentioned in the response"""
        # Simple regex-based extraction (could be enhanced with NLP)
        import re
        
        metrics = []
        
        # Look for percentage changes
        percentages = re.findall(r'[-+]?\d+\.?\d*%', response)
        metrics.extend(percentages)
        
        # Look for dollar amounts
        dollar_amounts = re.findall(r'\$[\d,]+\.?\d*[KMB]?', response)
        metrics.extend(dollar_amounts)
        
        return list(set(metrics))  # Remove duplicates
    
    def _calculate_confidence_score(self, processed_sources: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and recency"""
        
        total_sources = processed_sources['total_sources']
        if total_sources == 0:
            return 0.0
        
        # Weight different source types
        weights = {
            'real_time': 0.4,
            'news': 0.3,
            'earnings': 0.2,
            'historical': 0.1
        }
        
        weighted_score = 0.0
        for source_type, sources in processed_sources.items():
            if source_type == 'total_sources':
                continue
                
            if sources:
                avg_score = sum(s['score'] for s in sources) / len(sources)
                weighted_score += weights.get(source_type, 0.1) * avg_score
        
        return min(weighted_score, 1.0)
    
    def _generate_sources_summary(self, processed_sources: Dict[str, Any]) -> Dict[str, int]:
        """Generate summary of sources used"""
        return {
            'real_time': len(processed_sources['real_time']),
            'historical': len(processed_sources['historical']),
            'news': len(processed_sources['news']),
            'earnings': len(processed_sources['earnings']),
            'total': processed_sources['total_sources']
        }
    
    def _generate_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'response': "I apologize, but I encountered an issue processing your request. Please try again or rephrase your question.",
            'metadata': {
                'error': error_msg,
                'generated_at': datetime.now().isoformat(),
                'confidence_score': 0.0
            },
            'confidence_score': 0.0,
            'sources_summary': {'total': 0}
        }
    
    # System prompts for different contexts - optimized for conciseness
    def _get_morning_brief_prompt(self) -> str:
        return """You are a financial analyst providing morning market briefs. 
Focus on key overnight developments and market movements. Be concise and actionable."""
    
    def _get_portfolio_update_prompt(self) -> str:
        return """You are a portfolio advisor providing position updates. 
Focus on performance, key changes, and immediate actions needed. Be specific and practical."""
    
    def _get_risk_analysis_prompt(self) -> str:
        return """You are a risk analyst identifying threats and opportunities. 
Focus on quantifiable risks and mitigation strategies. Provide clear metrics."""
    
    def _get_market_analysis_prompt(self) -> str:
        return """You are a market strategist analyzing trends and opportunities. 
Focus on key drivers, technical levels, and sector movements. Be insightful and forward-looking."""
    
    def _get_earnings_analysis_prompt(self) -> str:
        return """You are an equity analyst reviewing earnings results. 
Focus on surprises, guidance, and stock implications. Provide specific financial metrics."""
    
    def _get_general_query_prompt(self) -> str:
        return """You are a financial advisor answering user questions. 
Provide accurate, helpful information tailored to the user's needs. Be clear and professional."""


# Lightweight version for free tier testing
class LightweightLanguageAgent:
    """
    Simplified Language Agent optimized for free tier usage
    """
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.last_request_time = 0
        
    async def simple_query(self, query: str, context_data: str = "") -> str:
        """Simple query method with minimal token usage"""
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_request_time < 3.0:  # 3 second minimum
            wait_time = 3.0 - (current_time - self.last_request_time)
            await asyncio.sleep(wait_time)
        
        prompt = f"""Financial Query: {query}
Context: {context_data}

Provide a brief, focused response (max 100 words)."""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=150,
                        temperature=0.3
                    )
                )
            )
            
            self.last_request_time = time.time()
            
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "Unable to generate response at this time."
                
        except Exception as e:
            if "429" in str(e):
                return "Rate limit reached. Please wait a moment and try again."
            else:
                return f"Error: {str(e)}"


# Example usage and testing
async def test_language_agent():
    """Test the Language Agent with sample data"""
    
    # Sample data sources
    sample_sources = [
        DataSource(
            content="AAPL down 2.3% in pre-market trading following mixed earnings report",
            source_type="real_time",
            timestamp=datetime.now(),
            relevance_score=0.9,
            confidence=0.95,
            metadata={'symbol': 'AAPL', 'price_change': -2.3}
        ),
        DataSource(
            content="Tech sector showing weakness due to rising interest rate concerns",
            source_type="news",
            timestamp=datetime.now() - timedelta(hours=2),
            relevance_score=0.8,
            confidence=0.85,
            metadata={'sector': 'Technology'}
        )
    ]
    
    # Sample context
    context = AnalysisContext(
        user_query="What's happening with my tech holdings this morning?",
        context_type=ContextType.MORNING_BRIEF,
        response_type=ResponseType.BRIEF,
        user_preferences={'focus': 'risk_management', 'level': 'intermediate'},
        portfolio_context={'holdings': 'AAPL, MSFT, GOOGL', 'risk_tolerance': 'Moderate'}
    )
    
    # Initialize agent (you'll need to provide your Gemini API key)
    agent = LanguageAgent(gemini_api_key="AIzaSyBZhEyYLeNGqCrxCLIAOxUeXBdCYW5Kly4")
    
    # Generate response
    try:
        response = await agent.synthesize_response(sample_sources, context)
        print("Generated Response:")
        print(response['response'])
        print(f"\nConfidence Score: {response['confidence_score']}")
        print(f"Sources Used: {response['sources_summary']}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Example usage with rate limit handling
    
    async def demo_lightweight_agent():
        """Demo lightweight agent for free tier"""
        print("=== Lightweight Language Agent Demo (Free Tier Optimized) ===")
        
        # Use lightweight agent for testing
        agent = LightweightLanguageAgent(gemini_api_key="AIzaSyBZhEyYLeNGqCrxCLIAOxUeXBdCYW5Kly4")
        
        test_queries = [
            "What does a 2% market drop mean for tech stocks?",
            "Should I be concerned about rising bond yields?",
            "How do earnings surprises typically affect stock prices?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            try:
                response = await agent.simple_query(query)
                print(f"Response: {response}")
                
                # Wait between queries to respect rate limits
                if i < len(test_queries):
                    print("Waiting 3 seconds before next query...")
                    await asyncio.sleep(3)
                    
            except Exception as e:
                print(f"Error: {e}")

    
    # Example of how to run:
    asyncio.run(demo_lightweight_agent())