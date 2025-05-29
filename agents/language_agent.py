import asyncio
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Simple market data structure for morning brief"""
    content: str
    data_type: str  # 'price', 'news', 'economic'
    timestamp: datetime
    importance: float  # 0.0 to 1.0

class MorningBriefAgent:
    """
    Simplified Language Agent focused only on morning market briefs
    Optimized for free tier usage with rate limiting
    """
    
    def __init__(self, gemini_api_key: str):
        """Initialize the Morning Brief Agent"""
        genai.configure(api_key=gemini_api_key)
        
        # Use flash model for free tier efficiency
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1000,  # Keep brief
                top_p=0.8,
                top_k=32
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 3.0  # 3 seconds between requests
        
        # Morning brief template
        self.system_prompt = """You are a precise financial analyst tasked with generating comprehensive and accurate morning market briefs for Company PMs. Your analysis must synthesize market news, FRED economic indicators, SEC filings, stock data, and other critical financial information into actionable summaries.

CRITICAL DATA ACCURACY REQUIREMENTS:
- Cross-verify all stock prices across multiple data sources (API feeds, stock_info, stock_quotes)
- Prioritize the most recent and reliable data; flag discrepancies if data sources conflict
- Verify all numerical data before inclusion
- Always calculate percentage changes using: ((current - previous) / previous) * 100
- Present economic data with full context—avoid reporting raw numbers alone

MANDATORY ECONOMIC CONTEXT INTEGRATION:
- GDP: Report growth or contraction rates with historical comparison
- VIX: Include current level, trend, and volatility implications
- Treasury Yields: Report 2Y and 10Y yields with commentary on yield curve shape and implications
- Fed Funds Rate: Current rate and monetary policy outlook
- Unemployment & Inflation: Highlight recent trends and policy consequences
- Consumer Sentiment: Provide interpretation of latest reading and market relevance
- Clearly explain the implications of each economic indicator for investor decision-making

PORTFOLIO-FOCUSED ANALYSIS PRIORITY:
- Focus on user's portfolio holdings with verified price data and percent changes
- Include upcoming earnings dates and analyst expectations (must be verified)
- Highlight relevant filings or regulatory changes and their potential impact
- Compare portfolio performance to broader sector and market trends
- Provide sector-specific analysis relevant to holdings

BRIEF STRUCTURE (350-400 words):

**Overnight Market Summary:**
- Specific index movements with percentages and drivers
- Notable currency and commodity shifts
- Key overnight events and their market impact

**Economic Indicators Analysis:**
- Latest FRED data with trends, growth rates, and investment implications
- Central bank policy direction and interest rate expectations
- Credit market signals (spreads, volatility measures)
- Employment and inflation analysis

**Portfolio Holdings Deep Dive:**
- Verified current prices with accurate % changes and volume commentary
- Upcoming catalysts (earnings, events, filings) with verified dates
- Competitive positioning and sector trends
- Risk/opportunity summary with actionable price levels

**Actionable Intelligence:**
- 3 or 4 prioritized recommendations with rationale
- Critical technical levels (support/resistance) to monitor
- Risk management guidance tailored to portfolio
- Key upcoming macro/calendar events

QUALITY CONTROL CHECKLIST (must be completed before final output):
✓ All price data verified from multiple sources
✓ Economic data contextualized with trends and rates
✓ Portfolio analysis based on accurate and recent figures
✓ Earnings dates/events confirmed for accuracy
✓ Indicators interpreted with market impact clarity
✓ No contradictions across data sources. If any verify by yourself and provide correct output.
✓ Actionable insights are specific, data-driven, and clearly justified

PROFESSIONAL STANDARDS:
- Use precise financial terminology and clear metrics
- Maintain an objective, analytical tone focused on market-moving insights
- Provide supporting data for all major claims
- Focus solely on information relevant to investment decisions
- Never speculate without evidence or introduce unverified assumptions

Remember: Accuracy is critical—any incorrect figure or misinterpretation may result in poor investment outcomes. Always validate all data before completing the brief."""


    async def generate_morning_brief(self, 
                                   market_data: List[MarketData],
                                   user_portfolio: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate a concise morning market brief
        
        Args:
            market_data: List of market data points
            user_portfolio: Optional list of user's holdings (tickers)
            
        Returns:
            Dict with brief text and metadata
        """
        
        try:
            # Rate limiting
            await self._rate_limit_wait()
            
            # Process and prioritize data
            processed_data = self._process_market_data(market_data)
            
            # Build focused prompt
            prompt = self._build_brief_prompt(processed_data, user_portfolio)
            
            # Generate brief
            response = await self._generate_response(prompt)
            
            return {
                'brief': response,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points_used': len(processed_data),
                'portfolio_focused': bool(user_portfolio)
            }
            
        except Exception as e:
            logger.error(f"Error generating morning brief: {str(e)}")
            return self._generate_fallback_brief(str(e))
    
    def _process_market_data(self, market_data: List[MarketData]) -> List[MarketData]:
        """Process and prioritize market data by importance and recency"""
        
        # Sort by importance and recency
        sorted_data = sorted(
            market_data, 
            key=lambda x: (x.importance, -time.mktime(x.timestamp.timetuple())),
            reverse=True
        )
        
        # Take top 5 most important items to stay within token limits
        return sorted_data[:5]
    
    def _build_brief_prompt(self, 
                           market_data: List[MarketData], 
                           user_portfolio: Optional[List[str]]) -> str:
        """Build concise prompt for morning brief"""
        
        # Prepare market data summary
        data_summary = []
        for data in market_data:
            data_summary.append(f"• {data.content}")
        
        data_text = "\n".join(data_summary) if data_summary else "Limited market data available"
        
        # Portfolio context
        portfolio_text = ""
        if user_portfolio:
            portfolio_text = f"\nUser Portfolio: {', '.join(user_portfolio)}"
        
        prompt = f"""{self.system_prompt}

MARKET DATA:
{data_text}
{portfolio_text}

Generate a morning brief covering:
1. Key overnight market movements
2. Major news impacting markets
3. What to watch today
{f"4. Specific insights for portfolio holdings" if user_portfolio else ""}

Keep it concise and actionable."""
        
        return prompt
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response with error handling"""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                logger.warning("Empty response from Gemini")
                return self._get_default_brief()
                
        except Exception as e:
            error_str = str(e)
            
            # Handle rate limit specifically
            if "429" in error_str or "quota" in error_str.lower():
                logger.warning("Rate limit exceeded")
                return "Markets are active this morning. Please check financial news sources like Bloomberg, CNBC, or Yahoo Finance for the latest updates. Rate limit reached - please try again in a few minutes."
            else:
                logger.error(f"Error generating response: {error_str}")
                raise
    
    async def _rate_limit_wait(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def _generate_fallback_brief(self, error_msg: str) -> Dict[str, Any]:
        """Generate fallback brief when main generation fails"""
        
        fallback_text = """Good morning! I'm currently experiencing technical difficulties generating your personalized brief.
        
For this morning's market update, please check:
• Financial news websites (Bloomberg, CNBC, Reuters)
• Your broker's market summary
• Yahoo Finance or Google Finance for major indices

Key things to typically watch in morning briefs:
• Overnight futures movements
• Asian and European market closes
• Pre-market earnings releases
• Economic data releases scheduled for today

I apologize for the inconvenience and will be back online shortly."""
        
        return {
            'brief': fallback_text,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points_used': 0,
            'portfolio_focused': False,
            'error': error_msg
        }
    
    def _get_default_brief(self) -> str:
        """Default brief when no specific data is available"""
        return """Good morning! Here's what to watch in markets today:

• Check major indices (S&P 500, Nasdaq, Dow) for overnight futures movement
• Monitor any pre-market earnings announcements
• Watch for economic data releases scheduled today
• Keep an eye on sector rotation and commodity prices

For detailed market information, please visit financial news sources or your preferred trading platform."""

    # Convenience method for quick queries
    async def quick_market_question(self, question: str) -> str:
        """Handle quick market-related questions with minimal token usage"""
        
        await self._rate_limit_wait()
        
        simple_prompt = f"""Brief financial question: {question}

Provide a concise, factual response in 1-2 sentences."""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    simple_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=100,
                        temperature=0.2
                    )
                )
            )
            
            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text
            else:
                return "I'm unable to answer that question right now. Please try rephrasing or check financial news sources."
                
        except Exception as e:
            if "429" in str(e):
                return "Rate limit reached. Please wait a moment and try again."
            else:
                return f"Error processing question: {str(e)}"


# Example usage and testing
async def demo_morning_brief():
    """Demonstrate the morning brief agent"""
    
    print("=== Morning Brief Agent Demo ===")
    
    # Initialize agent (replace with your API key)
    agent = MorningBriefAgent(gemini_api_key="YOUR_API_KEY_HERE")
    
    # Sample market data
    sample_data = [
        MarketData(
            content="S&P 500 futures down 0.8% in pre-market trading",
            data_type="price",
            timestamp=datetime.now(),
            importance=0.9
        ),
        MarketData(
            content="Apple reports mixed Q4 earnings, stock down 2% after hours",
            data_type="news",
            timestamp=datetime.now(),
            importance=0.8
        ),
        MarketData(
            content="Fed officials signal potential rate pause in upcoming meeting",
            data_type="economic",
            timestamp=datetime.now(),
            importance=0.95
        )
    ]
    
    # Sample portfolio
    user_portfolio = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    try:
        # Generate morning brief
        brief_result = await agent.generate_morning_brief(sample_data, user_portfolio)
        
        print("Morning Brief:")
        print("=" * 50)
        print(brief_result['brief'])
        print("\nMetadata:")
        print(f"Generated at: {brief_result['generated_at']}")
        print(f"Data points used: {brief_result['data_points_used']}")
        print(f"Portfolio focused: {brief_result['portfolio_focused']}")
        
        # Demo quick question
        print("\n" + "=" * 50)
        print("Quick Question Demo:")
        question = "What does rising bond yields mean for tech stocks?"
        answer = await agent.quick_market_question(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_morning_brief())