from .base_agent import BaseAgent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from typing import Any, Dict, List

class LanguageAgent(BaseAgent):
    def __init__(self):
        super().__init__("language_agent")
        self.llm = OpenAI(temperature=0.7)
        self.templates = self.load_prompt_templates()
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_type = task.get("task_type")
        data = task.get("data")
        
        if task_type == "market_brief":
            return self.generate_market_brief(data)
        elif task_type == "risk_summary":
            return self.generate_risk_summary(data)
        
        return {"error": "Unknown task type"}
    
    def generate_market_brief(self, data: Dict) -> Dict[str, Any]:
        prompt = self.templates["market_brief"].format(**data)
        response = self.llm(prompt)
        
        return {
            "brief": response,
            "tone": "professional",
            "length": len(response.split())
        }
    
    def load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        return {
            "market_brief": PromptTemplate(
                input_variables=["portfolio_data", "market_data", "news_summary"],
                template="""
                Based on the following data, create a concise market brief:
                Portfolio: {portfolio_data}
                Market Data: {market_data}
                News: {news_summary}
                
                Provide a professional, actionable summary.
                """
            )
        }
    
    def get_capabilities(self) -> List[str]:
        return ["text_generation", "summarization", "analysis_synthesis"]