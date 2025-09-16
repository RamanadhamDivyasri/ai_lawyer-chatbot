# backend.py
# LawyerBot AI Legal Assistant Backend
# 
# How to run locally:
# 1. Install dependencies: pip install -r requirements.txt
# 2. Copy .env.example to .env and add your Gemini API key
# 3. Run: python frontend.py
#
# This backend handles LLM orchestration, prompt engineering, and safety measures

import os
import re
import time
import json
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

from dotenv import load_dotenv
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredLegalResponseParser(BaseOutputParser):
    """Parse the LLM response into structured legal format"""
    
    def parse(self, text: str) -> Dict[str, str]:
        """Parse the response into structured sections"""
        sections = {
            'issue': '',
            'principles': '',
            'analysis': '',
            'conclusion': '',
            'recommendations': '',
            'disclaimer': ''
        }
        
        # Split by section headers (case insensitive)
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            lower_line = line.lower()
            if 'issue:' in lower_line or line.startswith('**Issue'):
                current_section = 'issue'
                sections[current_section] = line + '\n'
            elif 'relevant principles:' in lower_line or 'principles:' in lower_line or line.startswith('**Relevant Principles') or line.startswith('**Principles'):
                current_section = 'principles'
                sections[current_section] = line + '\n'
            elif 'analysis:' in lower_line or line.startswith('**Analysis'):
                current_section = 'analysis'
                sections[current_section] = line + '\n'
            elif 'conclusion:' in lower_line or line.startswith('**Conclusion'):
                current_section = 'conclusion'
                sections[current_section] = line + '\n'
            elif 'recommendations:' in lower_line or line.startswith('**Recommendations'):
                current_section = 'recommendations'
                sections[current_section] = line + '\n'
            elif 'disclaimer:' in lower_line or line.startswith('**Disclaimer'):
                current_section = 'disclaimer'
                sections[current_section] = line + '\n'
            elif current_section:
                sections[current_section] += line + '\n'
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
            
        return sections

class RequestThrottler:
    """Simple in-memory request throttling"""
    
    def __init__(self, max_requests_per_minute: int = 5):
        self.max_requests = max_requests_per_minute
        self.requests = defaultdict(deque)  # session_id -> deque of timestamps
    
    def is_allowed(self, session_id: str) -> bool:
        """Check if request is allowed for this session"""
        now = datetime.now()
        session_requests = self.requests[session_id]
        
        # Remove old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        while session_requests and session_requests[0] < cutoff:
            session_requests.popleft()
        
        # Check if under limit
        if len(session_requests) >= self.max_requests:
            return False
        
        # Add current request
        session_requests.append(now)
        return True
    
    def time_until_allowed(self, session_id: str) -> int:
        """Return seconds until next request is allowed"""
        if not self.requests[session_id]:
            return 0
        
        oldest_request = self.requests[session_id][0]
        time_since_oldest = datetime.now() - oldest_request
        return max(0, 60 - int(time_since_oldest.total_seconds()))

class PromptInjectionDetector:
    """Simple prompt injection detection"""
    
    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"forget\s+everything",
        r"you\s+are\s+now",
        r"act\s+as\s+if",
        r"pretend\s+to\s+be",
        r"system\s*:",
        r"assistant\s*:",
        r"human\s*:",
        r"<\s*system\s*>",
        r"<\s*\/\s*system\s*>",
        r"prompt\s*injection",
        r"jailbreak",
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS]
    
    def detect(self, text: str) -> bool:
        """Detect potential prompt injection attempts"""
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                logger.warning(f"Potential prompt injection detected: {pattern.pattern}")
                return True
        return False
    
    def sanitize(self, text: str) -> str:
        """Sanitize text by removing/neutralizing injection attempts"""
        sanitized = text
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub("[REMOVED]", sanitized)
        return sanitized

class LawyerBot:
    """Main LawyerBot class handling LLM orchestration and legal responses"""
    
    # Main prompt template for legal responses
    LEGAL_PROMPT_TEMPLATE = """You are LawyerBot, an AI legal information assistant. You provide general legal principles and information, NOT jurisdiction-specific legal advice.

CRITICAL INSTRUCTIONS:
1. Start EVERY response with: "I am LawyerBot, an AI legal information assistant. I provide general legal principles and not jurisdiction-specific legal advice."
2. Structure your response with these exact sections:
   - **Issue:** [Clearly state the legal issue/question]
   - **Relevant Principles:** [General legal principles that apply]
   - **Analysis:** [Explain how the principles relate to the situation]
   - **Conclusion:** [Summarize the key points]
   - **Recommendations:** [General guidance on next steps]
   - **Disclaimer:** [Standard legal disclaimer]

3. Focus on GENERAL legal principles, not specific jurisdictions
4. Cite typical sources like "statutes," "case law," "contract terms," etc.
5. Never provide jurisdiction-specific advice
6. Always end with a comprehensive disclaimer

STANDARD DISCLAIMER TO INCLUDE:
"This response provides general legal information only and is not a substitute for professional legal advice from a licensed attorney. Laws vary by jurisdiction, and you should consult with a qualified lawyer in your area for advice specific to your situation."

User Question: {user_question}

{context_section}

Please provide a structured legal response following the format above."""
    
    def __init__(self):
        """Initialize LawyerBot with Gemini API and safety components"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash-lite')
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        # Initialize safety components
        self.throttler = RequestThrottler(max_requests_per_minute=int(os.getenv('MAX_REQUESTS_PER_MIN', '5')))
        self.injection_detector = PromptInjectionDetector()
        self.response_parser = StructuredLegalResponseParser()
        
        # Setup LangChain prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["user_question", "context_section"],
            template=self.LEGAL_PROMPT_TEMPLATE
        )
        
        logger.info(f"LawyerBot initialized with model: {self.model_name}")
    
    def _validate_input(self, text: str) -> tuple[bool, str]:
        """Validate and sanitize user input"""
        if not text or not text.strip():
            return False, "Input cannot be empty"
        
        if len(text) > 5000:  # Reasonable limit
            return False, "Input too long (max 5000 characters)"
        
        # Check for prompt injection
        if self.injection_detector.detect(text):
            logger.warning("Prompt injection attempt blocked")
            return False, "Input contains potentially harmful content and has been rejected"
        
        return True, "Valid"
    
    def _format_context_section(self, context: Optional[str]) -> str:
        """Format optional context for the prompt"""
        if not context or not context.strip():
            return ""
        
        # Validate context as well
        is_valid, _ = self._validate_input(context)
        if not is_valid:
            return ""
        
        return f"\nAdditional Context: {context.strip()}"
    
    def get_response(self, user_text: str, user_context: Optional[str] = None, session_id: str = "default") -> Dict[str, Any]:
        """
        Main function to get structured legal response
        
        Args:
            user_text: The legal question from user
            user_context: Optional additional context
            session_id: Session identifier for rate limiting
            
        Returns:
            Dict with role, structured_answer, raw_llm_output, and metadata
        """
        try:
            # Rate limiting check
            if not self.throttler.is_allowed(session_id):
                wait_time = self.throttler.time_until_allowed(session_id)
                return {
                    "role": "LawyerBot",
                    "structured_answer": f"Rate limit exceeded. Please wait {wait_time} seconds before your next request.",
                    "raw_llm_output": {"error": "rate_limited", "wait_time": wait_time},
                    "success": False
                }
            
            # Input validation
            is_valid, validation_msg = self._validate_input(user_text)
            if not is_valid:
                return {
                    "role": "LawyerBot",
                    "structured_answer": f"Input validation failed: {validation_msg}",
                    "raw_llm_output": {"error": "validation_failed", "message": validation_msg},
                    "success": False
                }
            
            # Sanitize input (additional safety layer)
            sanitized_text = self.injection_detector.sanitize(user_text)
            context_section = self._format_context_section(user_context)
            
            # Generate prompt using LangChain template
            formatted_prompt = self.prompt_template.format(
                user_question=sanitized_text,
                context_section=context_section
            )
            
            logger.info(f"Processing legal query for session: {session_id}")
            
            # Call Gemini API
            response = self.model.generate_content(
                formatted_prompt,
                generation_config={
                    "temperature": 0.3,  # Lower temperature for more consistent legal responses
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            
            # Parse response text
            response_text = response.text if response.text else "No response generated"
            
            # Structure the response
            structured_sections = self.response_parser.parse(response_text)
            
            # Create structured answer
            structured_answer = self._format_structured_answer(structured_sections, response_text)
            
            # Prepare raw LLM output for debugging
            raw_output = {
                "model": self.model_name,
                "prompt_tokens": len(formatted_prompt.split()),  # Approximate
                "response_tokens": len(response_text.split()),   # Approximate
                "full_response": response_text,
                "structured_sections": structured_sections,
                "generation_config": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 40
                }
            }
            
            return {
                "role": "LawyerBot",
                "structured_answer": structured_answer,
                "raw_llm_output": raw_output,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "role": "LawyerBot",
                "structured_answer": "I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists.",
                "raw_llm_output": {"error": str(e), "type": "generation_error"},
                "success": False
            }
    
    def _format_structured_answer(self, sections: Dict[str, str], full_response: str) -> str:
        """Format the structured answer for display"""
        # If sections were properly parsed, use them
        if any(sections.values()):
            formatted = []
            for section_name, content in sections.items():
                if content:
                    formatted.append(content)
            return "\n\n".join(formatted)
        
        # Fallback to full response if parsing failed
        return full_response

# Example usage and testing
if __name__ == "__main__":
    # This section demonstrates the expected behavior with example inputs
    
    print("=== LawyerBot Backend Test Examples ===\n")
    
    # Example 1: Contract dispute
    example1_input = "I signed a contract to buy a car, but the seller is now refusing to deliver it. What are my options?"
    
    print("Example 1 Input:")
    print(f"'{example1_input}'\n")
    
    print("Expected Output Structure:")
    print("""
**Issue:** Breach of contract claim regarding non-delivery of purchased vehicle

**Relevant Principles:** Contract law principles including offer, acceptance, consideration, and breach remedies. When a party fails to perform contractual obligations, the non-breaching party may have several legal remedies available under contract law.

**Analysis:** This appears to be a potential breach of contract situation where the seller has failed to deliver the agreed-upon vehicle. Contract remedies typically include specific performance, monetary damages, or contract rescission.

**Conclusion:** Multiple legal remedies may be available depending on the specific contract terms and applicable law.

**Recommendations:** Review the contract terms, document the breach, consider mediation, and consult with a local attorney for jurisdiction-specific advice.

**Disclaimer:** This response provides general legal information only and is not a substitute for professional legal advice from a licensed attorney...
    """)
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Employment issue
    example2_input = "My employer terminated me without notice. Is this legal?"
    
    print("Example 2 Input:")
    print(f"'{example2_input}'\n")
    
    print("Expected Output Structure:")
    print("""
**Issue:** Termination of employment without notice - legality and potential remedies

**Relevant Principles:** Employment law generally distinguishes between at-will employment and employment with cause requirements. Notice requirements vary by jurisdiction and employment type.

**Analysis:** The legality of termination without notice depends on factors such as employment contract terms, at-will employment status, applicable labor laws, and potential discrimination or wrongful termination claims.

**Conclusion:** Employment termination laws vary significantly by jurisdiction and employment type.

**Recommendations:** Review employment contract, document circumstances of termination, file for unemployment benefits if eligible, and consult with an employment attorney for specific legal advice.

**Disclaimer:** This response provides general legal information only and is not a substitute for professional legal advice from a licensed attorney...
    """)