"""
Advanced Topic Extraction for Analytics
Uses LangChain and NLP to extract meaningful investment topics from queries
"""

import re
from typing import List, Dict, Set
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv

load_dotenv()


class AdvancedTopicExtractor:
    """
    Uses LLM-based extraction to identify meaningful investment topics from queries.
    Much more accurate than regex/stopword approaches.
    """
    
    def __init__(self):
        self.model = ChatOpenAI(
            model="gpt-4o-mini",  # Fast and cheap for extraction
            temperature=0  # Deterministic extraction
        )
        
        # Financial domain categories for validation
        self.financial_categories = {
            'asset_classes': {
                'stocks', 'bonds', 'commodities', 'real estate', 'cash', 'alternatives',
                'equities', 'fixed income', 'derivatives', 'currencies', 'forex',
                'cryptocurrency', 'crypto', 'bitcoin', 'ethereum'
            },
            'markets': {
                'emerging markets', 'developed markets', 'frontier markets',
                'asia', 'europe', 'americas', 'china', 'india', 'japan', 'usa',
                'latin america', 'middle east', 'africa'
            },
            'sectors': {
                'technology', 'healthcare', 'financials', 'energy', 'utilities',
                'industrials', 'materials', 'consumer', 'telecommunications',
                'real estate', 'consumer discretionary', 'consumer staples'
            },
            'economic_terms': {
                'inflation', 'deflation', 'recession', 'growth', 'gdp', 'unemployment',
                'interest rates', 'monetary policy', 'fiscal policy', 'quantitative easing',
                'yields', 'spreads', 'volatility', 'liquidity'
            },
            'investment_strategies': {
                'value', 'growth', 'momentum', 'quality', 'dividend', 'income',
                'active', 'passive', 'etf', 'mutual fund', 'hedge fund',
                'long short', 'market neutral'
            },
            'themes': {
                'esg', 'sustainability', 'climate', 'renewable energy', 'green',
                'ai', 'artificial intelligence', 'automation', 'digitalization',
                'demographics', 'aging population', 'urbanization'
            }
        }
        
        # Flatten all financial terms for quick lookup
        self.all_financial_terms = set()
        for category_terms in self.financial_categories.values():
            self.all_financial_terms.update(category_terms)
        
        # Setup extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial topic extraction specialist. Your job is to extract ONLY the meaningful investment/financial topics from user queries.

RULES:
1. Extract ONLY investment-relevant topics (asset classes, markets, sectors, economic terms, strategies)
2. Ignore query words like: provide, latest, insight, outlook, please, give, show, tell, explain
3. Ignore generic words like: being, having, making, getting, doing
4. Extract multi-word phrases when relevant (e.g., "emerging markets", "fixed income", "interest rates")
5. Normalize terms (e.g., "bond" and "bonds" → "bonds")
6. Return ONLY topics that relate to finance/investing/economics

EXAMPLES:
Query: "Please provide the latest insight on commodities"
Topics: ["commodities"]

Query: "What's the outlook for emerging markets and fixed income?"
Topics: ["emerging markets", "fixed income"]

Query: "How are interest rates affecting bonds in Asia?"
Topics: ["interest rates", "bonds", "asia"]

Query: "Give me information about ESG investing"
Topics: ["esg", "investing"]

Return a JSON object with a "topics" array containing the extracted topics."""),
            ("user", "Extract investment topics from this query: {query}")
        ])
    
    def extract_topics_with_llm(self, query_text: str) -> List[str]:
        """
        Use LLM to extract topics from query.
        Falls back to hybrid approach if LLM fails.
        """
        try:
            # Use LLM for extraction
            chain = self.extraction_prompt | self.model
            response = chain.invoke({"query": query_text})
            
            # Parse JSON response
            content = response.content.strip()
            
            # Handle JSON extraction
            import json
            
            # Try to find JSON in response
            if '{' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                
                topics = data.get('topics', [])
                
                # Validate and clean topics
                validated_topics = self._validate_topics(topics)
                
                if validated_topics:
                    return validated_topics
            
            # If LLM extraction failed, fall back to hybrid
            return self._hybrid_extraction(query_text)
            
        except Exception as e:
            print(f"LLM extraction failed: {e}, falling back to hybrid approach")
            return self._hybrid_extraction(query_text)
    
    def _validate_topics(self, topics: List[str]) -> List[str]:
        """
        Validate extracted topics against financial domain knowledge.
        """
        validated = []
        
        for topic in topics:
            topic_lower = topic.lower().strip()
            
            # Skip empty or very short topics
            if not topic_lower or len(topic_lower) < 3:
                continue
            
            # Check if it's a known financial term
            if topic_lower in self.all_financial_terms:
                validated.append(topic_lower)
                continue
            
            # Check if it contains a known financial term
            for financial_term in self.all_financial_terms:
                if financial_term in topic_lower or topic_lower in financial_term:
                    validated.append(topic_lower)
                    break
            else:
                # Not a known term but might still be valid (e.g., new company, event)
                # Apply heuristics
                if self._is_likely_financial_topic(topic_lower):
                    validated.append(topic_lower)
        
        return list(set(validated))  # Remove duplicates
    
    def _is_likely_financial_topic(self, term: str) -> bool:
        """
        Use heuristics to determine if a term is likely a financial topic.
        """
        # Exclude common query words
        query_words = {
            'provide', 'latest', 'insight', 'insights', 'outlook', 'please',
            'give', 'show', 'tell', 'explain', 'describe', 'discuss',
            'information', 'details', 'summary', 'overview', 'update',
            'analysis', 'report', 'article', 'being', 'having', 'making',
            'getting', 'doing', 'going', 'coming', 'saying', 'looking',
            'wanting', 'needing', 'thinking', 'knowing', 'seeing'
        }
        
        if term in query_words:
            return False
        
        # Exclude generic verbs
        if term.endswith('ing') and len(term) < 8:
            return False
        
        # Must be at least 4 characters
        if len(term) < 4:
            return False
        
        # If it's capitalized, might be a company/place name
        if term[0].isupper():
            return True
        
        # If it contains numbers, might be relevant (e.g., "2025 outlook")
        if any(char.isdigit() for char in term):
            return True
        
        # Otherwise, be conservative
        return False
    
    def _hybrid_extraction(self, query_text: str) -> List[str]:
        """
        Hybrid approach: combine pattern matching with financial term matching.
        Used as fallback when LLM extraction fails.
        """
        query_lower = query_text.lower()
        extracted = []
        
        # 1. Extract known multi-word financial phrases first
        for term in sorted(self.all_financial_terms, key=len, reverse=True):
            if len(term.split()) > 1 and term in query_lower:
                extracted.append(term)
                # Remove from query to avoid duplication
                query_lower = query_lower.replace(term, '')
        
        # 2. Extract known single-word financial terms
        words = re.findall(r'\b[a-z]+\b', query_lower)
        for word in words:
            if word in self.all_financial_terms and word not in extracted:
                extracted.append(word)
        
        # 3. Extract capitalized terms (might be companies, places)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query_text)
        for cap_word in capitalized:
            if cap_word.lower() not in extracted and len(cap_word) > 3:
                extracted.append(cap_word.lower())
        
        return list(set(extracted))


class BatchTopicExtractor:
    """
    Efficiently extract topics from multiple queries using batch processing.
    """
    
    def __init__(self):
        self.extractor = AdvancedTopicExtractor()
        self.cache = {}  # Cache query -> topics mapping
    
    def extract_batch(self, queries: List[str]) -> Dict[str, List[str]]:
        """
        Extract topics from multiple queries efficiently.
        Uses caching to avoid re-processing identical queries.
        """
        results = {}
        
        for query in queries:
            # Check cache first
            if query in self.cache:
                results[query] = self.cache[query]
            else:
                # Extract topics
                topics = self.extractor.extract_topics_with_llm(query)
                self.cache[query] = topics
                results[query] = topics
        
        return results
    
    def get_topic_statistics(self, query_topic_map: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Get frequency statistics for topics across all queries.
        """
        topic_counts = {}
        
        for topics in query_topic_map.values():
            for topic in topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by frequency
        sorted_topics = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_topics


# Standalone extraction function for easy use
def extract_topics_from_query(query: str) -> List[str]:
    """
    Simple function to extract topics from a single query.
    
    Usage:
        topics = extract_topics_from_query("What's the outlook for commodities?")
        # Returns: ['commodities']
    """
    extractor = AdvancedTopicExtractor()
    return extractor.extract_topics_with_llm(query)


# Test function
def test_extraction():
    """Test the extraction system with various queries."""
    extractor = AdvancedTopicExtractor()
    
    test_queries = [
        "Please provide the latest insight on commodities",
        "What's the outlook for emerging markets and fixed income?",
        "How are interest rates affecting bonds in Asia?",
        "Give me information about ESG investing",
        "What's being said about inflation lately?",
        "Latest implications of Fed policy on equities",
        "China technology sector outlook",
        "Renewable energy investment opportunities"
    ]
    
    print("Testing Advanced Topic Extraction\n" + "="*60)
    
    for query in test_queries:
        topics = extractor.extract_topics_with_llm(query)
        print(f"\nQuery: {query}")
        print(f"Topics: {topics}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_extraction()