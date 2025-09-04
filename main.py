from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import keepa
import openai
import requests
import os
import uuid
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import pprint
from datetime import datetime, timedelta
import numpy as np

# Load environment variables
load_dotenv()

# Initialize API
app = FastAPI(title="Amazon FBA Analyzer API")

# API keys configuration (from environment)
keepa_api_key = os.getenv("KEEPA_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
optisage_api_url = os.getenv("OPTISAGE_API_URL")

# Validate keys
if not keepa_api_key or not openai_api_key or not optisage_api_url:
    raise EnvironmentError("Missing API keys in environment variables")

# Initialize APIs
keepa_api = keepa.Keepa(keepa_api_key)
openai_client = openai.OpenAI(api_key=openai_api_key)

# Session storage (use Redis or DB in production)
active_sessions = {}

# Request models
class AnalyzeRequest(BaseModel):
    asin: str
    cost_price: float
    marketplaceId: int
    isAmazonFulfilled: bool

class ChatRequest(BaseModel):
    session_id: str
    question: str

# Optisage profitability API call
def call_optisage_profitability_api(asin, cost_price, marketplaceId=1, isAmazonFulfilled=False):
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    payload = {
        "asin": asin,
        "marketplaceId": marketplaceId,
        "isAmazonFulfilled": isAmazonFulfilled,
        "costPrice": cost_price
    }
    response = requests.post(optisage_api_url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data['data']

# Extract 30-day average sales rank
def get_30_day_average_sales_rank(product):
    sales_rank_data = product.get('salesRanks', {}).get('SALES', [])
    stats = product.get('stats') or {}
    monthly_sales = stats.get('sales30', 0)

    if sales_rank_data:
        return sales_rank_data[0]
    
    if monthly_sales >= 5000:
        return 1000
    elif monthly_sales >= 3000:
        return 5000
    elif monthly_sales >= 1000:
        return 10000
    elif monthly_sales >= 500:
        return 30000
    elif monthly_sales >= 100:
        return 80000
    else:
        return 200000

# Demand scoring (0-1 scale)
def estimate_demand_score(monthly_sales):
    if monthly_sales >= 300:
        return 1.0
    elif monthly_sales >= 100:
        return 0.4
    else:
        return 0.0

# Profitability scoring (0-1 scale)
def calculate_profitability_score(roi, profit_margin):
    score = 0
    
    if roi >= 50:
        score += 0.5
    elif roi >= 30:
        score += 0.3
    elif roi >= 10:
        score += 0.1
    
    if profit_margin >= 25:
        score += 0.5
    elif profit_margin >= 20:
        score += 0.4
    elif profit_margin >= 15:
        score += 0.3
    elif profit_margin >= 10:
        score += 0.2
    elif profit_margin >= 1:
        score += 0.1
    
    return min(score, 1.0)

# Keepa Chart Insight Extractor with NumPy array handling
class KeepaChartInsights:
    @staticmethod
    def extract_chart_insights(product: Dict) -> Dict[str, Any]:
        """Extract insights from Keepa charts handling NumPy arrays"""
        try:
            insights = {
                "demand_analysis": KeepaChartInsights.analyze_demand_patterns(product),
                "pricing_analysis": KeepaChartInsights.analyze_pricing_strategy(product),
                "competition_analysis": KeepaChartInsights.analyze_competition(product),
                "key_metrics": KeepaChartInsights.extract_key_metrics(product)
            }
            return insights
        except Exception as e:
            return {"error": f"Failed to extract insights: {str(e)}"}

    @staticmethod
    def analyze_demand_patterns(product: Dict) -> Dict:
        """Analyze sales rank patterns handling NumPy arrays"""
        try:
            data = product.get('data', {})
            
            # Handle NumPy arrays properly
            sales_rank_data = data.get('SALES', [])
            sales_rank_times = data.get('SALES_time', [])
            
            # Convert NumPy arrays to lists if needed
            if hasattr(sales_rank_data, 'tolist'):
                sales_rank_data = sales_rank_data.tolist()
            if hasattr(sales_rank_times, 'tolist'):
                sales_rank_times = sales_rank_times.tolist()
            
            current_rank = product.get('salesRank', 0)
            monthly_sales = product.get('stats', {}).get('sales30', 0)

            # Check if we have valid data
            has_sales_data = len(sales_rank_data) > 0 if hasattr(sales_rank_data, '__len__') else bool(sales_rank_data)
            has_time_data = len(sales_rank_times) > 0 if hasattr(sales_rank_times, '__len__') else bool(sales_rank_times)
            
            if not has_sales_data or not has_time_data:
                return {
                    "current_rank": current_rank,
                    "monthly_sales": monthly_sales,
                    "insight": "No historical sales rank data available",
                    "data_available": {
                        "SALES_length": len(sales_rank_data) if hasattr(sales_rank_data, '__len__') else 0,
                        "SALES_time_length": len(sales_rank_times) if hasattr(sales_rank_times, '__len__') else 0
                    }
                }

            # Ensure both arrays have the same length
            min_length = min(len(sales_rank_data), len(sales_rank_times))
            sales_rank_data = sales_rank_data[:min_length]
            sales_rank_times = sales_rank_times[:min_length]

            # Pair time + rank and filter out invalid values (-1)
            rank_history = []
            for i in range(min_length):
                rank = sales_rank_data[i]
                time = sales_rank_times[i]
                if rank > 0:  # Filter out -1 values (no rank)
                    rank_history.append((time, rank))

            if not rank_history:
                return {
                    "current_rank": current_rank,
                    "monthly_sales": monthly_sales,
                    "insight": "All sales rank data points were invalid (-1)",
                    "original_data_points": min_length
                }

            # Get recent data (last 90 days)
            recent_history = rank_history[-90:] if len(rank_history) > 90 else rank_history
            ranks = [rank for _, rank in recent_history]

            # Calculate trends and patterns
            trend = "stable"
            if len(ranks) > 1:
                if ranks[-1] < ranks[0] * 0.7:  # Improved by 30%
                    trend = "improving"
                elif ranks[-1] > ranks[0] * 1.3:  # Worsened by 30%
                    trend = "declining"

            volatility = np.std(ranks) / np.mean(ranks) if np.mean(ranks) > 0 else 0

            return {
                "current_rank": ranks[-1] if ranks else current_rank,
                "average_rank_90d": int(np.mean(ranks)) if ranks else 0,
                "best_rank_90d": min(ranks) if ranks else 0,
                "worst_rank_90d": max(ranks) if ranks else 0,
                "trend": trend,
                "volatility": round(volatility, 3),
                "data_points": len(ranks),
                "monthly_sales": monthly_sales,
                "total_data_points": min_length
            }
            
        except Exception as e:
            return {
                "current_rank": product.get('salesRank', 0),
                "monthly_sales": product.get('stats', {}).get('sales30', 0),
                "error": f"Demand analysis failed: {str(e)}"
            }

    @staticmethod
    def analyze_pricing_strategy(product: Dict) -> Dict:
        """Analyze pricing patterns handling NumPy arrays"""
        try:
            data = product.get('data', {})
            pricing_insights = {}
            
            # Try common pricing keys
            price_types = {
                'AMAZON': 'amazon_price',
                'NEW': 'marketplace_new_price', 
                'USED': 'marketplace_used_price',
                'BUY_BOX_SHIPPING': 'buy_box_price'
            }
            
            for keepa_key, insight_key in price_types.items():
                if keepa_key in data:
                    prices = data[keepa_key]
                    
                    # Handle NumPy arrays
                    if hasattr(prices, 'tolist'):
                        prices = prices.tolist()
                    
                    # Check if we have valid data
                    has_prices = len(prices) > 0 if hasattr(prices, '__len__') else bool(prices)
                    
                    if has_prices:
                        # Filter valid prices (>0)
                        valid_prices = [p for p in prices if p > 0]
                        if valid_prices:
                            recent_prices = valid_prices[-30:]
                            avg_price = np.mean(recent_prices)
                            min_price = min(recent_prices)
                            max_price = max(recent_prices)
                            volatility = (max_price - min_price) / avg_price if avg_price > 0 else 0
                            
                            pricing_insights[insight_key] = {
                                "current": recent_prices[-1],
                                "average_30d": round(avg_price, 2),
                                "min_30d": min_price,
                                "max_30d": max_price,
                                "range": round(max_price - min_price, 2),
                                "volatility": round(volatility, 3),
                                "data_points": len(recent_prices)
                            }
            
            return pricing_insights if pricing_insights else {"insight": "No pricing data available"}
                
        except Exception as e:
            return {"error": f"Pricing analysis failed: {str(e)}"}

    @staticmethod
    def analyze_competition(product: Dict) -> Dict:
        """Analyze competitive landscape handling NumPy arrays"""
        try:
            data = product.get('data', {})
            offers = product.get('offers', [])
            fba_offers = [offer for offer in offers if offer and offer.get('isFBA')]
            
            total_offers = len(offers)
            fba_count = len(fba_offers)
            
            # Get offer count history (handle NumPy arrays)
            new_offer_history = data.get('COUNT_NEW', [])
            if hasattr(new_offer_history, 'tolist'):
                new_offer_history = new_offer_history.tolist()
            
            used_offer_history = data.get('COUNT_USED', [])
            if hasattr(used_offer_history, 'tolist'):
                used_offer_history = used_offer_history.tolist()
            
            competition_level = "low"
            if total_offers > 20:
                competition_level = "very_high"
            elif total_offers > 10:
                competition_level = "high"
            elif total_offers > 5:
                competition_level = "medium"
            
            return {
                "total_competitors": total_offers,
                "fba_competitors": fba_count,
                "competition_level": competition_level,
                "new_offer_history": new_offer_history[-30:] if new_offer_history else [],
                "used_offer_history": used_offer_history[-30:] if used_offer_history else [],
                "market_saturation": round(fba_count / total_offers, 2) if total_offers > 0 else 0
            }
        except Exception as e:
            return {"error": f"Competition analysis failed: {str(e)}"}

    @staticmethod
    def extract_key_metrics(product: Dict) -> Dict:
        """Extract key performance metrics"""
        try:
            stats = product.get('stats', {})
            reviews = product.get('reviews', {})
            
            return {
                "monthly_sales": stats.get('sales30', 0),
                "estimated_revenue": stats.get('revenue30', 0),
                "review_count": reviews.get('total', 0),
                "average_rating": reviews.get('average', 0),
                "sales_rank": product.get('salesRank', 0),
                "offer_count": len(product.get('offers', [])),
                "is_amazon": product.get('isAmazon', False)
            }
        except Exception as e:
            return {"error": f"Key metrics extraction failed: {str(e)}"}

# Amazon FBA Analyzer
class AmazonFBAAnalyzer:
    def __init__(self):
        self.current_analysis = None
        self.keepa_insights = None
        self.chat_history = []
        self.keepa_api = keepa_api
        self.openai_client = openai_client

    def get_product_analysis(self, asin: str, cost_price: float, marketplaceId: int, isAmazonFulfilled: bool) -> Dict[str, Any]:
        products = self.keepa_api.query(asin)
        if not products:
            return None

        product = products[0]

        # Original analysis logic
        is_amazon = product.get('isAmazon', False)
        fba_sellers = product.get('fbaOfferCount', 0)
        
        def is_buy_box_eligible(product):
            fba_offers = product.get('fbaOfferCount', 0)
            if fba_offers > 0:
                return True
            offers = product.get('offers', [])
            for offer in offers:
                if offer and offer.get('isFBA') and offer.get('offerIsActive'):
                    return True
            bbe_counts = product.get('buyBoxEligibleOfferCounts', [])
            if len(bbe_counts) >= 4:
                fba_count = bbe_counts[0] + bbe_counts[2]
                return fba_count > 0
            return False
        
        buy_box_eligible = is_buy_box_eligible(product)
        variations = product.get('variations')
        offer_count = product.get('offerCount', 0)
        sales_rank = get_30_day_average_sales_rank(product)
        monthly_sales = product.get('monthlySold', 0)

        # Optisage profitability
        profitability_data = call_optisage_profitability_api(asin, cost_price, marketplaceId, isAmazonFulfilled)
        roi = profitability_data['roi']
        profit_margin = profitability_data['profitMargin']

        profitability_score = calculate_profitability_score(roi, profit_margin)
        demand_score = estimate_demand_score(monthly_sales)

        # Original scoring logic
        if roi <= 0 or profit_margin <= 0:
            score_breakdown = {
                "profitability": 0,
                "estimated_demand": 0,
                "buy_box_eligible": 0,
                "sales_rank_impact": 0,
                "fba_sellers": 0,
                "amazon_on_listing": 0,
                "variation_listing": 0,
                "offer_count": 0
            }
            total_score = 0
            category = "Poor"
        else:
            score_breakdown = {
                "profitability": round(profitability_score * 3, 1),
                "estimated_demand": round(demand_score * 2, 1),
                "buy_box_eligible": 1.0 if buy_box_eligible else 0.0,
                "sales_rank_impact": 1.0 if sales_rank < 10000 else 0.0,
                "fba_sellers": 1.0 if fba_sellers < 4 else 0.0,
                "amazon_on_listing": 0.5 if not is_amazon else 0.0,
                "variation_listing": 0.5 if not variations else 0.0,
                "offer_count": 0.5 if offer_count < 4 else 0.0,
            }
            total_score = round(sum(score_breakdown.values()), 1)

            if total_score >= 8.0:
                category = "Excellent"
            elif total_score >= 6.0:
                category = "Good"
            elif total_score >= 4.0:
                category = "Moderate"
            else:
                category = "Poor"

        # Extract Keepa insights for enhanced chat
        self.keepa_insights = KeepaChartInsights.extract_chart_insights(product)

        # Original response structure
        self.current_analysis = {
            "asin": asin,
            "title": product.get('title', 'N/A'),
            "score_breakdown": score_breakdown,
            "total_score": total_score,
            "category": category,
            "roi": roi,
            "profit_margin": profit_margin,
            "monthly_sales": monthly_sales,
            "is_profitable": roi > 0 and profit_margin > 0,
            "keepa_insights": self.keepa_insights
        }

        return self.current_analysis

    def query_openai(self, prompt: str) -> str:
        """Enhanced query with Keepa chart context"""
        if not self.current_analysis:
            return "Please analyze a product first."

        # Prepare context
        keepa_insights = self.keepa_insights or {}
        keepa_context = ""
        
        if "error" in keepa_insights:
            keepa_context = f"Note: {keepa_insights['error']}"
        else:
            keepa_context = f"KEEPA CHART INSIGHTS:\n{pprint.pformat(keepa_insights)}"

        system_msg = {
            "role": "system",
            "content": f"""You are an expert Amazon FBA analyst. Use available data:

            PRODUCT ANALYSIS:
            {pprint.pformat({k: v for k, v in self.current_analysis.items() if k != 'keepa_insights'})}
            
            {keepa_context}
            
            Provide detailed, data-driven answers based on the available information.
            """
        }

        messages = [system_msg] + self.chat_history[-6:] + [{"role": "user", "content": prompt}]

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            reply = response.choices[0].message.content
            self.chat_history.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply}
            ])
            return reply
        except Exception as e:
            return f"Error generating response: {str(e)}"

# API Endpoints
@app.post("/analyze")
async def analyze_product(request: AnalyzeRequest):
    try:
        analyzer = AmazonFBAAnalyzer()
        session_id = str(uuid.uuid4())
        analysis = analyzer.get_product_analysis(
            request.asin, request.cost_price, request.marketplaceId, request.isAmazonFulfilled
        )

        if not analysis:
            raise HTTPException(status_code=404, detail="Product not found")

        active_sessions[session_id] = analyzer

        return {
            "session_id": session_id,
            "score": analysis["total_score"],
            "category": analysis["category"],
            "breakdown": analysis["score_breakdown"],
            "roi": analysis["roi"],
            "profit_margin": analysis["profit_margin"],
            "monthly_sales": analysis["monthly_sales"],
            "is_profitable": analysis["is_profitable"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_analysis(request: ChatRequest):
    """Enhanced chat endpoint with Keepa chart insights"""
    analyzer = active_sessions.get(request.session_id)
    if not analyzer:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invalid session")
    
    try:
        response = analyzer.query_openai(request.question)
        return {
            "session_id": request.session_id,
            "response": response,
            "chat_history": analyzer.chat_history[-4:]
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))
