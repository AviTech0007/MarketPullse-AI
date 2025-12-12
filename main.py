"""
MarketPulse AI - Backend API
A real-time financial news analysis system with AI-powered investment advice
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from enum import Enum
import random
import hashlib

app = FastAPI(
    title="MarketPulse AI API",
    description="AI-powered financial news analysis and investment advisory",
    version="1.0.0"
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. Define the origins (front-end URLs) that are allowed to make requests.
# You MUST replace the placeholder URL below with the actual URL of your deployed frontend
# (after you deploy it in Step 4).

origins = [
    # Placeholder: Replace with your actual deployed front-end URL (e.g., from Render/Netlify/Vercel)
    "https://your-frontend-service-name.onrender.com", 
    "https://marketpullse-ai-3.onrender.com",   # For local development of your front-end
    "https://marketpullse-ai-3.onrender.com",   # For local testing
]

# 2. Add the CORS middleware to your FastAPI app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # The list of allowed domains
    allow_credentials=True,
    allow_methods=["*"],    # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allows all necessary headers
)

# ... your other @app.get and @app.post functions follow ...

# CORS configuration for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class NewsArticle(BaseModel):
    id: str
    headline: str
    source: str
    url: str
    timestamp: datetime
    companies: List[str]
    sentiment: SentimentType
    relevance_score: float

class MarketImpactScore(BaseModel):
    stock_symbol: str
    company_name: str
    impact_score: float
    probability_up: float
    probability_down: float
    predicted_change_range: str
    timeframe: str
    confidence: float

class InvestmentAdvice(BaseModel):
    stock_symbol: str
    action: str
    reasoning: str
    key_headlines: List[str]
    risk_level: str
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    timestamp: datetime

class Alert(BaseModel):
    id: str
    stock_symbol: str
    alert_type: str
    message: str
    impact_score: float
    timestamp: datetime
    is_read: bool

class UserPreferences(BaseModel):
    user_id: str
    risk_profile: RiskProfile
    watchlist: List[str]
    alert_threshold: float
    notification_enabled: bool

class StockAnalysisRequest(BaseModel):
    stock_symbol: str
    risk_profile: Optional[RiskProfile] = RiskProfile.MODERATE

class NewsAnalysisResponse(BaseModel):
    articles: List[NewsArticle]
    impact_scores: List[MarketImpactScore]
    advice: InvestmentAdvice

# ============================================================================
# MOCK DATA
# ============================================================================

NIFTY50_STOCKS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd"},
    {"symbol": "TCS", "name": "Tata Consultancy Services"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd"},
    {"symbol": "INFY", "name": "Infosys Ltd"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd"},
    {"symbol": "ITC", "name": "ITC Ltd"},
    {"symbol": "SBIN", "name": "State Bank of India"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank"}
]

NEWS_SOURCES = ["Economic Times", "Moneycontrol", "Bloomberg", "Reuters", "LiveMint"]

SAMPLE_HEADLINES = [
    "announces record quarterly profits, beats estimates",
    "faces regulatory scrutiny over market practices",
    "plans major expansion into renewable energy sector",
    "reports decline in profit margins amid rising costs",
    "launches innovative product line targeting millennials",
    "CEO announces strategic partnership with global leader",
    "stock surges on strong earnings guidance",
    "faces headwinds from global economic slowdown"
]

def generate_mock_news(stock_symbol: str, count: int = 5) -> List[NewsArticle]:
    articles = []
    company_name = next((s["name"] for s in NIFTY50_STOCKS if s["symbol"] == stock_symbol), stock_symbol)
    
    for i in range(count):
        sentiment = random.choice([SentimentType.POSITIVE, SentimentType.NEGATIVE, SentimentType.NEUTRAL])
        headline = f"{company_name} {random.choice(SAMPLE_HEADLINES)}"
        
        article = NewsArticle(
            id=hashlib.md5(f"{stock_symbol}{i}{datetime.now()}".encode()).hexdigest()[:16],
            headline=headline,
            source=random.choice(NEWS_SOURCES),
            url=f"https://news.example.com/article/{i}",
            timestamp=datetime.now() - timedelta(hours=random.randint(1, 24)),
            companies=[stock_symbol],
            sentiment=sentiment,
            relevance_score=round(random.uniform(0.6, 1.0), 2)
        )
        articles.append(article)
    
    return articles

def analyze_sentiment_impact(articles: List[NewsArticle]) -> MarketImpactScore:
    positive_count = sum(1 for a in articles if a.sentiment == SentimentType.POSITIVE)
    negative_count = sum(1 for a in articles if a.sentiment == SentimentType.NEGATIVE)
    
    total = len(articles)
    prob_up = (positive_count + 0.5 * (total - positive_count - negative_count)) / total
    prob_down = 1 - prob_up
    
    impact_score = round(abs(prob_up - 0.5) * 200, 2)
    
    stock_symbol = articles[0].companies[0] if articles else "UNKNOWN"
    company_name = next((s["name"] for s in NIFTY50_STOCKS if s["symbol"] == stock_symbol), stock_symbol)
    
    return MarketImpactScore(
        stock_symbol=stock_symbol,
        company_name=company_name,
        impact_score=impact_score,
        probability_up=round(prob_up, 2),
        probability_down=round(prob_down, 2),
        predicted_change_range=f"{round(prob_up * 5, 1)}-{round(prob_up * 8, 1)}%",
        timeframe="48 hours",
        confidence=round(random.uniform(0.7, 0.95), 2)
    )

def generate_advice(impact: MarketImpactScore, articles: List[NewsArticle], risk_profile: RiskProfile) -> InvestmentAdvice:
    if impact.probability_up > 0.7:
        action = "BUY"
        reasoning = f"Strong positive signal detected with {int(impact.probability_up * 100)}% probability of upward movement."
    elif impact.probability_up > 0.55:
        action = "WATCH" if risk_profile == RiskProfile.CONSERVATIVE else "BUY"
        reasoning = f"Moderate positive signal. {int(impact.probability_up * 100)}% chance of price increase."
    elif impact.probability_down > 0.65:
        action = "SELL"
        reasoning = f"Negative sentiment dominates with {int(impact.probability_down * 100)}% probability of decline."
    else:
        action = "HOLD"
        reasoning = "Mixed signals. Recommend holding position and monitoring developments."
    
    risk_levels = {
        RiskProfile.CONSERVATIVE: "Low",
        RiskProfile.MODERATE: "Medium",
        RiskProfile.AGGRESSIVE: "High"
    }
    
    key_headlines = [a.headline for a in sorted(articles, key=lambda x: x.relevance_score, reverse=True)[:3]]
    
    current_price = 1000
    stop_loss = round(current_price * 0.95, 2) if action == "BUY" else None
    target_price = round(current_price * (1 + float(impact.predicted_change_range.split('-')[1].strip('%')) / 100), 2) if action == "BUY" else None
    
    return InvestmentAdvice(
        stock_symbol=impact.stock_symbol,
        action=action,
        reasoning=reasoning,
        key_headlines=key_headlines,
        risk_level=risk_levels[risk_profile],
        stop_loss=stop_loss,
        target_price=target_price,
        timestamp=datetime.now()
    )

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def root():
    return {
        "service": "MarketPulse AI",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "stocks": "/api/stocks",
            "analyze": "/api/analyze"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/stocks")
def get_available_stocks():
    return {
        "stocks": NIFTY50_STOCKS,
        "total": len(NIFTY50_STOCKS),
        "index": "NIFTY 50"
    }

@app.post("/api/analyze", response_model=NewsAnalysisResponse)
def analyze_stock(request: StockAnalysisRequest):
    stock = request.stock_symbol.upper()
    risk_profile = request.risk_profile
    
    valid_symbols = [s["symbol"] for s in NIFTY50_STOCKS]
    if stock not in valid_symbols:
        raise HTTPException(status_code=404, detail=f"Stock symbol {stock} not found")
    
    articles = generate_mock_news(stock, count=10)
    impact = analyze_sentiment_impact(articles)
    advice = generate_advice(impact, articles, risk_profile)
    
    return NewsAnalysisResponse(
        articles=articles[:5],
        impact_scores=[impact],
        advice=advice
    )

@app.get("/api/news/{stock_symbol}")
def get_stock_news(stock_symbol: str, limit: int = 20):
    stock = stock_symbol.upper()
    
    valid_symbols = [s["symbol"] for s in NIFTY50_STOCKS]
    if stock not in valid_symbols:
        raise HTTPException(status_code=404, detail=f"Stock symbol {stock} not found")
    
    articles = generate_mock_news(stock, count=limit)
    
    return {
        "stock_symbol": stock,
        "articles": articles,
        "count": len(articles)
    }

@app.get("/api/impact-score/{stock_symbol}")
def get_impact_score(stock_symbol: str):
    stock = stock_symbol.upper()
    
    valid_symbols = [s["symbol"] for s in NIFTY50_STOCKS]
    if stock not in valid_symbols:
        raise HTTPException(status_code=404, detail=f"Stock symbol {stock} not found")
    
    articles = generate_mock_news(stock, count=10)
    impact = analyze_sentiment_impact(articles)
    
    return impact

@app.get("/api/alerts")
def get_alerts(user_id: str = "default_user", unread_only: bool = False):
    alerts = []
    
    for i, stock in enumerate(NIFTY50_STOCKS[:5]):
        alert = Alert(
            id=hashlib.md5(f"{stock['symbol']}{i}".encode()).hexdigest()[:16],
            stock_symbol=stock["symbol"],
            alert_type="HIGH_IMPACT" if random.random() > 0.5 else "MODERATE_IMPACT",
            message=f"Significant news detected for {stock['name']}. Impact score: {random.randint(60, 95)}",
            impact_score=round(random.uniform(60, 95), 2),
            timestamp=datetime.now() - timedelta(minutes=random.randint(5, 120)),
            is_read=False if unread_only or random.random() > 0.5 else True
        )
        alerts.append(alert)
    
    if unread_only:
        alerts = [a for a in alerts if not a.is_read]
    
    return {
        "alerts": sorted(alerts, key=lambda x: x.timestamp, reverse=True),
        "total": len(alerts),
        "unread": len([a for a in alerts if not a.is_read])
    }

@app.get("/api/dashboard")
def get_dashboard_data(user_id: str = "default_user"):
    top_stocks = []
    for stock in NIFTY50_STOCKS[:5]:
        articles = generate_mock_news(stock["symbol"], count=5)
        impact = analyze_sentiment_impact(articles)
        top_stocks.append({
            "symbol": stock["symbol"],
            "name": stock["name"],
            "impact_score": impact.impact_score,
            "probability_up": impact.probability_up,
            "trend": "up" if impact.probability_up > 0.5 else "down"
        })
    
    return {
        "user_id": user_id,
        "timestamp": datetime.now(),
        "top_movers": sorted(top_stocks, key=lambda x: x["impact_score"], reverse=True),
        "total_alerts": random.randint(3, 12),
        "unread_alerts": random.randint(1, 5),
        "watchlist_count": random.randint(5, 15)
    }

@app.on_event("startup")
async def startup_event():
    print("=" * 70)
    print("🚀 MarketPulse AI Backend Starting...")
    print("=" * 70)
    print(f"📊 Loaded {len(NIFTY50_STOCKS)} stocks from NIFTY 50")
    print("🔗 API Documentation: https://marketpullse-ai-3.onrender.com/docs")
    print("💡 Interactive API: https://marketpullse-ai-3.onrender.com/redoc")
    print("=" * 70)
@app.get("/api/price-history/{stock_symbol}")
def get_price_history(stock_symbol: str, days: int = 30):
    """Get historical price data for a stock"""
    stock = stock_symbol.upper()
    
    # Mock data (replace with real API call)
    prices = []
    for i in range(days):
        prices.append({
            "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
            "price": round(1000 + random.uniform(-50, 50), 2),
            "volume": random.randint(1000000, 5000000)
        })
    
    return {
        "stock_symbol": stock,
        "prices": prices,
        "period_days": days
    }
    class CompareStocksRequest(BaseModel):
    stock_symbols: List[str]

@app.post("/api/compare-stocks")
def compare_stocks(request: CompareStocksRequest):
    """Compare multiple stocks side by side"""
    results = []
    
    for symbol in request.stock_symbols:
        articles = generate_mock_news(symbol, count=5)
        impact = analyze_sentiment_impact(articles)
        
        results.append({
            "symbol": symbol,
            "impact_score": impact.impact_score,
            "probability_up": impact.probability_up,
            "sentiment": "bullish" if impact.probability_up > 0.6 else "bearish"
        })
    
    return {
        "comparison": results,
        "best_performer": max(results, key=lambda x: x["impact_score"])
    }
    @app.get("/api/trending")
def get_trending_stocks(limit: int = 5):
    """Get top trending stocks based on news volume and impact"""
    trending = []
    
    for stock in NIFTY50_STOCKS[:limit]:
        articles = generate_mock_news(stock["symbol"], count=10)
        impact = analyze_sentiment_impact(articles)
        
        trending.append({
            "symbol": stock["symbol"],
            "name": stock["name"],
            "impact_score": impact.impact_score,
            "news_count": len(articles),
            "trend": "up" if impact.probability_up > 0.5 else "down"
        })
    
    # Sort by impact score
    trending.sort(key=lambda x: x["impact_score"], reverse=True)
    
    return {
        "trending_stocks": trending,
        "timestamp": datetime.now()
    }
    @app.get("/api/search")
def search_stocks(query: str):
    """Search for stocks by name or symbol"""
    query = query.upper()
    
    results = [
        stock for stock in NIFTY50_STOCKS 
        if query in stock["symbol"].upper() or query in stock["name"].upper()
    ]
    
    return {
        "query": query,
        "results": results,
        "count": len(results)
    }
    @app.get("/api/sector-analysis/{sector}")
def get_sector_analysis(sector: str):
    """Analyze stocks by sector (Banking, IT, Energy, etc.)"""
    
    # Mock sector mapping (expand this based on your needs)
    sectors = {
        "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK"],
        "it": ["TCS", "INFY"],
        "energy": ["RELIANCE"]
    }
    
    sector_stocks = sectors.get(sector.lower(), [])
    
    if not sector_stocks:
        raise HTTPException(status_code=404, detail=f"Sector {sector} not found")
    
    analysis = []
    for symbol in sector_stocks:
        articles = generate_mock_news(symbol, count=5)
        impact = analyze_sentiment_impact(articles)
        
        analysis.append({
            "symbol": symbol,
            "impact_score": impact.impact_score,
            "probability_up": impact.probability_up
        })
    
    avg_impact = sum(s["impact_score"] for s in analysis) / len(analysis)
    
    return {
        "sector": sector,
        "stocks": analysis,
        "average_impact": round(avg_impact, 2),
        "sector_sentiment": "bullish" if avg_impact > 50 else "bearish"
    }
    class Portfolio(BaseModel):
    stocks: List[dict]  # [{"symbol": "RELIANCE", "quantity": 10, "buy_price": 1000}]

@app.post("/api/portfolio-analysis")
def analyze_portfolio(portfolio: Portfolio):
    """Analyze an entire portfolio"""
    
    portfolio_analysis = []
    total_value = 0
    total_gain_loss = 0
    
    for holding in portfolio.stocks:
        symbol = holding["symbol"]
        quantity = holding["quantity"]
        buy_price = holding["buy_price"]
        
        # Get current analysis
        articles = generate_mock_news(symbol, count=5)
        impact = analyze_sentiment_impact(articles)
        
        # Mock current price (replace with real data)
        current_price = buy_price * (1 + random.uniform(-0.1, 0.2))
        
        gain_loss = (current_price - buy_price) * quantity
        value = current_price * quantity
        
        portfolio_analysis.append({
            "symbol": symbol,
            "quantity": quantity,
            "buy_price": buy_price,
            "current_price": round(current_price, 2),
            "value": round(value, 2),
            "gain_loss": round(gain_loss, 2),
            "gain_loss_percent": round((gain_loss / (buy_price * quantity)) * 100, 2),
            "recommendation": "HOLD" if impact.probability_up > 0.5 else "CONSIDER_SELLING"
        })
        
        total_value += value
        total_gain_loss += gain_loss
    
    return {
        "portfolio": portfolio_analysis,
        "total_value": round(total_value, 2),
        "total_gain_loss": round(total_gain_loss, 2),
        "gain_loss_percent": round((total_gain_loss / total_value) * 100, 2) if total_value > 0 else 0
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)