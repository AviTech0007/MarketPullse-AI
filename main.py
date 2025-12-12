"""
MarketPulse AI - Production Backend with Real APIs
Real-time financial news analysis with YFinance and NewsAPI integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from enum import Enum
import random
import hashlib
import os
import requests

# YFinance for real stock data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available, using mock data")

# TextBlob for sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: textblob not available, using basic sentiment")

app = FastAPI(
    title="MarketPulse AI API",
    description="AI-powered financial news analysis and investment advisory with real-time data",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY","ae892df256774192b7bfd3ac5d072cc9")
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
    current_price: Optional[float] = None
    timestamp: datetime
class StockAnalysisRequest(BaseModel):
    stock_symbol: str
    risk_profile: Optional[RiskProfile] = RiskProfile.MODERATE

class NewsAnalysisResponse(BaseModel):
    articles: List[NewsArticle]
    impact_scores: List[MarketImpactScore]
    advice: InvestmentAdvice

class StockPriceData(BaseModel):
    symbol: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    volume: int
    high_52week: float
    low_52week: float
    market_cap: Optional[int] = None
    company_name: str

# ============================================================================
# STOCK DATA
# ============================================================================

NIFTY50_STOCKS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "ticker": "RELIANCE.NS"},
    {"symbol": "TCS", "name": "Tata Consultancy Services", "ticker": "TCS.NS"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "ticker": "HDFCBANK.NS"},
    {"symbol": "INFY", "name": "Infosys Ltd", "ticker": "INFY.NS"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "ticker": "ICICIBANK.NS"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "ticker": "HINDUNILVR.NS"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "ticker": "BHARTIARTL.NS"},
    {"symbol": "ITC", "name": "ITC Ltd", "ticker": "ITC.NS"},
    {"symbol": "SBIN", "name": "State Bank of India", "ticker": "SBIN.NS"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "ticker": "KOTAKBANK.NS"}
]

# ============================================================================
# REAL API FUNCTIONS
# ============================================================================

def get_real_stock_data(stock_symbol: str) -> Optional[dict]:
    """Fetch real stock price data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        stock_info = next((s for s in NIFTY50_STOCKS if s["symbol"] == stock_symbol), None)
        if not stock_info:
            return None
        
        ticker = yf.Ticker(stock_info["ticker"])
        info = ticker.info
        hist = ticker.history(period="2d")
        
        if hist.empty:
            return None
        
        current_price = float(hist['Close'].iloc[-1])
        previous_close = float(info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price))
        
        change = current_price - previous_close
        change_percent = (change / previous_close * 100) if previous_close != 0 else 0
        
        return {
            "symbol": stock_symbol,
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
            "high_52week": round(float(info.get('fiftyTwoWeekHigh', current_price * 1.2)), 2),
            "low_52week": round(float(info.get('fiftyTwoWeekLow', current_price * 0.8)), 2),
            "market_cap": info.get('marketCap'),
            "company_name": stock_info["name"]
        }
    except Exception as e:
        print(f"Error fetching stock data for {stock_symbol}: {e}")
        return None

def analyze_sentiment_advanced(text: str) -> SentimentType:
    """Advanced sentiment analysis using TextBlob"""
    if not text:
        return SentimentType.NEUTRAL
    
    if TEXTBLOB_AVAILABLE:
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return SentimentType.POSITIVE
            elif polarity < -0.1:
                return SentimentType.NEGATIVE
            else:
                return SentimentType.NEUTRAL
        except Exception as e:
            print(f"TextBlob error: {e}")
    
    # Fallback to keyword-based sentiment
    text_lower = text.lower()
    positive_words = ["surge", "profit", "growth", "gain", "up", "rise", "bullish", "strong", "beat", "outperform"]
    negative_words = ["fall", "loss", "decline", "down", "drop", "bearish", "weak", "crash", "underperform", "miss"]
    
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return SentimentType.POSITIVE
    elif neg_count > pos_count:
        return SentimentType.NEGATIVE
    else:
        return SentimentType.NEUTRAL

def fetch_real_news(stock_symbol: str, company_name: str, count: int = 10) -> List[NewsArticle]:
    """Fetch real news from NewsAPI"""
    
    if not NEWS_API_KEY:
        print("NewsAPI key not configured, using mock data")
        return generate_mock_news(stock_symbol, count)
    
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": company_name,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(count * 2, 20),  # Fetch more to filter
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"NewsAPI error: {response.status_code}")
            return generate_mock_news(stock_symbol, count)
        
        data = response.json()
        articles = []
        
        for article in data.get("articles", []):
            if not article.get("title") or article.get("title") == "[Removed]":
                continue
            
            title = article.get("title", "")
            description = article.get("description", "")
            full_text = f"{title} {description}"
            
            sentiment = analyze_sentiment_advanced(full_text)
            
            try:
                published_at = article.get("publishedAt", "")
                if published_at:
                    timestamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.now()
            except:
                timestamp = datetime.now()
            
            relevance_score = 0.9 if company_name.lower() in title.lower() else 0.7
            
            news_article = NewsArticle(
                id=hashlib.md5(article["url"].encode()).hexdigest()[:16],
                headline=title,
                source=article.get("source", {}).get("name", "Unknown"),
                url=article.get("url", ""),
                timestamp=timestamp,
                companies=[stock_symbol],
                sentiment=sentiment,
                relevance_score=relevance_score
            )
            articles.append(news_article)
            
            if len(articles) >= count:
                break
        
        if not articles:
            return generate_mock_news(stock_symbol, count)
        
        return articles
        
    except Exception as e:
        print(f"Error fetching real news: {e}")
        return generate_mock_news(stock_symbol, count)

def generate_mock_news(stock_symbol: str, count: int = 5) -> List[NewsArticle]:
    """Generate mock news articles as fallback"""
    articles = []
    company_name = next((s["name"] for s in NIFTY50_STOCKS if s["symbol"] == stock_symbol), stock_symbol)
    
    headlines = [
        f"{company_name} announces record quarterly profits, beats analyst estimates",
        f"{company_name} faces regulatory scrutiny over market practices",
        f"{company_name} plans major expansion into renewable energy sector",
        f"{company_name} reports decline in profit margins amid rising costs",
        f"{company_name} launches innovative product line targeting millennials",
        f"{company_name} CEO announces strategic partnership with global leader",
        f"{company_name} stock surges on strong earnings guidance",
        f"{company_name} faces headwinds from global economic slowdown",
        f"{company_name} invests heavily in digital transformation initiatives",
        f"{company_name} receives positive analyst upgrade from major firm"
    ]
    
    sources = ["Economic Times", "Moneycontrol", "Bloomberg", "Reuters", "LiveMint", "Business Standard"]
    
    for i in range(count):
        sentiment = random.choice([SentimentType.POSITIVE, SentimentType.NEGATIVE, SentimentType.NEUTRAL])
        
        article = NewsArticle(
            id=hashlib.md5(f"{stock_symbol}{i}{datetime.now()}".encode()).hexdigest()[:16],
            headline=random.choice(headlines),
            source=random.choice(sources),
            url=f"https://news.example.com/article/{i}",
            timestamp=datetime.now() - timedelta(hours=random.randint(1, 48)),
            companies=[stock_symbol],
            sentiment=sentiment,
            relevance_score=round(random.uniform(0.6, 1.0), 2)
        )
        articles.append(article)
    
    return articles

def analyze_sentiment_impact(articles: List[NewsArticle]) -> MarketImpactScore:
    """Analyze news sentiment and calculate market impact"""
    if not articles:
        return None
    
    positive_count = sum(1 for a in articles if a.sentiment == SentimentType.POSITIVE)
    negative_count = sum(1 for a in articles if a.sentiment == SentimentType.NEGATIVE)
    neutral_count = len(articles) - positive_count - negative_count
    
    total = len(articles)
    
    # Weighted probability calculation
    prob_up = (positive_count + 0.3 * neutral_count) / total if total > 0 else 0.5
    prob_down = 1 - prob_up
    
    # Impact score: 0-100 scale based on sentiment strength
    sentiment_strength = abs(positive_count - negative_count) / total if total > 0 else 0
    impact_score = sentiment_strength * 100
    
    # Confidence based on volume and recency
    recent_articles = sum(1 for a in articles if (datetime.now() - a.timestamp).days < 1)
    confidence = min(0.95, 0.6 + (recent_articles / total * 0.35))
    
    stock_symbol = articles[0].companies[0]
    company_name = next((s["name"] for s in NIFTY50_STOCKS if s["symbol"] == stock_symbol), stock_symbol)
    
    predicted_min = round(prob_up * 3, 1)
    predicted_max = round(prob_up * 7, 1)
    
    return MarketImpactScore(
        stock_symbol=stock_symbol,
        company_name=company_name,
        impact_score=round(impact_score, 2),
        probability_up=round(prob_up, 2),
        probability_down=round(prob_down, 2),
        predicted_change_range=f"{predicted_min}-{predicted_max}%",
        timeframe="48 hours",
        confidence=round(confidence, 2)
    )

def generate_advice(impact: MarketImpactScore, articles: List[NewsArticle], 
                   risk_profile: RiskProfile, stock_data: Optional[dict]) -> InvestmentAdvice:
    """Generate investment advice based on analysis"""
    
    # Determine action based on probability and risk profile
    if impact.probability_up > 0.7:
        action = "BUY"
        reasoning = f"Strong positive signal detected with {int(impact.probability_up * 100)}% probability of upward movement. Recent news sentiment is highly favorable."
    elif impact.probability_up > 0.6:
        if risk_profile == RiskProfile.CONSERVATIVE:
            action = "WATCH"
            reasoning = f"Moderate positive signal ({int(impact.probability_up * 100)}% probability up). Conservative investors should monitor closely before entering."
        else:
            action = "BUY"
            reasoning = f"Moderate positive signal with {int(impact.probability_up * 100)}% upward probability. Good opportunity for moderate to aggressive investors."
    elif impact.probability_down > 0.7:
        action = "SELL"
        reasoning = f"Strong negative sentiment detected with {int(impact.probability_down * 100)}% probability of decline. Consider reducing exposure."
    elif impact.probability_down > 0.6:
        action = "HOLD"
        reasoning = f"Cautious outlook with {int(impact.probability_down * 100)}% downward probability. Hold existing positions and reassess."
    else:
        action = "HOLD"
        reasoning = "Mixed signals in market sentiment. Recommend maintaining current positions while monitoring developments closely."
    
    # Risk level based on profile
    risk_levels = {
        RiskProfile.CONSERVATIVE: "Low",
        RiskProfile.MODERATE: "Medium",
        RiskProfile.AGGRESSIVE: "High"
    }
    
    # Get top 3 most relevant headlines
    key_headlines = [a.headline for a in sorted(articles, key=lambda x: x.relevance_score, reverse=True)[:3]]
    
    # Calculate price targets using REAL data if available
    current_price = None
    stop_loss = None
    target_price = None
    if stock_data:
        current_price = stock_data["current_price"]
        
        if action == "BUY":
            # Stop loss: 5% below current price
            stop_loss = round(current_price * 0.95, 2)
            # Target: Based on predicted range
            max_gain = float(impact.predicted_change_range.split('-')[1].strip('%'))
            target_price = round(current_price * (1 + max_gain / 100), 2)
        elif action == "SELL":
            # Target for selling: current price is the target
            target_price = current_price
    
    return InvestmentAdvice(
        stock_symbol=impact.stock_symbol,
        action=action,
        reasoning=reasoning,
        key_headlines=key_headlines,
        risk_level=risk_levels[risk_profile],
        stop_loss=stop_loss,
        target_price=target_price,
        current_price=current_price,
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
        "version": "2.0.0",
        "features": {
            "real_stock_prices": YFINANCE_AVAILABLE,
            "real_news": NEWS_API_KEY is not None,
            "advanced_sentiment": TEXTBLOB_AVAILABLE
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "stocks": "/api/stocks",
            "analyze": "/api/analyze",
            "stock_price": "/api/stock-price/{symbol}"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "apis_configured": {
            "yfinance": YFINANCE_AVAILABLE,
            "newsapi": NEWS_API_KEY is not None,
            "textblob": TEXTBLOB_AVAILABLE
        }
    }

@app.get("/api/stocks")
def get_available_stocks():
    return {
        "stocks": [{"symbol": s["symbol"], "name": s["name"]} for s in NIFTY50_STOCKS],
        "total": len(NIFTY50_STOCKS),
        "index": "NIFTY 50"
    }

@app.get("/api/stock-price/{stock_symbol}", response_model=StockPriceData)
def get_stock_price(stock_symbol: str):
    """Get real-time stock price data from Yahoo Finance"""
    stock = stock_symbol.upper()
    
    valid_symbols = [s["symbol"] for s in NIFTY50_STOCKS]
    if stock not in valid_symbols:
        raise HTTPException(status_code=404, detail=f"Stock symbol {stock} not found")
    
    stock_data = get_real_stock_data(stock)
    
    if not stock_data:
        raise HTTPException(status_code=503, detail=f"Could not fetch real-time data for {stock}. Service may be temporarily unavailable.")
    
    return StockPriceData(**stock_data)

@app.post("/api/analyze", response_model=NewsAnalysisResponse)
def analyze_stock(request: StockAnalysisRequest):
    """
    Main analysis endpoint - combines real stock data and news sentiment
    to provide AI-powered investment recommendations
    """
    stock = request.stock_symbol.upper()
    risk_profile = request.risk_profile
    
    valid_symbols = [s["symbol"] for s in NIFTY50_STOCKS]
    if stock not in valid_symbols:
        raise HTTPException(status_code=404, detail=f"Stock symbol {stock} not found")
    
    # Get company name
    company_name = next((s["name"] for s in NIFTY50_STOCKS if s["symbol"] == stock), stock)
    
    # Fetch REAL news (or mock as fallback)
    articles = fetch_real_news(stock, company_name, count=10)
    
    # Get REAL stock price data
    stock_data = get_real_stock_data(stock)
    
    # Analyze sentiment impact
    impact = analyze_sentiment_impact(articles)
    
    # Generate investment advice
    advice = generate_advice(impact, articles, risk_profile, stock_data)
    
    return NewsAnalysisResponse(
        articles=articles[:5],  # Return top 5 articles
        impact_scores=[impact],
        advice=advice
    )

@app.get("/api/news/{stock_symbol}")
def get_stock_news(stock_symbol: str, limit: int = 20):
    """Get recent news articles for a specific stock"""
    stock = stock_symbol.upper()
    
    valid_symbols = [s["symbol"] for s in NIFTY50_STOCKS]
    if stock not in valid_symbols:
        raise HTTPException(status_code=404, detail=f"Stock symbol {stock} not found")
    
    company_name = next((s["name"] for s in NIFTY50_STOCKS if s["symbol"] == stock), stock)
    articles = fetch_real_news(stock, company_name, count=limit)
    
    return {
        "stock_symbol": stock,
        "company_name": company_name,
        "articles": articles,
        "count": len(articles),
        "using_real_data": NEWS_API_KEY is not None
    }

@app.get("/api/impact-score/{stock_symbol}")
def get_impact_score(stock_symbol: str):
    """Get market impact score for a stock based on news sentiment"""
    stock = stock_symbol.upper()
    
    valid_symbols = [s["symbol"] for s in NIFTY50_STOCKS]
    if stock not in valid_symbols:
        raise HTTPException(status_code=404, detail=f"Stock symbol {stock} not found")
    
    company_name = next((s["name"] for s in NIFTY50_STOCKS if s["symbol"] == stock), stock)
    articles = fetch_real_news(stock, company_name, count=10)
    impact = analyze_sentiment_impact(articles)
    
    return impact

@app.get("/api/dashboard")
def get_dashboard_data(user_id: str = "default_user"):
    """Get dashboard with top movers and market overview"""
    top_stocks = []
    
    for stock in NIFTY50_STOCKS[:5]:
        try:
            stock_data = get_real_stock_data(stock["symbol"])
            articles = fetch_real_news(stock["symbol"], stock["name"], count=5)
            impact = analyze_sentiment_impact(articles)
            
            top_stocks.append({
                "symbol": stock["symbol"],
                "name": stock["name"],
                "current_price": stock_data["current_price"] if stock_data else None,
                "change_percent": stock_data["change_percent"] if stock_data else None,
                "impact_score": impact.impact_score,
                "probability_up": impact.probability_up,
                "trend": "up" if impact.probability_up > 0.5 else "down"
            })
        except:
            continue
    
    # Sort by impact score
    top_stocks.sort(key=lambda x: x["impact_score"], reverse=True)
    
    return {
        "user_id": user_id,
        "timestamp": datetime.now(),
        "top_movers": top_stocks,
        "market_status": "open" if datetime.now().hour < 15 else "closed",
        "using_real_data": YFINANCE_AVAILABLE and NEWS_API_KEY is not None
    }

@app.on_event("startup")
async def startup_event():
    print("=" * 70)
    print("🚀 MarketPulse AI Backend Starting (Production Mode)...")
    print("=" * 70)
    print(f"📊 Loaded {len(NIFTY50_STOCKS)} stocks from NIFTY 50")
    print(f"💹 YFinance: {'✅ Enabled' if YFINANCE_AVAILABLE else '❌ Disabled'}")
    print(f"📰 NewsAPI: {'✅ Enabled' if NEWS_API_KEY else '❌ Disabled (using mock data)'}")
    print(f"🧠 TextBlob Sentiment: {'✅ Enabled' if TEXTBLOB_AVAILABLE else '❌ Disabled'}")
    print("🔗 API Documentation: https://marketpullse-ai-15.onrender.com")
    print("=" * 70)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
