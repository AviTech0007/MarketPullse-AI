from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import os
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS to allow requests from your frontend
CORS(app, origins=[
    "https://market-pulse-hub3000.onrender.com",
    "https://pixel-perfect-ui6.onrender.com",
    "http://localhost:5173",
    "http://localhost:8080"
])

# Get API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY') Howdy! 
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "MarketPulse AI Backend is operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """
    Fetch real-time stock data using yfinance
    Returns: Current price, day's data, and basic info
    """
    try:
        # Fetch stock data from yfinance
        stock = yf.Ticker(symbol.upper())
        
        # Get current info
        info = stock.info
        
        # Get historical data for today
        hist = stock.history(period='1d', interval='1m')
        
        if hist.empty:
            # If intraday data not available, get 5 days of data
            hist = stock.history(period='5d')
        
        # Prepare chart data
        chart_data = []
        for timestamp, row in hist.iterrows():
            chart_data.append({
                'time': timestamp.isoformat(),
                'price': float(row['Close']),
                'volume': int(row['Volume'])
            })
        
        # Get current price (most recent close)
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else info.get('currentPrice', 0)
        
        # Calculate day change
        if len(hist) > 1:
            previous_close = float(hist['Close'].iloc[0])
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100
        else:
            previous_close = info.get('previousClose', current_price)
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0
        
        response_data = {
            'symbol': symbol.upper(),
            'name': info.get('longName', symbol.upper()),
            'currentPrice': current_price,
            'previousClose': previous_close,
            'change': round(change, 2),
            'changePercent': round(change_percent, 2),
            'dayHigh': float(info.get('dayHigh', 0)),
            'dayLow': float(info.get('dayLow', 0)),
            'volume': int(info.get('volume', 0)),
            'marketCap': info.get('marketCap', 0),
            'fiftyTwoWeekHigh': float(info.get('fiftyTwoWeekHigh', 0)),
            'fiftyTwoWeekLow': float(info.get('fiftyTwoWeekLow', 0)),
            'chartData': chart_data,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        response = {
            'error': str(e),
            'message': f'Failed to fetch data for {symbol}'
        }
        return jsonify(response), 500

@app.route('/api/stocks/search', methods=['GET'])
def search_stocks():
    """
    Search for stocks by symbol or name
    """
    query = request.args.get('q', '').upper()
    
    if not query:
        return jsonify({'error': 'Query parameter required'}), 400
    
    try:
        # Common stocks list (you can expand this)
        common_stocks = [
            {'symbol': 'AAPL', 'name': 'Apple Inc.'},
            {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
            {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
            {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
            {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
            {'symbol': 'NVDA', 'name': 'NVIDIA Corporation'},
            {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.'},
            {'symbol': 'V', 'name': 'Visa Inc.'},
            {'symbol': 'WMT', 'name': 'Walmart Inc.'},
        ]
        
        # Filter based on query
        results = [
            stock for stock in common_stocks
            if query in stock['symbol'] or query in stock['name'].upper()
        ]
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/news/<symbol>', methods=['GET'])
def get_stock_news(symbol):
    """
    Fetch news for a specific stock using News API
    """
    if not NEWS_API_KEY:
        return jsonify({
            'error': 'News API key not configured',
            'articles': []
        }), 500
    
    try:
        # Get company name from yfinance
        stock = yf.Ticker(symbol.upper())
        company_name = stock.info.get('longName', symbol.upper())
        
        # Fetch news from News API
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': f'{company_name} OR {symbol}',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 10,
            'apiKey': NEWS_API_KEY
        }
        
        response = requests.get(url, params=params)
        news_data = response.json()
        
        if response.status_code != 200:
            return jsonify({
                'error': 'Failed to fetch news',
                'articles': []
            }), response.status_code
        
        articles = []
        for article in news_data.get('articles', []):
            articles.append({
                'title': article.get('title'),
                'description': article.get('description'),
                'url': article.get('url'),
                'source': article.get('source', {}).get('name'),
                'publishedAt': article.get('publishedAt'),
                'urlToImage': article.get('urlToImage')
            })
        
        return jsonify({
            'symbol': symbol.upper(),
            'articles': articles
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'articles': []
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_stock():
    """
    Generate AI prediction using Groq API
    Requires: symbol, riskProfile in request body
    """
    if not GROQ_API_KEY:
        return jsonify({
            'error': 'Groq API key not configured'
        }), 500
    
    try:
        data = request.get_json()
        symbol = data.get('symbol', '').upper()
        risk_profile = data.get('riskProfile', 'medium')
        
        if not symbol:
            return jsonify({'error': 'Symbol required'}), 400
        
        # Fetch real stock data
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period='1mo')
        
        # Calculate metrics
        current_price = info.get('currentPrice', 0)
        change_percent = info.get('52WeekChange', 0) * 100
        volume = info.get('volume', 0)
        avg_volume = info.get('averageVolume', 0)
        
        # Prepare prompt for Groq
        prompt = f"""
        Analyze {symbol} ({info.get('longName', symbol)}) stock and provide a trading recommendation.
        
        Current Data:
        - Current Price: ${current_price}
        - 52-Week Change: {change_percent:.2f}%
        - Volume: {volume:,}
        - Average Volume: {avg_volume:,}
        - Market Cap: ${info.get('marketCap', 0):,}
        
        Risk Profile: {risk_profile}
        
        Provide:
        1. Recommendation (BUY/HOLD/SELL)
        2. Confidence Level (0-100)
        3. Reasoning (2-3 sentences)
        4. Risk Assessment
        
        Format as JSON with keys: recommendation, confidence, reasoning, risk
        """
        
        # Call Groq API
        groq_url = 'https://api.groq.com/openai/v1/chat/completions'
        
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        groq_payload = {
            'model': 'llama-3.3-70b-versatile',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': 1024
        }
        
        groq_response = requests.post(groq_url, headers=headers, json=groq_payload)
        groq_data = groq_response.json()
        
        if groq_response.status_code != 200:
            return jsonify({
                'error': 'Failed to generate prediction',
                'recommendation': 'HOLD',
                'confidence': 50,
                'reasoning': 'Unable to generate AI prediction at this time.'
            }), 500
        
        # Extract text from Groq response
        ai_text = groq_data.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        return jsonify({
            'symbol': symbol,
            'prediction': ai_text,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'recommendation': 'HOLD',
            'confidence': 50,
            'reasoning': 'Error generating prediction.'
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=False)
    