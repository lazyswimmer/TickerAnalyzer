import yfinance as yf
import pandas as pd
import numpy as np
from financedatabase import Equities
from polygon import RESTClient  # Only used if user adds API key
import statsmodels.api as sm
import alpha_vantage as av   # Not used unless key added


# ============================================================
# 1. Quantitative (price, financials, ratios, valuation metrics)
# ============================================================

def retrieve_quantitative_data(ticker):
    stock = yf.Ticker(ticker)

    data = {
        "price_history": stock.history(period="5y"),
        "info": stock.info,
        "balance_sheet": stock.balance_sheet,
        "income_statement": stock.income_stmt,
        "cashflow": stock.cashflow,
        "quarterly_financials": stock.quarterly_financials,
        "major_holders": stock.major_holders,
        "institutional_holders": stock.institutional_holders,
    }

    return data



# ============================================================
# 2. Sentiment Data (news headlines from YFinance)
# ============================================================

def retrieve_sentiment_data(ticker):
    stock = yf.Ticker(ticker)

    # YFinance provides basic news articles
    news = stock.news

    df = pd.DataFrame(news)
    return df



# ============================================================
# 3. Technical Indicators (RSI, MACD, MAs, Volatility, etc.)
# ============================================================

def retrieve_technical_indicators(ticker):
    df = yf.download(ticker, period="1y")

    # Moving averages
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Volatility
    df["Volatility"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)

    return df



# ============================================================
# 4. Broad Market Data (SPY, QQQ, sector ETFs)
# ============================================================

def retrieve_broad_market_data():
    tickers = ["SPY", "QQQ", "IWM", "DIA",
               "XLF", "XLK", "XLY", "XLE", "XLI",
               "XLV", "XLP", "XLU", "XLRE", "XLB"]

    market_data = yf.download(tickers, period="1y")["Close"]

    return market_data



# ============================================================
# 5. Industry & Competitor Data
#    Using financedatabase (free, offline)
# ============================================================

def retrieve_industry_competitor_data(ticker):
    # Get industry/sector from Yahoo Finance
    info = yf.Ticker(ticker).info
    industry = info.get("industry")
    sector = info.get("sector")

    # Load the financedatabase equities dataset (class-based)
    eq = Equities()
    df = eq.select()   # THIS IS THE CORRECT METHOD IN YOUR VERSION

    # Filter competitors by industry or sector
    competitors = df[
        (df["industry"] == industry) |
        (df["sector"] == sector)
    ]

    return competitors.head(20)


# ============================================================
# 6. Risk Factors (10-K keywords + Macro data)
# ============================================================

def retrieve_risk_factors(ticker):
    stock = yf.Ticker(ticker)

    keywords = ["risk", "inflation", "recession", "competition",
                "regulation", "supply", "rates", "litigation"]

    # Use company long business summary for lightweight risk extraction
    summary = stock.info.get("longBusinessSummary", "")

    found_keywords = [k for k in keywords if k.lower() in summary.lower()]

    return {
        "long_summary": summary,
        "keyword_flags": found_keywords,
    }



# ============================================================
# 7. Assess Stock Edge (simple rule system)
# ============================================================

def assess_stock_edge(ticker):
    quant = retrieve_quantitative_data(ticker)
    tech = retrieve_technical_indicators(ticker)

    price = quant["price_history"]["Close"]

    edge = {
        "uptrend": price.iloc[-1] > price.rolling(200).mean().iloc[-1],
        "above_SMA_50": tech["Close"].iloc[-1] > tech["SMA_50"].iloc[-1],
        "positive_MACD": tech["MACD"].iloc[-1] > tech["Signal"].iloc[-1],
        "low_volatility_rank": tech["Volatility"].iloc[-1]
    }

    return edge



# ============================================================
# 8. Stock Niche Scoring
# ============================================================

def check_stock_niche(ticker):
    info = yf.Ticker(ticker).info

    moat_signals = {
        "brand": "brand" in str(info).lower(),
        "patents": info.get("numberOfPatents", 0),
        "market_share": info.get("marketCap", 0),
        "unique_product": any(word in str(info).lower()
                              for word in ["ecosystem", "unique", "monopoly"]),
    }

    return moat_signals



# ============================================================
# 9. Full Assessment
# ============================================================

def comprehensive_stock_assessment(ticker):
    return {
        "quantitative": retrieve_quantitative_data(ticker),
        "sentiment": retrieve_sentiment_data(ticker),
        "technical": retrieve_technical_indicators(ticker),
        "broad_market": retrieve_broad_market_data(),
        "industry_competitors": retrieve_industry_competitor_data(ticker),
        "risk_factors": retrieve_risk_factors(ticker),
        "edge": assess_stock_edge(ticker),
        "niche": check_stock_niche(ticker)
    }

assessment = comprehensive_stock_assessment(input().upper())
print(assessment["quantitative"])
print(assessment["sentiment"])
print(assessment["technical"])
print(assessment["broad_market"])
print(assessment["industry_competitors"])
print(assessment["risk_factors"])
print(assessment["edge"])
print(assessment["niche"])
