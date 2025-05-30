  
**Architecture Diagram:**  

<img width="682" alt="Screenshot 2025-05-30 at 10 33 17 PM" src="https://github.com/user-attachments/assets/a3facc41-bb03-476b-8844-1859187e7f9a" />

# **Financial Data Collectors \- Comprehensive Analysis**

This analysis covers three Python modules designed for collecting financial and economic data from different APIs, each serving specific roles in a broader financial data ingestion system.

## **1\. Alpha Vantage Collector (alpha\_vantage\_collector.py)**

### **Purpose**

The Alpha Vantage Collector is designed to gather real-time stock market data and earnings information. It focuses on individual stock performance metrics and fundamental analysis data.

### **Key Features**

#### **Class Structure**

* **Class**: AlphaVantageCollector  
* **Initialization**: Requires an API key for authentication  
* **Base URL**: https://www.alphavantage.co/query  
* **Logging**: Integrated error logging for debugging and monitoring

#### **Core Methods**

**1\. get\_stock\_quote(symbol) Method**

* **Function**: Retrieves real-time stock quotes  
* **API Endpoint**: Uses GLOBAL\_QUOTE function  
* **Returns**: Comprehensive stock data including:  
  * Current price and price changes  
  * Trading volume  
  * Percentage change  
  * Timestamp and source attribution

**2\. get\_earnings\_data(symbol) Method**

* **Function**: Fetches quarterly earnings reports  
* **API Endpoint**: Uses EARNINGS function  
* **Data Scope**: Last 4 quarters of earnings data  
* **Returns**: Detailed earnings metrics including:  
  * Reported vs. estimated EPS  
  * Earnings surprises and percentages  
  * Fiscal date information

---

## **2\. FRED Collector (fred\_collector.py)**

### **Purpose**

The Federal Reserve Economic Data (FRED) Collector is the most sophisticated module, designed to gather macroeconomic indicators that impact financial markets, particularly focusing on data relevant to Asian technology sector analysis.

### **Key Features**

#### **Class Structure**

* **Class**: FREDCollector  
* **Base URL**: https://api.stlouisfed.org/fred  
* **Comprehensive Rate Limiting**: 100ms minimum between requests  
* **Demo Mode**: Fallback when API key unavailable

#### **Economic Indicators Coverage**

**17 Key Economic Indicators:**

1. **GDP** \- Gross Domestic Product  
2. **INFLATION\_CPI** \- Consumer Price Index  
3. **UNEMPLOYMENT** \- Unemployment Rate  
4. **FEDERAL\_FUNDS\_RATE** \- Federal Reserve Interest Rate  
5. **VIX\_VOLATILITY** \- Market Volatility Index  
6. **USD\_INDEX** \- US Dollar Strength Index  
7. **TREASURY\_10Y** \- 10-Year Treasury Yield  
8. **TREASURY\_2Y** \- 2-Year Treasury Yield  
9. **TREASURY\_3M** \- 3-Month Treasury Rate  
10. **INDUSTRIAL\_PRODUCTION** \- Manufacturing Output  
11. **CONSUMER\_SENTIMENT** \- Consumer Confidence  
12. **HOUSING\_STARTS** \- New Housing Construction  
13. **RETAIL\_SALES** \- Consumer Spending  
14. **CORE\_PCE** \- Core Personal Consumption Expenditures  
15. **BREAKEVEN\_10Y** \- Inflation Expectations  
16. **CORPORATE\_AAA\_SPREAD** \- Corporate Bond Spreads  
17. **HIGH\_YIELD\_SPREAD** \- High-Yield Bond Risk Premium

#### **Advanced Data Processing**

**Dynamic Lookback Periods:**

* **Daily Data**: 30 days default  
* **Monthly Data**: 90 days (3 months)  
* **Quarterly Data**: 365 days (1 year)

**Relevance Scoring System:**

* Each indicator has a calculated relevance score (0.0-1.0)  
* **Highest Priority** (0.90-0.95): Federal Funds Rate, VIX Volatility  
* **High Priority** (0.75-0.89): Treasury rates, USD Index, Core PCE  
* **Medium Priority** (0.50-0.74): CPI, Consumer Sentiment, GDP  
* **Lower Priority** (0.40-0.49): Housing Starts, Retail Sales

#### **Metadata Enhancement**

Each data point includes:

* **Content Classification**: 'economic\_indicator'  
* **Tagging System**: Searchable tags for data organization  
* **Frequency Information**: Daily/Monthly/Quarterly classification  
* **Units**: Proper measurement units for each indicator  
* **Source Attribution**: Clear data provenance

---

## **3\. Yahoo Finance Collector (yahoo\_finance\_collector.py)**

### **Purpose**

The Yahoo Finance Collector focuses specifically on Asian technology stocks, providing market data for major tech companies in the Asia-Pacific region.

### **Key Features**

#### **Class Structure**

* **Class**: YahooFinanceCollector  
* **No API Key Required**: Uses the yfinance Python library  
* **Focus**: Asian technology sector stocks

#### **Stock Coverage**

**7 Major Asian Tech Stocks:**

1. **TSM** \- Taiwan Semiconductor Manufacturing  
2. **005930.KS** \- Samsung Electronics (Korean Exchange)  
3. **BABA** \- Alibaba Group  
4. **TCEHY** \- Tencent Holdings (ADR)  
5. **6758.T** \- Sony Corporation (Tokyo Exchange)  
6. **ASML** \- ASML Holding (European, but semiconductor-focused)  
7. **9984.T** \- SoftBank Group (Tokyo Exchange)

#### **Data Collection Methodology**

**Stock Information Gathered:**

* **Basic Identifiers**: Symbol, company name, sector  
* **Price Data**: Current price, previous close, price changes  
* **Valuation Metrics**: Market capitalization, P/E ratio  
* **Trading Activity**: Current volume, average volume  
* **Historical Context**: 5-day price history for trend analysis

---

### **Data Flow Integration**

These three collectors are designed to work together in a larger financial data pipeline:

1. **Alpha Vantage**: Provides detailed individual stock analysis  
2. **FRED**: Supplies macroeconomic context and market conditions  
3. **Yahoo Finance**: Offers broad market coverage with real-time pricing

## **Conclusion**

This suite of financial data collectors provides a comprehensive foundation for gathering both microeconomic (individual stocks) and macroeconomic (economic indicators) data. The modular design allows for flexible deployment, while the sophisticated error handling and rate limiting ensure reliable operation in production environments. The focus on Asian technology markets, combined with broad economic context, makes this particularly suitable for regional investment analysis and market research applications.

---

# **Financial Data Scrapers \- In-Depth Analysis**

## **Overview**

These two Python modules form part of a comprehensive financial data ingestion system designed to collect earnings information and regulatory filings from multiple sources. They represent a sophisticated approach to financial data aggregation with support for both US and international markets.

## **File 1: earnings\_scraper.py**

### **Purpose**

This module specializes in collecting earnings-related data from multiple sources, including earnings calendars, analyst coverage, and fundamental analysis. It's designed to provide comprehensive earnings intelligence for investment analysis.

### **Core Class: EarningsTranscriptScraper**

#### **Key Features:**

* **Multi-source earnings data collection**  
* **Robust error handling and fallback mechanisms**  
* **Anti-bot detection circumvention**  
* **Sentiment analysis capabilities**  
* **International ticker support**

### **Methods Analysis**

#### **1\. get\_earnings\_calendar(symbols: List\[str\])**

**Purpose**: Retrieves upcoming earnings dates and estimates for specified stock symbols.

**Implementation Strategy**:

* Uses yfinance API as primary data source  
* Handles multiple response formats (DataFrame vs Dictionary)  
* Implements three-tier fallback system:  
  1. Primary: ticker.calendar method  
  2. Secondary: ticker.info earnings data  
  3. Tertiary: Placeholder entries for tracking

**Data Structure Returned**:

{  
    'symbol': 'AAPL',  
    'earnings\_date': '2024-01-25',  
    'eps\_estimate': 2.11,  
    'revenue\_estimate': 118500000000,  
    'content\_type': 'earnings\_calendar',  
    'source': 'yahoo\_finance\_calendar',  
    'timestamp': '2024-01-20T10:30:00',  
    'tags': \['earnings\_calendar', 'aapl'\]  
}

#### **2\. scrape\_seeking\_alpha\_earnings(symbol: str)**

**Purpose**: Extracts earnings analysis articles from Seeking Alpha, a premium financial content platform.

**Anti-Bot Strategy**:

* **Enhanced Headers**: Mimics real browser behavior with comprehensive header set  
* **Session Management**: Uses requests.Session for cookie persistence  
* **Rate Limiting**: Random delays (1-3 seconds) between requests  
* **Multiple URL Patterns**: Tries different endpoint structures  
* **Graceful Degradation**: Falls back to alternative sources when blocked

**Scraping Logic**:

* Uses BeautifulSoup for HTML parsing  
* Employs multiple CSS selectors for robustness  
* Filters content using earnings-related keywords  
* Limits results to prevent aggressive scraping

**Fallback Mechanism**: When Seeking Alpha blocks access, automatically switches to \_get\_alternative\_earnings\_analysis()

#### **3\. \_get\_alternative\_earnings\_analysis(symbol: str)**

**Purpose**: Generates synthetic fundamental analysis when premium sources are unavailable.

**Analysis Metrics**:

* **P/E Ratio**: Forward and trailing price-to-earnings  
* **Price-to-Sales**: Revenue multiple analysis  
* **Profit Margins**: Operational efficiency metrics

**Intelligence Layer**: Provides automated interpretation of financial ratios with contextual commentary.

#### **4\. extract\_earnings\_sentiment(text: str)**

**Purpose**: Performs sentiment analysis on earnings-related text content.

**Methodology**:

* **Keyword-based Analysis**: Uses curated positive/negative keyword dictionaries  
* **Scoring Algorithm**: Calculates sentiment score from \-1 (very negative) to \+1 (very positive)  
* **Threshold-based Classification**: Categorizes sentiment as positive, negative, or neutral

**Keywords Categories**:

* **Positive**: 'beat', 'exceeded', 'strong', 'growth', 'outperformed'  
* **Negative**: 'missed', 'disappointed', 'weak', 'decline', 'underperformed'

#### **5\. get\_recent\_earnings\_reports(symbol: str, quarters: int \= 4\)**

**Purpose**: Retrieves historical quarterly earnings data for trend analysis.

## **File 2: sec\_filing\_scraper.py**

### **Purpose**

This module handles regulatory filing collection from multiple international exchanges, providing comprehensive compliance document access for fundamental analysis.

### **Core Class: SECFilingScraper**

#### **Multi-Exchange Architecture**

The scraper supports four major financial markets:

1. **SEC (US)**: Securities and Exchange Commission  
2. **KSE (Korea)**: Korea Stock Exchange  
3. **TSE (Japan)**: Tokyo Stock Exchange  
4. **SEHK (Hong Kong)**: Hong Kong Stock Exchange

### **Exchange-Specific Configurations**

#### **SEC (United States)**

* **Base URL**: https://www.sec.gov  
* **Filing Types**: 10-K (annual), 10-Q (quarterly), 8-K (current events)  
* **Search Method**: CIK-based lookup through EDGAR database

#### **KSE (Korea)**

* **System**: DART (Data Analysis, Retrieval and Transfer)  
* **URL**: https://kind.krx.co.kr/eng  
* **Document Types**: Annual reports, disclosure documents

#### **TSE (Japan)**

* **System**: TDnet (Timely Disclosure Network)  
* **URL**: https://www.release.tdnet.info  
* **Document Types**: Securities reports, earnings announcements

#### **SEHK (Hong Kong)**

* **System**: HKExNews  
* **URL**: https://www.hkexnews.hk  
* **Document Types**: Annual reports, interim reports

### **Methods Analysis**

#### **1\. get\_company\_filings(symbol: str, form\_types: List\[str\])**

**Purpose**: Main orchestration method that routes requests to appropriate exchange handlers.

**Logic Flow**:

1. Determine exchange from symbol format (e.g., '.KS' → KSE)  
2. Route to exchange-specific scraper  
3. Return standardized filing data structure

#### **2\. search\_company\_filings(company\_cik: str, form\_types: List\[str\])**

**Purpose**: SEC-specific filing search using Central Index Key (CIK).

**Process**:

1. **CIK Resolution**: Convert stock symbol to SEC CIK identifier  
2. **EDGAR Query**: Search SEC database for specified filing types  
3. **Result Parsing**: Extract filing metadata from HTML tables  
4. **URL Construction**: Build direct links to filing documents

#### **3\. Exchange-Specific Scrapers**

Each exchange has a dedicated scraper method:

##### **\_scrape\_kse\_filings(symbol: str) \- Korean Market**

* Extracts Korean company code from symbol  
* Interfaces with DART disclosure system  
* Handles Korean regulatory document formats

##### **\_scrape\_tse\_filings(symbol: str) \- Japanese Market**

* Processes Japanese stock codes  
* Connects to TDnet system  
* Manages Japanese securities reporting requirements

##### **\_scrape\_hkex\_filings(symbol: str) \- Hong Kong Market**

* Handles Hong Kong stock codes  
* Integrates with HKExNews system  
* Processes Hong Kong listing rule documents

#### **4\. \_get\_company\_cik(symbol: str)**

**Purpose**: Resolves US stock symbols to SEC Central Index Keys.

**Method**:

* Downloads SEC's official company tickers JSON file  
* Performs symbol-to-CIK mapping  
* Returns padded 10-digit CIK identifier

#### **5\. Company Information Methods**

**Purpose**: Provide fallback company identification for international exchanges.

**Implementation**: Maintains hardcoded dictionaries for major companies:

* \_get\_korean\_company\_info(): Samsung, SK Hynix, NAVER  
* \_get\_japanese\_company\_info(): Sony, SoftBank, Toyota  
* \_get\_hk\_company\_info(): Tencent, China Mobile, AIA

#### **6\. get\_filing\_content(filing\_url: str)**

**Purpose**: Extracts text content from regulatory documents.

**Process**:

1. **Document Retrieval**: Downloads filing document  
2. **HTML Parsing**: Uses BeautifulSoup for content extraction  
3. **Content Cleaning**: Removes scripts, styles, and formatting  
4. **Text Optimization**: Limits content to 5000 characters for processing efficiency

#### **7\. batch\_collect\_filings(symbols: List\[str\])**

**Purpose**: Orchestrates bulk filing collection with rate limiting.

**Features**:

* **Rate Limiting**: 1-second delays between requests  
* **Error Isolation**: Individual symbol failures don't affect batch  
* **Progress Tracking**: Comprehensive logging for monitoring

**News Scraper Overview**  
 The news\_scraper.py script implements a robust financial news scraping system using the NewsSourceScraper class. It aggregates, cleans, deduplicates, and analyzes financial news from multiple RSS feeds with advanced error handling and tagging.

---

**Key Features**

1. **Multi-Source RSS Scraping**

   * Sources: MarketWatch, CNBC, Google Finance, Bloomberg, Financial Times

   * Features: Source toggling, fallback rotation, rate limiting

2. **Error Handling**

   * Methods: Direct feedparser, requests with custom headers

   * Handles: HTTP errors, timeouts, malformed feeds

   * Fallback: Synthetic news generation when all sources fail

3. **Article Quality Assessment**

   * Scoring: 0–1 scale based on title/summary length, numeric content, and source

   * Filters: Title/content length, spam detection, link validation

4. **Deduplication**

   * Hash-based article ID

   * Cross-source duplicate removal

   * Session-persistent tracking

5. **Content Processing**

   * Text cleaning: Tag removal, decoding, normalization

   * Summary optimization: Sentence boundary truncation, 400-character cap

6. **Financial Tag Extraction**

   * Tags: Indices, companies, regions, events, indicators, sectors

   * Features: Composite tags, sentiment detection, sector classification

7. **Performance Monitoring**

   * Metrics: Source success rates, processing time, tag frequency

   * Logging: INFO, WARNING, ERROR, DEBUG levels

   ---

**Core Public Methods**

* scrape\_rss\_feeds(hours\_back=24, max\_articles\_per\_source=15)  
   Main scraping function returning a list of cleaned, tagged articles.

* test\_feeds()  
   Diagnostic tool to verify RSS feed health, latency, and content format.

  ---

**Internal Processing Methods**

* \_scrape\_single\_source\_enhanced() – Resilient scraping per source

* \_process\_feed\_entries\_enhanced() – Filtering, cleaning, validating entries

* \_extract\_finance\_tags\_enhanced() – Extracts financial tags and sentiment

* \_deduplicate\_articles() – Removes duplicate entries

  ---

**Configuration & Customization**

* **Sources:** Configurable via dictionary with per-source options

* **Rate Limiting:** Progressive backoff, respectful delays

* **Quality Thresholds:** Customizable quality, length, and tag rules

* **Fallbacks:** Synthetic article generation on complete failure

  ---

**Use Cases**

* Real-time market news aggregation

* Financial sentiment analysis

* Research article collection

* Topic/company-specific news monitoring

**Technical Considerations**

### **1\. Rate Limiting and Ethics**

Both modules implement responsible scraping practices with delays and request limiting to avoid overwhelming target servers.

### **2\. Error Handling**

Comprehensive exception handling ensures system resilience and provides detailed logging for debugging.

### **3\. Data Quality**

Multiple validation layers and fallback sources help ensure data accuracy and completeness.

### **4\. Maintainability**

Clear code structure, comprehensive logging, and modular design facilitate system maintenance and updates.

[image1]: <data:image/png;base64,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>
