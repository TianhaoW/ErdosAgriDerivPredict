# **U.S. Agricultural Futures Market Price Prediction**  

## **Overview**  

### **Futures Contracts**  
- **Definition**: A **futures contract** is a legally binding agreement to buy or sell an asset at a predetermined price (**futures price**) on a specific date in the future (**expiry date**).  

- **Market Participants**:  
  - **Commodity Producers** (e.g., farmers) use futures contracts to hedge against potential losses if prices decline.  
  - **Commodity Users** (e.g., food manufacturers) buy futures contracts to protect against potential price increases.  
  - **Speculators** and **arbitrageurs** trade futures to profit from price fluctuations.  

- **Trading Conventions**:  
  - Futures prices are determined through **bidding and asking** on exchanges, where traders compete to buy or sell contracts.  
  - All transactions incur **exchange fees** and require an initial **margin deposit** to ensure contract fulfillment.  
  - The **exchange** acts as an intermediary, ensuring both parties meet their contractual obligations.  
  - Most actively traded futures contracts are **standardized** by the exchange to ensure liquidity and uniformity.  

### **Agricultural Futures Markets**  
Our focus is on **U.S. agricultural futures markets**, primarily traded on the **Chicago Mercantile Exchange (CME)**, the largest and primary agricultural commodities exchange in the U.S.  

In agricultural futures markets, most contracts are **physically settled** rather than **cash settled**. This means:  
- The seller (writer) of the contract stores the commodity at a **CME-approved warehouse** before the expiry date.  
- On the expiry date, the contract holder receives a **warehouse receipt**, transferring ownership of the underlying commodity.  
- The contract holder becomes responsible for **storage costs** and **delivery expenses** from that point onward.  


We will analyze the following agricultural commodities. Their production cycles, primary growing regions, global competitors, and futures contract specifications will be detailed in a separate document.
- **Wheat**  
- **Soybeans**  
- **Sugar**  
- **Corn**  
- **Sunflower**

---

## **Project Goal**  
This is a group project for the [Erdos Institute](https://www.erdosinstitute.org/) 
2025 Spring data science bootcamp. Our team members are 
[Anshuman Bhardwaj](https://github.com/AnshumanGH91), 
[John Yin](https://github.com/johng23),
[Paul Rapoport](https://github.com/Lorxus), 
[Sam Auyeung](https://github.com/sunscorched), 
[Tianhao Wang](https://github.com/TianhaoW).

Due to data limitations, we focus on the **daily closing price** of the **nearest-to-expiry** futures contracts. Our objective is to analyze market trends and develop predictive models for agricultural futures prices.  

### **Project Stages**  
✅ **Stage 1:** Collect market data, perform EDA, and build a baseline predictive model using market data alone. *(Completed 🎯)*  
🔄 **Stage 2:** Incorporate **weather data** for improved forecasting.  
🚀 **Stage 3:** Use **satellite imagery** (via CNNs or other ML models) to assess crop growth conditions. This may involve classification tasks or using CNNs for feature extraction.  
📊 **Stage 4:** Analyze **news reports** and **USDA monthly reports** using NLP techniques such as **news classification** and **sentiment analysis**.  

### **Data Sources**  
We collect data from the following sources:  
1. **[yfinance API](https://pypi.org/project/yfinance/)** – Provides futures market data, including price and volume.  
2. **[USDA/NASS QuickStats API](https://quickstats.nass.usda.gov/)** – Agricultural production statistics and reports.  
3. **[Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/)** – Macroeconomic indicators such as interest rates and inflation.

---

### See the contribution_guide.md for the file structure of this repo.


 
