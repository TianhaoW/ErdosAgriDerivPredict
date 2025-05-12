# **U.S. Agricultural Futures Market Price Prediction**  

## **Overview**

### **What Are Futures Contracts?**  
A **futures contract** is a standardized agreement to buy or sell an asset at a predetermined price on a specific future date. These contracts are commonly used for hedging or speculation and are traded on regulated exchanges.

- **Key Participants**:  
  - **Producers** hedge against falling prices.  
  - **Buyers** (e.g., manufacturers) hedge against price increases.  
  - **Speculators** profit from market movements.

- **Trading Essentials**:  
  Contracts are traded via **bids and asks**, require **margin deposits**, and involve **exchange fees**. Most are **standardized** for liquidity, with the **exchange** acting as intermediary.

---

### **Focus: U.S. Agricultural Futures**  
Our project focuses on **agricultural futures** traded on the **Chicago Mercantile Exchange (CME)**. These contracts are typically **physically settled**, meaning:

- Sellers deliver a **warehouse receipt** for the commodity at a CME-approved facility.  
- Buyers assume **ownership**, along with **storage and delivery costs**.

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
- ✅ **Stage 1: Market Baseline**  
  Collected historical market data, performed exploratory data analysis (EDA), and built a baseline model using price and volume data alone.  
- ✅ **Stage 2: External Feature Integration**  
  Integrated **weather data** and **USDA agricultural reports** to enhance model performance.  
- 🚧 **Stage 3: Trading Strategy & Backtesting**  
  In progress: Developing trading strategies based on model forecasts and conducting **backtesting** to evaluate performance.

### **Data Sources**

- 📈 **[YFinance API](https://pypi.org/project/yfinance/)** – Daily market data for futures contracts (price, volume).  
- 🌾 **[USDA/NASS QuickStats API](https://quickstats.nass.usda.gov/)** – Agricultural production and crop condition reports.  
- ☁️ **[NOAA Weather API](https://www.weather.gov/documentation/services-web-api)**, **[ACIS API](https://www.rcc-acis.org/docs_webservices.html)** – Historical and real-time weather and climate data.  
- 💵 **[FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/)** – Macroeconomic indicators such as interest rates and inflation metrics.

---

## Contribution Guide
1. Set up your API key in the ~/config/config.toml
2. Download the cropland layer file from [USDA_cropland](https://www.nass.usda.gov/Research_and_Science/Cropland/Release/). Save it in `~/dataset/raw/`
3. Run `~/src/data/update_data.py` to obtain and auto update the dataset locally.

## File Structure

```bash
ErdosAgriDerivPredict/
├── README.md                # Overview of the project, setup instructions, and usage examples.
├── LICENSE                  # Project license.
├── .gitignore               # Files and directories to ignore (e.g., data, environment files).
├── environment.yml          # Conda environment file or requirements.txt for pip (depending on your setup).
├──
├── dataset/                 # The folder for datasets used by the model
├── notebooks/               # Jupyter notebooks for exploratory analysis, visualizations, or experiments.
├── config/                  # The config.toml file, which controls global setting for the model
├── tests/                   # Unit tests and integration tests.
├── deliverables/            # Reports required by the Erdos
├── src/
│   ├── data/                # Scripts for data ingestion, cleaning, and processing.
│   ├── models/              # Code for building, training, and evaluating models.
│   ├── utils/               # Utility functions (e.g., logging, configuration handling).
│   ├── constants/           # The static constants like cropValue code, FIPS code and etc.
├──
├── Gemfile                  # The gems that supports the Github Pages
├── _config.yml              # The configuration for the Github Pages
├── index.md                 # The index page of this project
├── _data/                   # .yml file used by the Github Pages
├── _includes/               # .html file for website components
├── _layouts/                # .html file for website layouts
├── assets/                  # .js and .css file for website scripts and stylesheet
├── imgs/                    # the images used for the website
├── pages/                   # .md files for the content of the website
```



 
