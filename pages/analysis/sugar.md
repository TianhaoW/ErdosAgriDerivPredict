---
layout: page
title: Sugar Futures Market
toc: true
toc_title: Table of Contents
---

## Sugar Production and Its Diverse Applications

### Major Sugar-Producing Regions in the U.S.

Sugar in the United States is primarily derived from two sources: **sugarcane** and **sugar beets**.

| Source      | Top States                                                                                                    | Notes                                                                                                    |
|-------------|---------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| Sugarcane   | Florida, Louisiana, Texas, Hawaii (historical)                                                                | Florida leads (mainly Palm Beach Co.); Louisiana is a strong producer. Hawaii ceased production in 2016. | 
| Sugar Beets | Minnesota, North Dakota, Michigan, Idaho, Nebraska, Montana, California, Wyoming, Oregon, Washington, Colorado | Beet production is centered in the Upper Midwest, Great Plains, and Pacific Northwest.                   | 

- Sugarcane is grown in the Southern U.S., with Florida and Louisiana as top producers.
- Sugar beets are cultivated in temperate regions, especially the Upper Midwest and West. 
- Together, these two sources form the basis of the U.S. sugar supply chain.
- Hawaii, once a major sugarcane producer, ended commercial operations in 2016.

{% include notification.html 
status="is-info is-light"
icon="false"
message="
### Historical Context
The history of sugar production in the U.S. is intertwined with significant socio-economic developments:

- **Slavery in Louisiana**: Sugar plantations in Louisiana heavily relied on enslaved labor, contributing to the entrenchment of slavery in the region.
- **Annexation of Hawaii**: Economic interests tied to sugar production played a role in the U.S. annexation of Hawaii in the late 19th century.
" %}

### Non-Food Applications of Sugar
Beyond its role in food and beverages, sugar serves various industrial and commercial purposes:

- **Pharmaceuticals**: Sugar enhances the taste of medications and acts as a binder in pill formulations.
- **Wastewater Treatment**: Liquid sugar is utilized to facilitate the denitrification process in wastewater management.
- **Agriculture**: Sugar solutions are applied in vineyards to stimulate microbial activity, promoting soil health.
- **Cleaning**: Sugar can be an effective agent for removing certain stains from clothing and appliances.
- **Bioplastics**: Sugar serves as a base material in producing bioplastics, offering a renewable alternative to conventional plastics. 

## Plots of Market Trends

{% include media-block.html 
    image="/imgs/sugar/price1.png"
    position="left"
    text="

### Single Timeline of Closing Price

There are low points such as the period leading up to 2008, 2016, and 2020 which align with certain global recessions.


"%}

{% include media-block.html 
    image="/imgs/sugar/price2.png"
    position="right"
    text="

### By-Year Closing Price 

There is a price drop around the 125th day which may correspond to the expiry of Sugar No. 16 contracts in April.

"%}

{% include media-block.html 
    image="/imgs/sugar/volatility.png"
    position="left"
    text="

### 30-Day Rolling Volatility of Closing Price


"%}

{% include media-block.html 
    image="/imgs/sugar/MA.png"
    position="right"
    text="
### Moving Averages of Closing Price

"%}

{% include media-block.html 
    image="/imgs/sugar/vol1.png"
    position="left"
    text="
### Single Timeline of Volume

"%}

{% include media-block.html 
    image="/imgs/sugar/vol2.png"
    position="right"
    text="

### By-Year Volume

Delivery months are January, March, May, July, September, November and the last trading days are on the 8th day of the prior month. There appear to be spikes in volume around these times.

"%}

{% include media-block.html 
    image="/imgs/sugar/vol3.png"
    position="left"
    text="

### Volume based on Days Till Expiry

Low volume near day of expiry until on the last trading day, there is a spike as traders clear their positions.

"%}

{% include media-block.html 
    image="/imgs/sugar/log_return.png"
    position="right"
    text="

### Sugar Futures Log Returns

Empirically, the log returns seem normally distributed with mean 0.

"%}

{% include media-block.html 
    image="/imgs/sugar/corn.png"
    position="left"
    text="
    
### Comparison of Sugar and Corn Futures (Standardized)

Since corn syrup is a widely used sweetener, we plot the two together. Very roughly, it does seem like the shape of the curve for sugar is ahead of the one for corn. E.g. one of the minima for sugar is near the end of 2023 while a similar minimum for corn occurs in February of 2024.

"%}

{% include media-block.html 
    image="/imgs/sugar/pacf.png"
    position="right"
    text="

### Partial Autocorrelation of 7-Day Log Returns

As a secondary point of interest, theory of geometric Brownian motion suggests there be spikes in the partial autocorrelation function at days $x \equiv 1 \pmod{7}$, which indeed is observed here.

"%}
