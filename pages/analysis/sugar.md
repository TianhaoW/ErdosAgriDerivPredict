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

"%}

{% include media-block.html 
    image="/imgs/sugar/price2.png"
    position="left"
    text="

### By-Year Closing Price 

"%}

{% include media-block.html 
    image="/imgs/sugar/volatility.png"
    position="left"
    text="
### 30-Day Rolling Volatility of Closing Price

"%}

{% include media-block.html 
    image="/imgs/sugar/MA.png"
    position="left"
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
    position="left"
    text="

### By-Year Volume

"%}

{% include media-block.html 
    image="/imgs/sugar/vol3.png"
    position="left"
    text="

### Volume based on Days Till Expiry

"%}

{% include media-block.html 
    image="/imgs/sugar/log_return.png"
    position="left"
    text="
### Sugar Futures Log Returns

"%}

{% include media-block.html 
    image="/imgs/sugar/corn.png"
    position="left"
    text="
### Comparison of Sugar and Corn Futures (Standardized)

"%}

Use the following to create media block. 

image = "address to your plot". Please use 1000*560 size picture (plt.figure(10,5.6)) and use plt.save_fig("../imgs/sugar/name") to save your plot

position="left" or "right" depending on if you want the plot on the left or right
text = "the message in markdown format"

{% include media-block.html 
    image="/imgs/sugar/price.png"
    position="left"
    text="
### Sugar Price Trends (Historical Perspective)
Your text here. Here are some examples. 
- **Bold Text**, 
- inline math $\int_a^b$
- [external link example](https://github.com)

Block math 

$$\lim_{x\rightarrow\infty} f_n(x)$$

"%}
