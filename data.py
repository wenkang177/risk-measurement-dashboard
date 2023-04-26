pip install streamlit_echarts
import streamlit as st # pip install streamlit
from streamlit_echarts import st_echarts
import numpy as np
import yfinance as yf
import pandas as pd
#Statistical calculation
from scipy.stats import norm   
import matplotlib.pyplot as plt
import scipy.stats as stats


stocks_data = {
    'AAPL': 'Apple Inc',
    'MSFT': 'Mircosoft',
    'AMZN':"Amazon",
    'NVDA':"NVIDIA",
    'GOOGL':"Google",
    'BRK-B':"Berkshire Hatway",
    "META":"Facebook"
  
}

stock_list = pd.DataFrame(list(stocks_data.items()), columns=['Stock Code', 'Company Name'])
stock_name = stock_list['Company Name'].tolist()
stock_code = stock_list['Stock Code'].tolist()
cofidence_interval = [0.9,0.95,0.99]  # less than 1

df = yf.download(stock_code, start="2014-01-02", end="2023-04-21",interval ="1d")
market_df = yf.download("^GSPC","2014-1-02","2023-4-21",interval ="1d")

df_m = yf.download(stock_code, start="2014-01-02", end="2023-04-21",interval ="1mo")
market_df_m = yf.download("^GSPC","2014-1-02","2023-4-21",interval ="1mo")


returns_m = market_df["Adj Close"].pct_change().dropna() # market return
returns_s = df["Adj Close"].pct_change().dropna() # stock return
returns_s_m = df_m["Adj Close"].pct_change().dropna()

weights = np.array([0.31,0.27,0.1,0.095,0.084,0.081,0.06])
return_p = (weights*returns_s).sum(axis=1)   #portfolio return
return_p_m = (weights*returns_s_m).sum(axis=1)

date_list = df.index.tolist()
date_list = [date.strftime('%Y-%m-%d') for date in date_list] #dare
month_list = df_m.index.tolist()
month_list = [date.strftime('%Y-%m') for date in month_list] # month

colors = ['#FF3333', '#FE9D30','#EC65FC','#F9F92D','#70FA48','#48D7FA','#4863FA']

test_weights = np.sum(weights) #test weight =1
print(test_weights)
#print(returns_m)


###########################################################################################
stocks_t2= {
    'AAPL': 'Apple Inc',
    'MSFT': 'Mircosoft',
    'AMZN':"Amazon",
    'XOM':"Exxom"
  
}

stock_list_t2 = pd.DataFrame(list(stocks_t2.items()), columns=['Stock Code', 'Company Name'])
stock_name_2 = stock_list_t2['Company Name'].tolist()
stock_code_2 = stock_list_t2['Stock Code'].tolist()


df_2 = yf.download(stock_code_2, start="2014-1-02", end="2023-04-21",interval ="1d")
df_m_2= yf.download(stock_code_2, start="2014-1-02", end="2023-04-21",interval ="1mo")


returns_s_2 = df_2["Adj Close"].pct_change().dropna() # stock return
returns_s_m_2 = df_m_2["Adj Close"].pct_change().dropna()

weights_2 = np.array([0.4,0.3,0.1,0.2])
return_p_2 = (weights_2*returns_s_2).sum(axis=1)   #portfolio return
return_p_m_2 = (weights_2*returns_s_m_2).sum(axis=1)

colors_2 = ['#FF3333', '#FE9D30','#70FA48','#48D7FA']


###############################   Other Risk Measurement #########################
Expected_rm = returns_m.mean()
rf = 0.03
rp = np.sum(weights*returns_s).mean()
# Calculate beta of each stock and portfolio
beta_2 = np.zeros(len(stocks_t2))
for i in range(len(stocks_t2)):
    covariance = np.cov(returns_s_2[stock_code_2[i]], returns_m)[0][1]
    variance = np.var(returns_m)
    # Calculate beta
    beta_2[i] = round(covariance / variance,2)
#portfolio Beta
betas_2 = beta_2.tolist()

portfolio_beta_2 = np.sum(weights_2 * betas_2)

# Define betas and stock_names

#portfolio standard deviation
cov = np.cov(returns_s_2, rowvar=False)
portfolio_std_dev_2 = np.sqrt(np.dot(weights_2.T, np.dot(cov, weights_2)))
##### Stanard deviation of each stock
std_dev_2 = returns_s_2.std().tolist()
 

expected_p = np.sum(returns_s_m*weights).mean()
sharpe_ratio_2 = (expected_p - rf) / portfolio_std_dev_2 #Sharpe ratio

#Treynor Ratio
t_r_2 = (rp-rf)/portfolio_beta_2


###############################   Other Risk Measurement of trader 2 #########################
Expected_rm = returns_m.mean()
rf = 0.03
rp = np.sum(weights*returns_s).mean()
# Calculate beta of each stock and portfolio
beta = np.zeros(len(stocks_data))
for i in range(len(stocks_data)):
    covariance = np.cov(returns_s[stock_code[i]], returns_m)[0][1]
    variance = np.var(returns_m)
    # Calculate beta
    beta[i] = round(covariance / variance,2)
#portfolio Beta
betas = beta.tolist()

portfolio_beta = np.sum(weights * betas)

# Define betas and stock_names

#portfolio standard deviation
covariance = np.cov(returns_s, rowvar=False)
portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
##### Stanard deviation of each stock
std_dev = returns_s.std().tolist()
 

expected_p = np.sum(returns_s*weights).mean()
sharpe_ratio = (expected_p - rf) / portfolio_std_dev #Sharpe ratio

#Treynor Ratio
t_r = (rp-rf)/portfolio_beta

######################### Function Parametric Method of trader 1 ######################### 
def para_day():
     mean_1 = round(return_p.rolling(window=250).mean(),4)
     mean_1 = np.nan_to_num(mean_1, nan=0)
     z_90 = norm.ppf(0.9) #z-score
     z_95 = norm.ppf(0.95)
     z_99 = norm.ppf(0.99)
     portfolio_std = return_p.rolling(window=250).std()
     portfolio_std = np.nan_to_num(portfolio_std, nan=0)
     # VaR
     VaR_90 = (mean_1- z_90*portfolio_std).astype(float).round(4)
     # CVaR
     Expected_Loss_90 = (((1 - z_90) / z_90) * norm.cdf(z_90) * portfolio_std -z_90*mean_1)
     CVaR_90 = (VaR_90+Expected_Loss_90).round(4)
     VaR_95 = (mean_1- z_95*portfolio_std).astype(float).round(4)
     # CVaR
     Expected_Loss_95 = (((1 - z_95) / z_95) * norm.cdf(z_95) * portfolio_std -z_95*mean_1)
     CVaR_95 = (VaR_95+Expected_Loss_95).round(4)
     VaR_99 = (mean_1- z_99*portfolio_std).astype(float).round(4)
     # CVaR
     Expected_Loss_99 = (((1 - z_99) / z_99) * norm.cdf(z_99) * portfolio_std -z_99*mean_1)
     CVaR_99 = (VaR_99+Expected_Loss_99).round(4)
     
     return VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99
 
def para_month():
    mean_m = round(return_p_m.rolling(window=20).mean(),4)
    mean_m = np.nan_to_num(mean_m, nan=0)
    z_90 = norm.ppf(0.9) #z-score
    z_95 = norm.ppf(0.95)
    z_99 = norm.ppf(0.99) #z-score
    portfolio_std_m = return_p_m.rolling(window=10).std()
    portfolio_std_m = np.nan_to_num(portfolio_std_m, nan=0)
    # VaR
    VaR_90 = (mean_m- z_90*portfolio_std_m).astype(float).round(4)
    # CVaR
    Expected_Loss_90 = (((1 - z_90) / z_90) * norm.cdf(z_90) * portfolio_std_m -z_90*mean_m)
    CVaR_90 = (VaR_90+Expected_Loss_90).round(4)
    VaR_95 = (mean_m- z_95*portfolio_std_m).astype(float).round(4)
    # CVaR
    Expected_Loss_95 = (((1 - z_95) / z_95) * norm.cdf(z_95) * portfolio_std_m -z_95*mean_m)
    CVaR_95 = (VaR_95+Expected_Loss_95).round(4)
    VaR_99 = (mean_m- z_99*portfolio_std_m).astype(float).round(4)
    # CVaR
    Expected_Loss_99 = (((1 - z_99) / z_99) * norm.cdf(z_99) * portfolio_std_m -z_99*mean_m)
    CVaR_99 = (VaR_99+Expected_Loss_99).round(4)
    
    return VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99

######################### Function Parametric Method of trader 2 #########################
def para_day_2():
    mean_1 = round(return_p_2.rolling(window=250).mean(),4)
    mean_1 = np.nan_to_num(mean_1, nan=0)
    z_90 = norm.ppf(0.9) #z-score
    z_95 = norm.ppf(0.95)
    z_99 = norm.ppf(0.99)
    portfolio_std = return_p_2.rolling(window=250).std()
    portfolio_std = np.nan_to_num(portfolio_std, nan=0)
     # VaR
    VaR_90 = (mean_1- z_90*portfolio_std).astype(float).round(4)
     # CVaR
    Expected_Loss_90 = (((1 - z_90) / z_90) * norm.cdf(z_90) * portfolio_std -z_90*mean_1)
    CVaR_90 = (VaR_90+Expected_Loss_90).round(4)
    VaR_95 = (mean_1- z_95*portfolio_std).astype(float).round(4)
     # CVaR
    Expected_Loss_95 = (((1 - z_95) / z_95) * norm.cdf(z_95) * portfolio_std -z_95*mean_1)
    CVaR_95 = (VaR_95+Expected_Loss_95).round(4)
    VaR_99 = (mean_1- z_99*portfolio_std).astype(float).round(4)
     # CVaR
    Expected_Loss_99 = (((1 - z_99) / z_99) * norm.cdf(z_99) * portfolio_std -z_99*mean_1)
    CVaR_99 = (VaR_99+Expected_Loss_99).round(4)
     
    return VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99 
 
def para_month_2():
    mean_1 = round(return_p_m_2.rolling(window=20).mean(),4)
    mean_1 = np.nan_to_num(mean_1, nan=0)
    z_90 = norm.ppf(0.9) #z-score
    z_95 = norm.ppf(0.95)
    z_99 = norm.ppf(0.99)
    portfolio_std = return_p_m_2.rolling(window=10).std()
    portfolio_std = np.nan_to_num(portfolio_std, nan=0)
     # VaR
    VaR_90 = (mean_1- z_90*portfolio_std).astype(float).round(4)
     # CVaR
    Expected_Loss_90 = (((1 - z_90) / z_90) * norm.cdf(z_90) * portfolio_std -z_90*mean_1)
    CVaR_90 = (VaR_90+Expected_Loss_90).round(4)
    VaR_95 = (mean_1- z_95*portfolio_std).astype(float).round(4)
     # CVaR
    Expected_Loss_95 = (((1 - z_95) / z_95) * norm.cdf(z_95) * portfolio_std -z_95*mean_1)
    CVaR_95 = (VaR_95+Expected_Loss_95).round(4)
    VaR_99 = (mean_1- z_99*portfolio_std).astype(float).round(4)
     # CVaR
    Expected_Loss_99 = (((1 - z_99) / z_99) * norm.cdf(z_99) * portfolio_std -z_99*mean_1)
    CVaR_99 = (VaR_99+Expected_Loss_99).round(4)
     
    return VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99 

def marketvar():
    mean_1 = round(returns_m.rolling(window=250).mean())
    mean_1 = np.nan_to_num(mean_1, nan=0)
    z_90 = norm.ppf(0.9) #z-score
    z_95 = norm.ppf(0.95)
    z_99 = norm.ppf(0.99)
    portfolio_std = returns_m.rolling(window=250).std()
    portfolio_std = np.nan_to_num(portfolio_std, nan=0)
     # VaR
    VaR_90 = (mean_1- z_90*portfolio_std).astype(float)
     # CVaR
    Expected_Loss_90 = (((1 - z_90) / z_90) * norm.cdf(z_90) * portfolio_std -z_90*mean_1)
    CVaR_90 = (VaR_90+Expected_Loss_90)
    VaR_95 = (mean_1- z_95*portfolio_std).astype(float)
     # CVaR
    Expected_Loss_95 = (((1 - z_95) / z_95) * norm.cdf(z_95) * portfolio_std -z_95*mean_1)
    CVaR_95 = (VaR_95+Expected_Loss_95)
    VaR_99 = (mean_1- z_99*portfolio_std).astype(float)
     # CVaR
    Expected_Loss_99 = (((1 - z_99) / z_99) * norm.cdf(z_99) * portfolio_std -z_99*mean_1)
    CVaR_99 = (VaR_99+Expected_Loss_99)
    mean_var90 =f"{round(np.trim_zeros(VaR_90).mean(),5)*100}%"
    mean_var95 =f"{round(np.trim_zeros(VaR_95).mean(),5)*100}%"
    mean_var99 =f"{round(np.trim_zeros(VaR_99).mean(),5)*100}%"
    mean_cvar90 =f"{round(np.trim_zeros(CVaR_90).mean(),5)*100}%"
    mean_cvar95 =f"{round(np.trim_zeros(CVaR_95).mean(),5)*100}%"
    mean_cvar95 =f"{round(np.trim_zeros(CVaR_99).mean(),5)*100}%"

    return mean_var90,mean_cvar90,mean_var95,mean_cvar95,mean_var99,mean_cvar95

  

##########################    Data visualization    ######################### 
st.title("Porfolio risk monitor dashboard")
st.markdown("This dashboard will show the risk measurement of trader.The risk measurement includes Beta,Standard Deviation,Sharpe ratio, Treynor Ratio")
st.caption('Risk Manager: Ng Wen Kang')
tab1, tab2 = st.tabs(["Trader 1 ", "Trader 2"])

with tab1:
   st.header("Trader 1")
   data_w = [{"value": weights, "name": stock_name, "itemStyle": {"color": colors[i]}} for i, (stock_name, weights) in enumerate(zip(stock_name, weights))]
   options = {
    "title": {"text": "Weight of stock", "subtext": "From Yfinance", "left": "center","textStyle": {"fontSize": 24}},
    "tooltip": {"trigger": "item"},
    "series": [
        {
            "name": "stock",
            "type": "pie",
            "radius": "50%",
            "data": [
                {"value": weights[i], "name": stock_name[i],"itemStyle": {"color": colors[i]}} for i in range(len(weights),)
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                }
            },
        }
    ],
}
   st_echarts(
    options=options, height="700px")
   
########################### Market Var & CVaR ****************************************
tab1.subheader("Market Expected VaR & CVaR" )


with tab1:
    st.write(pd.DataFrame({
    'Confidence Level': ["90%", "95%", "99%"],
    'Value at risk': [marketvar()[0], marketvar()[2], marketvar()[4]],
    'Conditional Value at risk': [marketvar()[1], marketvar()[3], marketvar()[5]],
}))

########################### Var & CVaR ****************************************
with tab1:
    tab1.subheader("Expected VaR & CVaR" )
    st.caption('This is using Parametric Method Mehtod to calculate .')

    tab1.var1, tab1.var2, tab1.var3 = st.columns(3)                         
    tab1.var1.metric("VaR of 90%", f"{round(np.mean(para_day()[0]),5)*100}%",round(np.mean(abs(para_day()[0]))+np.mean(abs(para_day_2()[0])),5),delta_color="inverse")
    tab1.var2.metric("VaR of 95%", f"{round(np.mean(para_day()[2]),5)*100}%",round(np.mean(abs(para_day()[2]))+np.mean(abs(para_day_2()[2])),5),delta_color="inverse")
    tab1.var3.metric("VaR of 99%", f"{round(np.mean(para_day()[4]),5)*100}%",round(np.mean(abs(para_day()[4]))+np.mean(abs(para_day_2()[4])),5),delta_color="inverse")
    
    tab1.cvar1, tab1.cvar2, tab1.cvar3 = st.columns(3)
    tab1.cvar1.metric("CVaR of 90%", f"{round(np.mean(para_day()[1]),5)*100}%",round(np.mean(abs(para_day()[1]))+np.mean(abs(para_day_2()[1])),5),delta_color="inverse")
    tab1.cvar2.metric("CVaR of 95%", f"{round(np.mean(para_day()[3]),5)*100}%",round(np.mean(abs(para_day()[3]))+np.mean(abs(para_day_2()[3])),5),delta_color="inverse")
    tab1.cvar3.metric("CVaR of 99%", f"{round(np.mean(para_day()[5]),5)*100}%",round(np.mean(abs(para_day()[5]))+np.mean(abs(para_day_2()[5])),5),delta_color="inverse")

###################         Other risk measurement     ########################
tab1.subheader("Other risk meaurement")
with tab1:
    tab1.col1, tab1.col2, tab1.col3,tab1.col4 = st.columns(4)
    tab1.col1.metric("Portfolio Beta", f"{round(portfolio_beta,2)}", round(portfolio_beta-portfolio_beta_2,3))
    tab1.col2.metric("Standard Deviation", f"{round(portfolio_std_dev,4)}",round(portfolio_std_dev-portfolio_std_dev_2,3))
    tab1.col3.metric("Sharpe ratio", f"{round(sharpe_ratio,2)}",round(sharpe_ratio-sharpe_ratio_2,3) )
    tab1.col4.metric("Treynor Ratio",f"{round(t_r,2)}",round(t_r-t_r_2,3))
    


##############   ########################  ######################   ############

data = [{"value": beta, "name": stock_name, "itemStyle": {"color": colors[i]}} for i, (stock_name, beta) in enumerate(zip(stock_name, betas))]
data_std_dev = [{"value": std_dev, "name": stock_name, "itemStyle": {"color": colors[i]}} for i, (stock_name, std_dev) in enumerate(zip(stock_name, std_dev))]

##################     beta & standard Deviation chart       ##############
with tab1:
        beta = {
            "title": {"text": "Beta of each stock"},
            "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: {c}"},
            "legend": None,
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {"type": "value", "name": "Beta", "nameLocation": "middle"},
            "yAxis": {"type": "category", "data": list(stock_name)},
            "series": [{"data": data, "type": "bar"}],
        }
        st_echarts(options=beta, height="300px")
    
        std = {
            "title": {"text": "Standard Deviation of each stock"},
            "xAxis": {
                "type": "category",
                "data": list(stock_name),
                "axisLabel": {"fontSize": 10, "interval": 0, "lineHeight": 14},},
            "yAxis": {"type": "value"},
            "series": [{"data": data_std_dev , "type": "bar"}],
            "tooltip": {
                "trigger": "axis",
                "formatter": "{b}: {c}",},}
        st_echarts(options=std, height="300px", width="800px")

################## ################## ################## ################## #####
       

with tab1:
    t_1,t_2,t_3 = st.tabs(["📈 Historical Method VaR & CVaR", "💲Parametric Method VaR & CVaR", "💰Monte Carlo Method VaR & CVaR"])
    st.caption('********* The value in first new date/month is horinzotal,becasue of using as histrocal data .')
#########################     Historical Method       ###########################          
with t_1: 
    confidence_level, period = st.columns(2)
    with t_1: 
        confidence_level, period = st.columns(2)
    with confidence_level:
      confidence_level = st.radio("Confidence Level", ("90%","95%","99%"), key='confidence_level_t1')
      
    with period:
          period = st.selectbox('Period', ('Day', 'Month'),key='period_t1')


    st.subheader("Historical Method VaR & CVaR")
    
    if confidence_level == "90%" and period == 'Day':
        VaR_90_Day = round(return_p.rolling(window=250).quantile(0.10),4)
        VaR_90_Day = np.array(np.nan_to_num(VaR_90_Day, nan=0))
        #av = (VaR_90_Day[VaR_90_Day<0]).tolist()
        #CVaR = [round(return_p[return_p<a].mean(),4) for a in av] # Calculate CVaR_90_Day
        av = (VaR_90_Day).tolist()
        CVaR = [round(return_p[return_p<a].mean(),4) for a in av] # Calculate CVaR_90_Day
        
       
        options = {
            "title": {"text": "VaR_90"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": date_list,},
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [{
                "name": "VaR",
                "type": "line",
                "data":  VaR_90_Day.tolist()
                },
                {"name": "CVaR",
                 "type": "line",
                 "data": CVaR
                    }
                ]}
       
        st_echarts(options=options, height="400px") 
        
        plt.hist(VaR_90_Day, bins=50)
        plt.hist(CVaR, bins=50)
        plt.xlabel('Historical var vs Cvar of trader 1 in 90% Cofidence Level')
        plt.show()        
        
    if confidence_level == "95%" and period == 'Day':
        VaR_95_Day = round(return_p.rolling(window=250).quantile(0.05),4)
        VaR_95_Day = np.nan_to_num(VaR_95_Day, nan=0)
        av = (VaR_95_Day).tolist()
        CVaR_95 = [round(return_p[return_p<a].mean(),4) for a in av] # Calculate CVaR_90_Day
        
        options = {
            "title": {"text": "VaR_95"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": date_list,},
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [{
                "name": "VaR",
                "type": "line",
                "data":  VaR_95_Day.tolist()
                },
                {"name": "CVaR",
                 "type": "line",
                 "data": CVaR_95
                    }
                ]}
        st_echarts(options=options, height="400px")
        
        plt.hist(VaR_95_Day, bins=50)
        plt.hist(CVaR, bins=50)
        plt.xlabel('Historical var vs Cvar of trader 1 in 95% Cofidence Level')
        plt.show()  
        
    if confidence_level == "99%" and period == 'Day':
      VaR_99_Day = round(return_p.rolling(window=250).quantile(0.01),4)
      VaR_99_Day = np.nan_to_num(VaR_99_Day, nan=0)
      av = (VaR_99_Day).tolist()
      CVaR_99 = [round(return_p[return_p<a].mean(),4) for a in av] # Calculate CVaR
      
      options = {
          "title": {"text": "VaR_99"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
          "xAxis": {
              "type": "category",
              "boundaryGap": False,
              "data": date_list,},
              "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
              "series": [{
                  "name": "VaR",
                  "type": "line",
                  "data": VaR_99_Day.tolist()},
                  {"name": "CVaR",
                   "type": "line",
                   "data": CVaR_99}]}
      
      st_echarts(options=options, height="400px") 
    
    if confidence_level == "90%"and period == 'Month':
      VaR_90 = round(return_p_m.rolling(window=20).quantile(0.10),4)
      VaR_90 = np.nan_to_num(VaR_90, nan=0)
      av = (VaR_90).tolist()
      CVaR = [round(return_p_m[return_p_m<a].mean(),4) for a in av] # Calculate CVaR
      
      options = {
          "title": {"text": "VaR_90"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
        "type": "category",
        "boundaryGap": False,
        "data": month_list,},
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
                {
            "name": "VaR",
            "type": "line",
            "data": VaR_90.tolist()},
            {"name": "CVaR",
             "type": "line",
             "data": CVaR}]}
      st_echarts(options=options, height="400px") 
      
    if confidence_level == "95%"and period == 'Month':
          VaR_95 = round(return_p_m.rolling(window=20).quantile(0.10),4)
          VaR_95 = np.nan_to_num(VaR_95, nan=0)
          av = (VaR_95).tolist()
          CVaR = [round(return_p_m[return_p_m<a].mean(),4) for a in av] # Calculate CVaR
          
          options = {
        "title": {"text": "VaR_95"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": month_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {
                "name": "VaR",
                "type": "line",
                "data": VaR_95.tolist()}, {"name": "CVaR",
                  "type": "line",
                  "data": CVaR}]}
          st_echarts(options=options, height="400px")
          
    if confidence_level == "99%"and period == 'Month':
       VaR_99 = round(return_p_m.rolling(window=20).quantile(0.01),4)
       VaR_99 = np.nan_to_num(VaR_99, nan=0)
       av = (VaR_99).tolist()
       CVaR = [round(return_p_m[return_p_m<a].mean(),4) for a in av] # Calculate CVaR
       
       options = {
     "title": {"text": "VaR_99"},
     "tooltip": {"trigger": "axis"},
     "legend": {"data": ["VaR","CVaR"]},
     "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
     "toolbox": {"feature": {"saveAsImage": {}}},
     "xAxis": {
         "type": "category",
         "boundaryGap": False,
         "data": month_list,
     },
     "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
     "series": [
         {
             "name": "VaR_99",
             "type": "line",
             "data": VaR_99.tolist()},
             {"name": "CVaR",
               "type": "line",
               "data": CVaR}]}
                                        
                                        
       st_echarts(options=options, height="400px")
       
       
         
#########################  #########################   ###########################        
     
#########################     Parametric Method       ###########################   
with t_2: 
    option_1, option_2 = st.columns(2)
    with option_1:
      option_1= st.radio("Confidence Level", ("90%","95%","99%"), key='confidence_level_t2')

    with option_2:
        option_2 = st.selectbox('Period', ('Day', 'Month'),key='period_t2')
        
    st.subheader("Parametric Method VaR & CVaR")
        
    if option_1 == "90%" and option_2 == 'Day':
        
       
        options = {
            "title": {"text":"VaR & CVaR"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": date_list,
            },
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [
                {"name": "VaR",
                 "type": "line",
                 "data": para_day()[0].tolist()},
                {"name": "CVaR",
                 "type": "line",
                 "data": para_day()[1].tolist()},
            ],
        }
        st_echarts(options=options, height="500px")
        
        plt.hist(para_day()[0], bins=50)
        plt.hist(para_day()[1], bins=50)
        plt.xlabel('Parametric var vs Cvar of trade 1 in 90% confidence level')
        plt.show()
        
    elif option_1 == "95%" and option_2 == 'Day':
     
      options = {
          "title": {"text":"VaR & CVaR"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
          "xAxis": {
              "type": "category",
              "boundaryGap": False,
              "data": date_list,
          },
          "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
          "series": [
              {"name": "VaR",
               "type": "line",
               "data": para_day()[2].tolist()},
              {"name": "CVaR",
               "type": "line",
               "data": para_day()[3].tolist()},
          ],
      }
      st_echarts(options=options, height="500px")  
    elif option_1 == "99%" and option_2 == 'Day':
     
     
      options = {
          "title": {"text":"VaR & CVaR"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
          "xAxis": {
              "type": "category",
              "boundaryGap": False,
              "data": date_list,
          },
          "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
          "series": [
              {"name": "VaR",
               "type": "line",
               "data": para_day()[4].tolist()},
              {"name": "CVaR",
               "type": "line",
               "data": para_day()[5].tolist()},
          ],
      }
      st_echarts(options=options, height="500px")             
    if option_1 == "90%" and option_2 == 'Month':
        
        options = {
        "title": {"text":"VaR & CVaR"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": month_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {"name": "VaR",
             "type": "line",
             "data": para_month()[0].tolist()},
            {"name": "CVaR",
             "type": "line",
             "data": para_month()[1].tolist()},
        ],
    }
        st_echarts(options=options, height="500px") 
       
        
    if option_1 == "95%" and option_2 == 'Month':
       
        options = {
        "title": {"text":"VaR & CVaR"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": month_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {"name": "VaR",
             "type": "line",
             "data": para_month()[2].tolist()},
            {"name": "CVaR",
             "type": "line",
             "data": para_month()[3].tolist()},
        ],
    }
        st_echarts(options=options, height="500px") 
    if option_1 == "99%" and option_2 == 'Month':
        
      options = {
      "title": {"text":"VaR & CVaR"},
      "tooltip": {"trigger": "axis"},
      "legend": {"data": ["VaR","CVaR"]},
      "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
      "toolbox": {"feature": {"saveAsImage": {}}},
      "xAxis": {
          "type": "category",
          "boundaryGap": False,
          "data": month_list,
      },
      "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
      "series": [
          {"name": "VaR",
           "type": "line",
           "data": para_month()[4].tolist()},
          {"name": "CVaR",
           "type": "line",
           "data": para_month()[5].tolist()},
      ],
  }
      st_echarts(options=options, height="500px")    

######################### ######################### ######################### 
with t_3: 
    option_1, option_2 = st.columns(2)
    with option_1:
      option_1 = st.radio("Confidence Level", ("90%","95%","99%"), key='confidence_level_t3')

    with option_2:
        option_2 = st.selectbox('Period', ('Day', 'Month'),key='period_t3')

    st.subheader("Mento Carlo Method VaR & CVaR")  
    
    def var():
        
        mean_1 = round(return_p.rolling(window=150).mean(),4)
        mean_1 = np.nan_to_num(mean_1, nan=0)
        portfolio_std = return_p.rolling(window=150).std()
        portfolio_std = np.nan_to_num(portfolio_std, nan=0)
        
        random_num = np.random.rand(len(return_p)) # function for random z-score,random choose len(returns) times
        z = norm.ppf(random_num).round(4) #z-score
        scenario_VaR =pd.Series(mean_1-z*portfolio_std).sort_values()
        VaR_90 = []
        VaR_95 = []
        VaR_99 = []
        CVaR_90 = []
        CVaR_95 = []
        CVaR_99 = []
        n_90 = int(0.1*return_p.count())
        n_95 = int(0.05*return_p.count())
        n_99 = int(0.01*return_p.count())
        
        for i in range(len(return_p)):
           z = norm.ppf(random_num).round(4)
           scenario_VaR = np.sort(mean_1[i] - z * portfolio_std[i])
           VaR_90.append(np.percentile(scenario_VaR, 100 - 100 * 0.9))
           VaR_95.append(np.percentile(scenario_VaR, 100 - 100 * 0.95))
           VaR_99.append(np.percentile(scenario_VaR, 100 - 100 * 0.99))
           CVaR_90.append(scenario_VaR[:n_90].mean()) 
           CVaR_95.append(scenario_VaR[:n_95].mean())   
           CVaR_99.append(scenario_VaR[:n_99].mean())   

        return(VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99)
       
    if option_1 == "90%"and option_2 == 'Day':
        options = {
        "title": {"text":"VaR & CVaR"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": date_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {"name": "VaR",
             "type": "line",
             "data": var()[0]},
            {"name": "CVaR",
             "type": "line",
             "data": var()[1]},
        ],
    }
        st_echarts(options=options, height="500px")     
        
        plt.hist(var()[0], bins=50)
        plt.hist(var()[1], bins=50)
        plt.xlabel('Monte Carlo var vs Cvar of trader 1 in 90 %confidence level')
        plt.show()
        
    if option_1 == "95%"and option_2 == 'Day':
         options = {
         "title": {"text":"VaR & CVaR"},
         "tooltip": {"trigger": "axis"},
         "legend": {"data": ["VaR","CVaR"]},
         "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
         "toolbox": {"feature": {"saveAsImage": {}}},
         "xAxis": {
             "type": "category",
             "boundaryGap": False,
             "data": date_list,
         },
         "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
         "series": [
             {"name": "VaR",
              "type": "line",
              "data": var()[2]},
             {"name": "CVaR",
              "type": "line",
              "data": var()[3]},
         ],
     }
         st_echarts(options=options, height="500px")     
    if option_1 == "99%"and option_2 == 'Day':     
         options = {
         "title": {"text":"VaR & CVaR"},
         "tooltip": {"trigger": "axis"},
         "legend": {"data": ["VaR","CVaR"]},
         "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
         "toolbox": {"feature": {"saveAsImage": {}}},
         "xAxis": {
             "type": "category",
             "boundaryGap": False,
             "data": date_list,
         },
         "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
         "series": [
             {"name": "VaR",
              "type": "line",
              "data": var()[4]},
             {"name": "CVaR",
              "type": "line",
              "data": var()[5]},
         ],
     }
         st_echarts(options=options, height="500px")
    
    def mvar():
        mean = round(return_p_m.rolling(window=15).mean(),4)
        mean = np.nan_to_num(mean, nan=0)
        portfolio_std = return_p_m.rolling(window=20).std()
        portfolio_std = np.nan_to_num(portfolio_std, nan=0)
        
        random_num = np.random.rand(len(return_p_m)) # function for random z-score,random choose len(returns) times
        z = norm.ppf(random_num).round(4) #z-score
        VaR_90 = []
        VaR_95 = []
        VaR_99 = []
        CVaR_90 = []
        CVaR_95 = []
        CVaR_99 = []
        n_90 = int(0.1*return_p_m.count())
        n_95 = int(0.05*return_p_m.count())
        n_99 = int(0.01*return_p_m.count())
        
        for i in range(len(return_p_m)):
           z = norm.ppf(random_num).round(4)
           scenario_VaR = np.sort(mean[i] - z * portfolio_std[i])
           VaR_90.append(np.percentile(scenario_VaR, 0.1))
           VaR_95.append(np.percentile(scenario_VaR, 0.05))
           VaR_99.append(np.percentile(scenario_VaR, 0.01))
           CVaR_90.append(scenario_VaR[:n_90].mean()) 
           CVaR_95.append(scenario_VaR[:n_95].mean())   
           CVaR_99.append(scenario_VaR[:n_99].mean()) 
           
        return(VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99) 
           
              
    if option_1 == "90%"and option_2 == 'Month':
         options = {
            "title": {"text":"VaR & CVaR"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": month_list,
            },
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [
                {"name": "VaR",
                 "type": "line",
                 "data": mvar()[0]},
                {"name": "CVaR",
                 "type": "line",
                 "data": mvar()[1]},
            ],
        }
         st_echarts(options=options, height="500px")

    if option_1 == "95%"and option_2 == 'Month':
              options = {
              "title": {"text":"VaR & CVaR"},
              "tooltip": {"trigger": "axis"},
              "legend": {"data": ["VaR","CVaR"]},
              "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
              "toolbox": {"feature": {"saveAsImage": {}}},
              "xAxis": {
                  "type": "category",
                  "boundaryGap": False,
                  "data": month_list,
              },
              "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
              "series": [
                  {"name": "VaR",
                   "type": "line",
                   "data": mvar()[2]},
                  {"name": "CVaR",
                   "type": "line",
                   "data": mvar()[3]},
              ],
          }
              st_echarts(options=options, height="500px")

    if option_1 == "99%"and option_2 == 'Month':
              options = {
              "title": {"text":"VaR & CVaR"},
              "tooltip": {"trigger": "axis"},
              "legend": {"data": ["VaR","CVaR"]},
              "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
              "toolbox": {"feature": {"saveAsImage": {}}},
              "xAxis": {
                  "type": "category",
                  "boundaryGap": False,
                  "data": month_list,
              },
              "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
              "series": [
                  {"name": "VaR",
                   "type": "line",
                   "data": mvar()[4]},
                  {"name": "CVaR",
                   "type": "line",
                   "data": mvar()[5]},
              ],
          }
              st_echarts(options=options, height="500px")



################################   Trader 2 ################################### 
with tab2:
    
######################### weight od stock ##############################

    st.header("Trader 2")
    data_w = [{"value": weights, "name": stock_name, "itemStyle": {"color": colors_2[i]}} for i, (stock_name_2, weights_2) in enumerate(zip(stock_name_2, weights_2))]
    options = {
    "title": {"text": "Weight of stock", "subtext": "From Yfinance", "left": "center","textStyle": {"fontSize": 24}},
    "tooltip": {"trigger": "item"},
    "series": [
        {
            "name": "stock",
            "type": "pie",
            "radius": "50%",
            "data": [
                {"value": weights_2[i], "name": stock_name_2[i],"itemStyle": {"color": colors_2[i]}} for i in range(len(weights_2),)
            ],
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                }
            },
        }
    ],
}
    st_echarts(
    options=options, height="700px")
########################### Var & CVaR ****************************************
    
tab2.subheader("Market Expected VaR & CVaR" )


with tab2:
    st.write(pd.DataFrame({
    'Confidence Level': ["90%", "95%", "99%"],
    'Value at risk': [marketvar()[0], marketvar()[2], marketvar()[4]],
    'Conditional Value at risk': [marketvar()[1], marketvar()[3], marketvar()[5]],
}))

       
with tab2:
    tab2.subheader("Expected VaR & CVaR" )
    st.caption('This is using Parametric Method Mehtod to calculate .')
    
    tab2.var1, tab2.var2, tab2.var3 = st.columns(3)
    tab2.var1.metric("VaR of 90%", f"{round(np.mean(para_day_2()[0]),5)*100}%",round(np.mean(para_day_2()[0])+np.mean(para_day()[0]),5),delta_color="inverse")
    tab2.var2.metric("VaR of 95%", f"{round(np.mean(para_day_2()[2]),5)*100}%",round(np.mean(para_day_2()[2])+np.mean(para_day()[2]),5),delta_color="inverse")
    tab2.var3.metric("VaR of 99%", f"{round(np.mean(para_day_2()[4]),5)*100}%",round(np.mean(para_day_2()[4])+np.mean(para_day()[4]),5),delta_color="inverse")

    tab2.cvar1, tab2.cvar2, tab2.cvar3 = st.columns(3)
    tab2.cvar1.metric("CVaR of 90%", f"{round(np.mean(para_day_2()[1]),5)*100}%",round(np.mean(para_day_2()[1])+np.mean(para_day()[1]),5),delta_color="inverse")
    tab2.cvar2.metric("CVaR of 95%", f"{round(np.mean(para_day_2()[3]),5)*100}%",round(np.mean(para_day_2()[3])+np.mean(para_day()[3]),5),delta_color="inverse")
    tab2.cvar3.metric("CVaR of 99%", f"{round(np.mean(para_day_2()[5]),5)*100}%",round(np.mean(para_day_2()[5])+np.mean(para_day()[5]),5),delta_color="inverse")


###################         Other risk measurement     ########################
tab2.subheader("Other risk meaurement")
with tab2:
    tab2.col1, tab2.col2, tab2.col3,tab2.col4 = st.columns(4)
    tab2.col1.metric("Portfolio Beta", f"{round(portfolio_beta_2,2)}", round(portfolio_beta_2-portfolio_beta,3))
    tab2.col2.metric("Standard Deviation", f"{round(portfolio_std_dev_2,4)}",round(portfolio_std_dev_2-portfolio_std_dev,3))
    tab2.col3.metric("Sharpe ratio", f"{round(sharpe_ratio_2,2)}",round(sharpe_ratio_2-sharpe_ratio,3) )
    tab2.col4.metric("Treynor Ratio",f"{round(t_r_2,2)}",round(t_r_2-t_r,3))
    
##############   ########################  ######################   ############

data_2 = [{"value": beta_2, "name": stock_name_2, "itemStyle": {"color": colors_2[i]}} for i, (stock_name_2, beta_2) in enumerate(zip(stock_name_2, betas_2))]
data_std_dev_2 = [{"value": std_dev_2, "name": stock_name_2, "itemStyle": {"color": colors_2[i]}} for i, (stock_name_2, std_dev_2) in enumerate(zip(stock_name_2, std_dev_2))]

##################     beta & standard Deviation chart       ##############
with tab2:
        beta = {
            "title": {"text": "Beta of each stock"},
            "tooltip": {"trigger": "item", "formatter": "{a} <br/>{b}: {c}"},
            "legend": None,
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {"type": "value", "name": "Beta", "nameLocation": "middle"},
            "yAxis": {"type": "category", "data": list(stock_name_2)},
            "series": [{"data": data_2, "type": "bar"}],
        }
        st_echarts(options=beta, height="300px")
    
        std = {
            "title": {"text": "Standard Deviation of each stock"},
            "xAxis": {
                "type": "category",
                "data": list(stock_name_2),
                "axisLabel": {"fontSize": 10, "interval": 0, "lineHeight": 14},},
            "yAxis": {"type": "value"},
            "series": [{"data": data_std_dev_2 , "type": "bar"}],
            "tooltip": {
                "trigger": "axis",
                "formatter": "{b}: {c}",},}
        st_echarts(options=std, height="300px", width="800px")

################## ################## ################## ################## #####
       

with tab2:
    t_1,t_2,t_3 = st.tabs(["📈 Historical Method VaR & CVaR", "💲Parametric Method VaR & CVaR", "💰Monte Carlo Method VaR & CVaR"])
    st.caption('********* The value in first new date/month is horinzotal,becasue of using as histrocal data .')
#########################     Historical Method       ###########################          
with t_1: 
    confidence_level, period = st.columns(2)
    with t_1: 
        confidence_level, period = st.columns(2)
    with confidence_level:
      confidence_level = st.radio("Confidence Level", ("90%","95%","99%"), key='confidence_level_t1_4')
      
    with period:
          period = st.selectbox('Period', ('Day', 'Month'),key='period_t1_1_4')


    st.subheader("Historical Method VaR & CVaR")
    
    if confidence_level == "90%" and period == 'Day':
        VaR_90_hd1 = round(return_p_2.rolling(window=250).quantile(0.10),4)
        VaR_90_hd1 = np.array(np.nan_to_num(VaR_90_hd1, nan=0))
        #av = (VaR_90_Day[VaR_90_Day<0]).tolist()
        #CVaR = [round(return_p[return_p<a].mean(),4) for a in av] # Calculate CVaR_90_Day
        av = (VaR_90_hd1).tolist()
        CVaR_90_hd1 = [round(return_p_2[return_p_2<a].mean(),4) for a in av] # Calculate CVaR_90_Day
      

      
        options = {
            "title": {"text": "VaR_90"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": date_list,},
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [{
                "name": "VaR",
                "type": "line",
                "data":  VaR_90_hd1.tolist()
                },
                {"name": "CVaR",
                 "type": "line",
                 "data": CVaR_90_hd1
                    }
                ]}
       
        st_echarts(options=options, height="400px") 
        
        plt.hist(VaR_90_hd1, bins=50)
        plt.hist(CVaR_90_hd1, bins=50)
        plt.xlabel('Historical var vs Cvar of trader 2 in 90% Cofidence Level')
        plt.show()  
               
        
    if confidence_level == "95%" and period == 'Day':
        VaR_95_hd1 = round(return_p.rolling(window=250).quantile(0.05),4)
        VaR_95_hd1 = np.nan_to_num(VaR_95_hd1, nan=0)
        av = (VaR_95_hd1).tolist()
        CVaR_95_hd1 = [round(return_p_2[return_p_2<a].mean(),4) for a in av] # Calculate CVaR_90_Day
        
        options = {
            "title": {"text": "VaR_95"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": date_list,},
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [{
                "name": "VaR",
                "type": "line",
                "data":  VaR_95_hd1.tolist()
                },
                {"name": "CVaR",
                 "type": "line",
                 "data": CVaR_95_hd1
                    }
                ]}
        st_echarts(options=options, height="400px") 
        
    if confidence_level == "99%" and period == 'Day':
      VaR_99_hd3 = round(return_p.rolling(window=250).quantile(0.01),4)
      VaR_99_hd3 = np.nan_to_num(VaR_99_hd3, nan=0)
      av = (VaR_99_hd3).tolist()
      CVaR_99_hd3 = [round(return_p_2[return_p_2<a].mean(),4) for a in av] # Calculate CVaR
      
      options = {
          "title": {"text": "VaR_99"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
          "xAxis": {
              "type": "category",
              "boundaryGap": False,
              "data": date_list,},
              "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
              "series": [{
                  "name": "VaR",
                  "type": "line",
                  "data": VaR_99_hd3.tolist()},
                  {"name": "CVaR",
                   "type": "line",
                   "data": CVaR_99_hd3}]}
      
      st_echarts(options=options, height="400px") 
    
    if confidence_level == "90%"and period == 'Month':
      VaR_90_hm1 = round(return_p_m_2.rolling(window=20).quantile(0.10),4)
      VaR_90_hm1 = np.nan_to_num(VaR_90_hm1, nan=0)
      av = (VaR_90_hm1).tolist()
      CVaR_90_hm1 = [round(return_p_m_2[return_p_m_2<a].mean(),4) for a in av] # Calculate CVaR
      
      options = {
          "title": {"text": "VaR_90"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
        "type": "category",
        "boundaryGap": False,
        "data": month_list,},
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
                {
            "name": "VaR",
            "type": "line",
            "data": VaR_90_hm1.tolist()},
            {"name": "CVaR",
             "type": "line",
             "data": CVaR_90_hm1}]}
      st_echarts(options=options, height="400px") 
      
    if confidence_level == "95%"and period == 'Month':
          VaR_95_hm = round(return_p_m_2.rolling(window=20).quantile(0.10),4)
          VaR_95_hm = np.nan_to_num(VaR_95_hm, nan=0)
          av = (VaR_95_hm).tolist()
          CVaR_95_hm = [round(return_p_m_2[return_p_m_2<a].mean(),4) for a in av] # Calculate CVaR
          
          options = {
        "title": {"text": "VaR_95"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": month_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {
                "name": "VaR",
                "type": "line",
                "data": VaR_95_hm.tolist()}, {"name": "CVaR",
                  "type": "line",
                  "data": CVaR_95_hm}]}
          st_echarts(options=options, height="400px")
          
    if confidence_level == "99%"and period == 'Month':
       VaR_99_hm = round(return_p_m_2.rolling(window=20).quantile(0.01),4)
       VaR_99_hm = np.nan_to_num(VaR_99_hm, nan=0)
       av = (VaR_99_hm).tolist()
       CVaR_99_hm = [round(return_p_m_2[return_p_m_2<a].mean(),4) for a in av] # Calculate CVaR
       
       options = {
     "title": {"text": "VaR_99"},
     "tooltip": {"trigger": "axis"},
     "legend": {"data": ["VaR","CVaR"]},
     "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
     "toolbox": {"feature": {"saveAsImage": {}}},
     "xAxis": {
         "type": "category",
         "boundaryGap": False,
         "data": month_list,
     },
     "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
     "series": [
         {
             "name": "VaR_99",
             "type": "line",
             "data": VaR_99_hm.tolist()},
             {"name": "CVaR",
               "type": "line",
               "data": CVaR_99_hm}]}
                                        
                                        
       st_echarts(options=options, height="400px")
     
       
         
#########################  #########################   ###########################        
     
#########################     Parametric Method       ###########################   
with t_2: 
    option_1, option_2 = st.columns(2)
    with option_1:
      option_1= st.radio("Confidence Level", ("90%","95%","99%"), key='confidence_level_t2_1')

    with option_2:
        option_2 = st.selectbox('Period', ('Day', 'Month'),key='period_t2_1')
        
    st.subheader("Parametric Method VaR & CVaR")
        
    if option_1 == "90%" and option_2 == 'Day':
        

        options = {
            "title": {"text":"VaR & CVaR"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": date_list,
            },
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [
                {"name": "VaR",
                 "type": "line",
                 "data": para_day_2()[0].tolist()},
                {"name": "CVaR",
                 "type": "line",
                 "data": para_day_2()[1].tolist()},
            ],
        }
        st_echarts(options=options, height="500px")
        
        plt.hist(para_day_2()[0], bins=50)
        plt.hist(para_day_2()[1], bins=50)
        plt.xlabel('Parametric var vs Cvar of trader 2 in 90% Confidence Level' )
        plt.show()
        
    if option_1 == "95%" and option_2 == 'Day':
     
      options = {
          "title": {"text":"VaR & CVaR"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
          "xAxis": {
              "type": "category",
              "boundaryGap": False,
              "data": date_list,
          },
          "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
          "series": [
              {"name": "VaR",
               "type": "line",
               "data": para_day_2()[2].tolist()},
              {"name": "CVaR",
               "type": "line",
               "data": para_day_2()[3].tolist()},
          ],
      }
      st_echarts(options=options, height="500px")  
    if option_1 == "99%" and option_2 == 'Day':
     
      options = {
          "title": {"text":"VaR & CVaR"},
          "tooltip": {"trigger": "axis"},
          "legend": {"data": ["VaR","CVaR"]},
          "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
          "toolbox": {"feature": {"saveAsImage": {}}},
          "xAxis": {
              "type": "category",
              "boundaryGap": False,
              "data": date_list,
          },
          "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
          "series": [
              {"name": "VaR",
               "type": "line",
               "data": para_day_2()[4].tolist()},
              {"name": "CVaR",
               "type": "line",
               "data": para_day_2()[5].tolist()},
          ],
      }
      st_echarts(options=options, height="500px")  
           
    if option_1 == "90%" and option_2 == 'Month':
       
        options = {
        "title": {"text":"VaR & CVaR"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": month_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {"name": "VaR",
             "type": "line",
             "data": para_month_2()[0].tolist()},
            {"name": "CVaR",
             "type": "line",
             "data": para_month_2()[1].tolist()},
        ],
    }
        st_echarts(options=options, height="500px") 
       
        
    if option_1 == "95%" and option_2 == 'Month':
       
        options = {
        "title": {"text":"VaR & CVaR"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": month_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {"name": "VaR",
             "type": "line",
             "data": para_month_2()[2].tolist()},
            {"name": "CVaR",
             "type": "line",
             "data": para_month_2()[3].tolist()},
        ],
    }
        st_echarts(options=options, height="500px") 
        
    if option_1 == "99%" and option_2 == 'Month':
        
     
      options = {
      "title": {"text":"VaR & CVaR"},
      "tooltip": {"trigger": "axis"},
      "legend": {"data": ["VaR","CVaR"]},
      "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
      "toolbox": {"feature": {"saveAsImage": {}}},
      "xAxis": {
          "type": "category",
          "boundaryGap": False,
          "data": month_list,
      },
      "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
      "series": [
          {"name": "VaR",
           "type": "line",
           "data": para_month_2()[4].tolist()},
          {"name": "CVaR",
           "type": "line",
           "data": para_month_2()[5]._p3.tolist()},
      ],
  }
      st_echarts(options=options, height="500px")    
      
     

######################### ######################### ######################### 
with t_3: 
    option_1, option_2 = st.columns(2)
    with option_1:
      option_1 = st.radio("Confidence Level", ("90%","95%","99%"), key='confidence_level_t3_3')

    with option_2:
        option_2 = st.selectbox('Period', ('Day', 'Month'),key='period_t3_3')
        
        def var2(): 
            mean_1 = round(return_p_2.rolling(window=250).mean(),4) # group 150 daily VaR as a sample
            mean_1 = np.nan_to_num(mean_1, nan=0)
            portfolio_std = return_p_2.rolling(window=250).std() 
            portfolio_std = np.nan_to_num(portfolio_std, nan=0)  # group 150 daily VaR as a sample tp find standard deviation
            
            random_num = np.random.rand(len(return_p)) # function for random z-score,random choose len(returns) times
            z = norm.ppf(random_num).round(4) #z-score
            scenario_VaR =pd.Series(mean_1-z*portfolio_std).sort_values()
            VaR_90 = []
            VaR_95 = []
            VaR_99 = []
            CVaR_90 = []
            CVaR_95 = []
            CVaR_99 = []
            n_90 = int(0.1*return_p.count())
            n_95 = int(0.05*return_p.count())
            n_99 = int(0.01*return_p.count())
            
            for i in range(len(return_p)):
               z = norm.ppf(random_num).round(4)
               scenario_VaR = np.sort(mean_1[i] - z * portfolio_std[i])
               VaR_90.append(np.percentile(scenario_VaR, 100 - 100 * 0.9))
               VaR_95.append(np.percentile(scenario_VaR, 100 - 100 * 0.95))
               VaR_99.append(np.percentile(scenario_VaR, 100 - 100 * 0.99))
               CVaR_90.append(scenario_VaR[:n_90].mean()) 
               CVaR_95.append(scenario_VaR[:n_95].mean())   
               CVaR_99.append(scenario_VaR[:n_99].mean())   

            return(VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99)
        
        def mvar2():
            
            mean_1 = round(return_p_m_2.rolling(window=15).mean(),4)
            mean_1 = np.nan_to_num(mean_1, nan=0)
            portfolio_std = return_p_m_2.rolling(window=20).std()
            portfolio_std = np.nan_to_num(portfolio_std, nan=0)
            
            random_num = np.random.rand(len(return_p_m_2)) # function for random z-score,random choose len(returns) times
            z = norm.ppf(random_num).round(4) #z-score
            scenario_VaR =pd.Series(mean_1-z*portfolio_std).sort_values()
            VaR_90 = []
            VaR_95 = []
            VaR_99 = []
            CVaR_90 = []
            CVaR_95 = []
            CVaR_99 = []
            n_90 = int(0.1*return_p_m_2.count())
            n_95 = int(0.05*return_p_m_2.count())
            n_99 = int(0.01*return_p_m_2.count())
            
            for i in range(len(return_p_m_2)):
               z = norm.ppf(random_num).round(4)
               scenario_VaR = np.sort(mean_1[i] - z * portfolio_std[i])
               VaR_90.append(np.percentile(scenario_VaR,0.1))
               VaR_95.append(np.percentile(scenario_VaR, 0.05))
               VaR_99.append(np.percentile(scenario_VaR, 0.01))
               CVaR_90.append(scenario_VaR[:n_90].mean()) 
               CVaR_95.append(scenario_VaR[:n_95].mean())   
               CVaR_99.append(scenario_VaR[:n_99].mean())   

            return(VaR_90,CVaR_90,VaR_95,CVaR_95,VaR_99,CVaR_99)
        

    st.subheader("Mento Carlo Method VaR & CVaR")  

    if option_1 == "90%"and option_2 == 'Day':
        options = {
        "title": {"text":"VaR & CVaR"},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": ["VaR","CVaR"]},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "toolbox": {"feature": {"saveAsImage": {}}},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": date_list,
        },
        "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
        "series": [
            {"name": "VaR",
             "type": "line",
             "data": var2()[0]},
            {"name": "CVaR",
             "type": "line",
             "data": var2()[1]},
        ],
    }
        st_echarts(options=options, height="500px")     
        

        
      
    if option_1 == "95%"and option_2 == 'Day':
         options = {
         "title": {"text":"VaR & CVaR"},
         "tooltip": {"trigger": "axis"},
         "legend": {"data": ["VaR","CVaR"]},
         "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
         "toolbox": {"feature": {"saveAsImage": {}}},
         "xAxis": {
             "type": "category",
             "boundaryGap": False,
             "data": date_list,
         },
         "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
         "series": [
             {"name": "VaR",
              "type": "line",
              "data": var2()[2]},
             {"name": "CVaR",
              "type": "line",
              "data": var2()[3]},
         ],
     }
         st_echarts(options=options, height="500px")     
    if option_1 == "99%"and option_2 == 'Day':     
         options = {
         "title": {"text":"VaR & CVaR"},
         "tooltip": {"trigger": "axis"},
         "legend": {"data": ["VaR","CVaR"]},
         "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
         "toolbox": {"feature": {"saveAsImage": {}}},
         "xAxis": {
             "type": "category",
             "boundaryGap": False,
             "data": date_list,
         },
         "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
         "series": [
             {"name": "VaR",
              "type": "line",
              "data": var2()[4]},
             {"name": "CVaR",
              "type": "line",
              "data": var2()[5]},
         ],
     }
         st_echarts(options=options, height="500px")
    
    
           
              
    if option_1 == "90%"and option_2 == 'Month':
         options = {
            "title": {"text":"VaR & CVaR"},
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["VaR","CVaR"]},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "toolbox": {"feature": {"saveAsImage": {}}},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": month_list,
            },
            "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
            "series": [
                {"name": "VaR",
                 "type": "line",
                 "data": mvar2()[0]},
                {"name": "CVaR",
                 "type": "line",
                 "data": mvar2()[1]},
            ],
        }
         st_echarts(options=options, height="500px")

    if option_1 == "95%"and option_2 == 'Month':
              options = {
              "title": {"text":"VaR & CVaR"},
              "tooltip": {"trigger": "axis"},
              "legend": {"data": ["VaR","CVaR"]},
              "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
              "toolbox": {"feature": {"saveAsImage": {}}},
              "xAxis": {
                  "type": "category",
                  "boundaryGap": False,
                  "data": month_list,
              },
              "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
              "series": [
                  {"name": "VaR",
                   "type": "line",
                   "data": mvar2()[2]},
                  {"name": "CVaR",
                   "type": "line",
                   "data": mvar2()[3]},
              ],
          }
              st_echarts(options=options, height="500px")

    if option_1 == "99%"and option_2 == 'Month':
              options = {
              "title": {"text":"VaR & CVaR"},
              "tooltip": {"trigger": "axis"},
              "legend": {"data": ["VaR","CVaR"]},
              "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
              "toolbox": {"feature": {"saveAsImage": {}}},
              "xAxis": {
                  "type": "category",
                  "boundaryGap": False,
                  "data": month_list,
              },
              "yAxis": {"type": 'value', "inverse": True, "min": None, "max": None},
              "series": [
                  {"name": "VaR",
                   "type": "line",
                   "data": mvar2()[4]},
                  {"name": "CVaR",
                   "type": "line",
                   "data": mvar2()[5]},
              ],
          }
              st_echarts(options=options, height="500px")
            
plt.hist(var2()[0], bins=50)
plt.hist(var2()[1], bins=50)
plt.xlabel('Monte Carlo var vs Cvar of trader 2 in 90% Confidence Level')
plt.show()


  ##### test case #####
data = pd.read_csv("test_var.csv")
var_data= data["Portfolio"][250:2340]
test_data = VaR_90_hd1[250:2340]
pearsons_coefficient = np.corrcoef(var_data, test_data)[0][1]    ## test trader_2 historical method
 ## test trader_2 historical method
 
var_data_1= data["Portfolio_P"][250:2340]
test_data_1 = para_day_2()[0][250:2340]
pearsons_coefficient_1 = np.corrcoef(var_data_1, test_data_1)[0][1]
 ## test trader_2 parametric method

        
    
