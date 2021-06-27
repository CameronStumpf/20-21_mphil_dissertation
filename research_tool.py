
from numpy.core.numeric import False_
import streamlit as st
import SessionState
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from PIL import Image
import time



#Stage class definition
class Stage:
    '''
    contains a process and an inventory
    necessary to first define each node, then the respective supplier/customer arcs
    '''
    
    def __init__(self, leadTime, stock, directCost, mu, sigma, z):
        self.mu = mu
        self.sigma = sigma
        self.z = z
        self.stock = stock

        self.s_in = 0  #inbound service time - diff. order placed and order received
        self.s_out = 0 #outbound service time will remain 0 if safety stock is stored
        self.l = leadTime  #lead time - time to complete processing activity
        
        self.directCost = directCost  #cost added by this process specifically
        self.cumulativeCost = directCost #cumulative cost added over process

        while self.stock == False:
            #RECALCULATE
            self.tau = self.s_in + self.l - self.s_out  #net replenishment time
            self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory
            
            if self.E == 0:
                break
            self.s_out += 1
        

        #before we can re-run, we need an initial calc
        self.tau = self.s_in + self.l - self.s_out  #net replenishment time
        self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory

        #if stock set to False, increase s_out until no stock stored
        #find minimum s_out to allow zero safety stock
        while self.stock == False:
            #RECALCULATE
            self.tau = self.s_in + self.l - self.s_out  #net replenishment time
            self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory
            
            if self.E == 0:
                break
            self.s_out += 1
        
        self.h = 1 * self.cumulativeCost
        self.Eh = self.E*self.h  #"STOCK VALUE
        


    def suppliers(self, *args):
        #FIND NEW S_IN AND CUMULATIVE COST
        #BY LOOKING AT S_OUT AND CUMULATIVE COST OF SUPPLIERS
        #CALCULATE A NEW S_OUT

        for item in args:
            if item.s_out > self.s_in:
                self.s_in = item.s_out
            self.cumulativeCost += item.cumulativeCost #cumulative cost in process so far
        
        #find the minimum s_out to work
        while self.stock == False:
            #RECALCULATE
            self.tau = self.s_in + self.l - self.s_out  #net replenishment time
            self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory
            
            if self.E == 0:
                break
            self.s_out += 1


        #before we can re-run, we need an initial calc
        self.tau = self.s_in + self.l - self.s_out  #net replenishment time
        self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory

        #if stock set to False, increase s_out until no stock stored
        #find minimum s_out to allow zero safety stock
        while self.stock == False:
            #RECALCULATE
            self.tau = self.s_in + self.l - self.s_out  #net replenishment time
            self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory
            
            if self.E == 0:
                break
            self.s_out += 1
        
        self.h = 1 * self.cumulativeCost
        self.Eh = self.E*self.h  #"STOCK VALUE

        

        
    def customers(self, *args):
        #FINDING NEW MU, SIGMA, Z
        #BY LOOKING AT MU, SIGMA, Z OF CUSTOMER PROCESSES
        #CALCULATE A NEW S_OUT

        if args != ():
            #NEW MU, SIGMA, Z
            temp_sigma = 0
            for item in args:
                self.z = item.z
                temp_sigma += item.sigma * item.sigma
                self.mu += item.mu
            self.sigma = math.sqrt(temp_sigma)

            #find the minimum s_out to work
            if self.stock == False:
                self.s_out = 0
            while self.stock == False:
                #RECALCULATE
                self.tau = self.s_in + self.l - self.s_out  #net replenishment time
                self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory
                
                if self.E == 0:
                    break
                self.s_out += 1
        
        #before we can re-run, we need an initial calc
        self.tau = self.s_in + self.l - self.s_out  #net replenishment time
        self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory

        #if stock set to False, increase s_out until no stock stored
        #find minimum s_out to allow zero safety stock
        while self.stock == False:
            #RECALCULATE
            self.tau = self.s_in + self.l - self.s_out  #net replenishment time
            self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory
            
            if self.E == 0:
                break
            self.s_out += 1
        
        self.h = 1 * self.cumulativeCost
        self.Eh = self.E*self.h  #"STOCK VALUE

            


    def calculate(self):
        #before we can re-run, we need an initial calc
        self.tau = self.s_in + self.l - self.s_out  #net replenishment time
        self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory

        #if stock set to False, increase s_out until no stock stored
        #find minimum s_out to allow zero safety stock
        while self.stock == False:
            #RECALCULATE
            self.tau = self.s_in + self.l - self.s_out  #net replenishment time
            self.E = self.z*self.sigma*math.sqrt(self.tau) #expected inventory
            
            if self.E == 0:
                break
            self.s_out += 1
        
        self.h = 1 * self.cumulativeCost
        self.Eh = self.E*self.h  #"STOCK VALUE




#######################################
###      Introduction Section       ###
#######################################

#st.set_page_config(layout="wide")
#Import Image
image = Image.open('pharma_example.png')
st.image(image, caption = 'pharma_example', use_column_width=True)


'''

---------------------------------------------------

'''


#######################################
###       demand variability        ###
#######################################

#load & clean data
df_demand = pd.read_csv("./data/demand_data.csv")
df_demand = df_demand.interpolate(method='linear') #interpolate missing data
df_demand[["Drug M, Market C", "Drug M, Market B", "Drug H, Market A"]] = df_demand[["Drug M, Market C", "Drug M, Market B", "Drug H, Market A"]].divide(1000000)

titleFontDict = {
    'weight': 'normal',
    'size': 18
}

tickFontDict = {
    'weight': 'normal',
    'size': 7
}




############################
##    DRUG M, MARKET C    ##
############################

#PLACEHOLDER FOR GRAPH
chartCols = st.beta_columns(1)

#SLIDERS
cols = st.beta_columns(2)
L_mu_value = float(round(df_demand["Drug M, Market C"].mean(), 3))
L_sigma_value = float(round(df_demand["Drug M, Market C"].std(), 3))

L_rollingAverage = cols[0].slider("M_C - Rolling Average, [1]", min_value=1, max_value=13, value=1, step=1)
L_mu = cols[1].slider("M_C - Average Demand (Mu), ["+str(L_mu_value)+"]", min_value=0.00, max_value=1.00, value=L_mu_value, step=0.05)
L_sigma = cols[0].slider("M_C - Standard Deviation, ["+str(L_sigma_value)+"]", min_value=0.00, max_value=0.50, value=L_sigma_value, step=0.02)
L_confidence = cols[1].slider("M_C - Confidence Interval (Z), [95.0%]", min_value=75.0, max_value=99.90, value=95.0, step=0.1)
L_z = stats.norm.ppf(L_confidence/100)

#PLOT - DATA AND EXPECTED DEMAND
fig_L, ax_L = plt.subplots(figsize=(10,4))
ax_L.set_title("Unit Sales (Million) per Week for Drug M in Market C", titleFontDict)
ax_L.plot(df_demand['Week'], df_demand['Drug M, Market C'].rolling(L_rollingAverage).mean())
ax_L.set(xlim=(-51, 13), ylim=(0, 1.5))

ax_L.axvline(x=0, ymin=0, ymax=1, color='black', linewidth = 1.5, linestyle='--') # -3 S.D. FROM MEAN

ax_L.axhline(y=L_mu, xmin=53/65, xmax=64/65, color='green', linewidth = 2.0, label="Average (Mu)") #MU LINE
ax_L.axhline(y=L_mu + L_z*L_sigma, xmin=53/65, xmax=64/65, color='blue', linewidth = 2.0) #confidence interval from ean

ax_L.text(0.8,L_mu + L_z*L_sigma + 0.02, str(round(L_confidence, 2))+'% Service Level',rotation=0)
ax_L.text(0.8,L_mu + 0.02, 'Av. Demand: '+str(round(L_sigma, 2))+'M',rotation=0)

ax_L.axhline(y=L_mu + 1*L_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +1 S.D. FROM MEAN
ax_L.axhline(y=L_mu + 2*L_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +2 S.D. FROM MEAN
ax_L.axhline(y=L_mu + 3*L_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +3 S.D. FROM MEAN
ax_L.axhline(y=L_mu - 1*L_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -1 S.D. FROM MEAN
ax_L.axhline(y=L_mu - 2*L_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -2 S.D. FROM MEAN
ax_L.axhline(y=L_mu - 3*L_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -3 S.D. FROM MEAN

ax_L.set_xticks([-51, -39, -26, -13, 0, 6.5])
ax_L.set_xticklabels(["Q -4", "Q -3", "Q -2", "Q -1", "Present\nDay", "Projected\nDemand\nVariability"])

chartCols[0].pyplot(fig_L)


'''

---------------------------------------------------

'''


############################
##    DRUG M, MARKET B    ##
############################

#PLACEHOLDER FOR GRAPH
chartCols = st.beta_columns(1)

#SLIDERS
cols = st.beta_columns(2)
K_mu_value = float(round(df_demand["Drug M, Market B"].mean(), 3))
K_sigma_value = float(round(df_demand["Drug M, Market B"].std(), 3))

K_rollingAverage = cols[0].slider("M_B - Rolling Average, [1]", min_value=1, max_value=13, value=1, step=1)
K_mu = cols[1].slider("M_B - Average Demand (Mu), ["+str(K_mu_value)+"]", min_value=0.00, max_value=1.00, value=K_mu_value, step=0.05)
K_sigma = cols[0].slider("M_B - Standard Deviation, ["+str(K_sigma_value)+"]", min_value=0.00, max_value=0.50, value=K_sigma_value, step=0.02)
K_confidence = cols[1].slider("M_B - Confidence Interval (Z), [95.0%]", min_value=75.0, max_value=99.9, value=95.0, step=0.1)
K_z = stats.norm.ppf(K_confidence/100)

#PLOT - DATA AND EXPECTED DEMAND
fig_K, ax_K = plt.subplots(figsize=(10,4))
ax_K.set_title("Unit Sales (Million) per Week for Drug M in Market B", titleFontDict)
ax_K.plot(df_demand['Week'], df_demand['Drug M, Market B'].rolling(K_rollingAverage).mean())
ax_K.set(xlim=(-51, 13), ylim=(0, 1.5))

ax_K.axvline(x=0, ymin=0, ymax=1, color='black', linewidth = 1.5, linestyle='--') # -3 S.D. FROM MEAN

ax_K.axhline(y=K_mu, xmin=53/65, xmax=64/65, color='green', linewidth = 2.0, label="Average (Mu)") #MU LINE
ax_K.axhline(y=K_mu + K_z*K_sigma, xmin=53/65, xmax=64/65, color='blue', linewidth = 2.0) #confidence interval from ean

ax_K.text(0.8,K_mu + K_z*K_sigma + 0.02, str(round(K_confidence, 2))+'% Service Level',rotation=0)
ax_K.text(0.8,K_mu + 0.02, 'Av. Demand: '+str(round(K_sigma, 2))+'M',rotation=0)

ax_K.axhline(y=K_mu + 1*K_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +1 S.D. FROM MEAN
ax_K.axhline(y=K_mu + 2*K_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +2 S.D. FROM MEAN
ax_K.axhline(y=K_mu + 3*K_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +3 S.D. FROM MEAN
ax_K.axhline(y=K_mu - 1*K_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -1 S.D. FROM MEAN
ax_K.axhline(y=K_mu - 2*K_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -2 S.D. FROM MEAN
ax_K.axhline(y=K_mu - 3*K_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -3 S.D. FROM MEAN

ax_K.set_xticks([-51, -39, -26, -13, 0, 6.5])
ax_K.set_xticklabels(["Q -4", "Q -3", "Q -2", "Q -1", "Present\nDay", "Projected\nDemand\nVariability"])

chartCols[0].pyplot(fig_K)


'''

---------------------------------------------------

'''


############################
##    DRUG H, MARKET A    ##
############################

#PLACEHOLDER FOR GRAPH
chartCols = st.beta_columns(1)

#SLIDERS
cols = st.beta_columns(2)
J_mu_value = float(round(df_demand["Drug H, Market A"].mean(), 3))
J_sigma_value = float(round(df_demand["Drug H, Market A"].std(), 3))

J_rollingAverage = cols[0].slider("H_A - Rolling Average, [1]", min_value=1, max_value=13, value=1, step=1)
J_mu = cols[1].slider("H_A - Average Demand (Mu), ["+str(J_mu_value)+"]", min_value=0.00, max_value=1.00, value=J_mu_value, step=0.05)
J_sigma = cols[0].slider("H_A - Standard Deviation, ["+str(J_sigma_value)+"]", min_value=0.00, max_value=0.50, value=J_sigma_value, step=0.02)
J_confidence = cols[1].slider("H_A - Confidence Interval (Z), [95.0%]", min_value=75.0, max_value=99.9, value=95.0, step=0.1)
J_z = stats.norm.ppf(J_confidence/100)

#PLOT - DATA AND EXPECTED DEMAND
fig_J, ax_J = plt.subplots(figsize=(10,4))
ax_J.set_title("Unit Sales per Week (Million) for Drug H in Market A", titleFontDict)
ax_J.plot(df_demand['Week'], df_demand['Drug H, Market A'].rolling(J_rollingAverage).mean())
ax_J.set(xlim=(-51, 13), ylim=(0, 1.5))

ax_J.axvline(x=0, ymin=0, ymax=1, color='black', linewidth = 1.5, linestyle='--') # -3 S.D. FROM MEAN

ax_J.axhline(y=J_mu, xmin=53/65, xmax=64/65, color='green', linewidth = 2.0, label="Average (Mu)") #MU LINE
ax_J.axhline(y=J_mu + J_z*J_sigma, xmin=53/65, xmax=64/65, color='blue', linewidth = 2.0) #confidence interval from ean

ax_J.text(0.8,J_mu + J_z*J_sigma + 0.02, str(round(J_confidence, 2))+'% Service Level',rotation=0)
ax_J.text(0.8,J_mu + 0.02, 'Av. Demand: '+str(round(J_sigma, 2))+'M',rotation=0)

ax_J.axhline(y=J_mu + 1*J_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +1 S.D. FROM MEAN
ax_J.axhline(y=J_mu + 2*J_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +2 S.D. FROM MEAN
ax_J.axhline(y=J_mu + 3*J_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # +3 S.D. FROM MEAN
ax_J.axhline(y=J_mu - 1*J_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -1 S.D. FROM MEAN
ax_J.axhline(y=J_mu - 2*J_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -2 S.D. FROM MEAN
ax_J.axhline(y=J_mu - 3*J_sigma, xmin=55/65, xmax=62/65, color='black', linewidth = 0.3) # -3 S.D. FROM MEAN

ax_J.set_xticks([-51, -39, -26, -13, 0, 6.5])
ax_J.set_xticklabels(["Q -4", "Q -3", "Q -2", "Q -1", "Present\nDay", "Projected\nDemand\nVariability"])

chartCols[0].pyplot(fig_J)


'''


---------------------------------------------------


'''

####################################################
###    Inventory Storage & Value Calculation     ###
####################################################


col_secondLast, = st.beta_columns(1)
image = Image.open('pharma_example.png')
col_secondLast.image(image, caption = 'pharma_example')

col4, col3, col2, col1 = st.beta_columns(4)

D_stock = col4.checkbox("D: Part00", value=False)
A_stock = col4.checkbox("A: Chem01", value=False)
B_stock = col4.checkbox("B: Chem02", value=False)

E_stock = col3.checkbox("E: Part01", value=False)
G_stock = col3.checkbox("G: Excip", value=False)
C_stock = col3.checkbox("C: API", value=False)
F_stock = col3.checkbox("F: Part02", value=False)

H_stock = col2.checkbox("H: Drug M", value=False)
I_stock = col2.checkbox("I: Drug H", value=False)

L_stock = col1.checkbox("L: Packing mkt C", value=True)
K_stock = col1.checkbox("K: Packing mkt B", value=True)
J_stock = col1.checkbox("J: Packing mkt A", value=True)

col_last = st.beta_columns(1)  ##for the table
cols_last2 = st.beta_columns(2)  ##for inventory value and calculate button


if cols_last2[0].button("Calculate Inventory"):

    #lead time / stock stored here? / cost added / mu / sigma / z
    z=2
    D = Stage(1, D_stock, 10, 0, 0, z)  #stage 4
    A = Stage(4, A_stock, 50, 0, 0, z)  #stage 4
    B = Stage(1, B_stock, 100, 0, 0, z)  #stage 4

    E = Stage(1, E_stock, 3, 0, 0, z)  #stage 3
    G = Stage(2, G_stock, 8, 0, 0, z)  #stage 3
    C = Stage(2, C_stock, 80, 0, 0, z)  #stage 3
    F = Stage(3, F_stock, 5, 0, 0, z)  #stage 3

    H = Stage(3, H_stock, 120, 0, 0, z)  #stage 2
    I = Stage(1, I_stock, 180, 0, 0, z)  #stage 2

    L = Stage(10, L_stock, 30, L_mu, L_sigma, L_z)  #stage 1
    K = Stage(2, K_stock, 25, K_mu, K_sigma, K_z)  #stage 1
    J = Stage(2, J_stock, 20, J_mu, J_sigma, J_z)  #stage 1


    #DEFINE SUPPLIERS
    D.suppliers()  #stage 4
    A.suppliers()  #stage 4
    B.suppliers()  #stage 4

    E.suppliers(D)  #stage 3
    G.suppliers()  #stage 3
    C.suppliers(A, B)  #stage 3
    F.suppliers()  #stage 3

    H.suppliers(E, G, C)  #stage 2
    I.suppliers(C, F)  #stage 2

    L.suppliers(H)  #stage 1
    K.suppliers(H)  #stage 1
    J.suppliers(I)  #stage 1


    #DEFINE CUSTOMERS
    L.customers()  #stage 1
    K.customers()  #stage 1
    J.customers()  #stage 1

    H.customers(L, K)  #stage 2
    I.customers(J)  #stage 2

    E.customers(H)  #stage 3
    G.customers(H)  #stage 3
    C.customers(H, I)  #stage 3
    F.customers(I)  #stage 3

    D.customers(E)  #stage 4
    A.customers(C)  #stage 4
    B.customers(C)  #stage 4


    #CALCULATE
    L.calculate()  #stage 1
    K.calculate()  #stage 1
    J.calculate()  #stage 1

    H.calculate()  #stage 2
    I.calculate()  #stage 2

    E.calculate()  #stage 3
    G.calculate()  #stage 3
    C.calculate()  #stage 3
    F.calculate()  #stage 3

    D.calculate()  #stage 4
    A.calculate()  #stage 4
    B.calculate()  #stage 4





    stageList = [
        D,A,B, E,G,C,F, H,I, L,K,J
    ]
    nameList = [
        "D - Part00",
        "A - chem01",
        "B - chem02",
        "E - Part01",
        "G - Excipient",
        "C - API",
        "F - Part02",
        "H - Drug M",
        "I - Drug H",
        "L - Packing mkt C",
        "K - Packing mkt B",
        "J - Packing mkt A",
    ]

    df = pd.DataFrame(columns = [
        "Stage Name",
        "Mu",
        "Sigma",
        "Z",
        "S_out",
        "S_in",
        "Unit Value",
        "Inv.",
        "Inv. Value"
    ])

    inventory_cost = 0
    for i in range(len(stageList)):
        row = {
            "Stage Name" : nameList[i],
            "Mu" : "%.3fM" % stageList[i].mu,
            "Sigma" : "%.3f" % stageList[i].sigma,
            "Z" : "%.3f" % stageList[i].z,
            "S_out" : "%.0f" % stageList[i].s_out,
            "S_in" : "%.0f" % stageList[i].s_in,
            "Unit Value" : "£%.0f" % stageList[i].cumulativeCost,
            "Inv." : "%.3fM" % stageList[i].E,
            "Inv. Value" : "£%.2fM" % stageList[i].Eh
        }
        df = df.append(row, ignore_index=True)
        inventory_cost += stageList[i].Eh

    df = df.set_index('Stage Name')



    col_last[0].write(df)
    cols_last2[1].write("Inventory Value: " "£%.1fM" % inventory_cost)







