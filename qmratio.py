from scipy.stats import linregress
import matplotlib.pyplot as mt
import math
import numpy as np
import statistics as st
data_100v = [2.4,1.6,1.0,0.8]
data_110v = [2.6,1.6,1.2,0.9]
data_120v = [2.7,1.8,1.2,1.0]
data_130v = [2.8,1.8,1.3,1.0]
data_140v = [3.0,1.9,1.4,1.1]
radius = [2.0,3.0,4.0,5.0]
b2_100v,b2_110v,b2_120v,b2_130v,b2_140v = [],[],[],[],[]
other_100v,other_110v,other_120v,other_130v,other_140v = [],[],[],[],[]
mo = ((4 * math.pi) / 10**(7))
for i in range(4):
    other_100v.append((2 * 100) * 10000 / radius[i]**2)
    other_110v.append((2 * 110) * 10000 / radius[i]**2)
    other_120v.append((2 * 120) * 10000 / radius[i]**2)
    other_130v.append((2 * 130) * 10000 / radius[i]**2)
    other_140v.append((2 * 140) * 10000 / radius[i]**2)
    b2_100v.append(((8 * mo * data_100v[i] * 154)/(math.sqrt(125) * 0.2))**2)
    b2_110v.append(((8 * mo * data_110v[i] * 154)/(math.sqrt(125) * 0.2))**2)
    b2_120v.append(((8 * mo * data_120v[i] * 154)/(math.sqrt(125) * 0.2))**2)
    b2_130v.append(((8 * mo * data_130v[i] * 154)/(math.sqrt(125) * 0.2))**2)
    b2_140v.append(((8 * mo * data_140v[i] * 154)/(math.sqrt(125) * 0.2))**2)
real_other_data = np.array([other_100v[0],other_110v[0],other_120v[0],other_130v[0],other_140v[0]])
real_b2_data = np.array([b2_100v[0],b2_110v[0],b2_120v[0],b2_130v[0],b2_140v[0]])
real_other_data1 = np.array([other_100v[1],other_110v[1],other_120v[1],other_130v[1],other_140v[1]])
real_b2_data1 = np.array([b2_100v[1],b2_110v[1],b2_120v[1],b2_130v[1],b2_140v[1]])
real_other_data2 = np.array([other_100v[2],other_110v[2],other_120v[2],other_130v[2],other_140v[2]])
real_b2_data2 = np.array([b2_100v[2],b2_110v[2],b2_120v[2],b2_130v[2],b2_140v[2]])
real_other_data3 = np.array([other_100v[3],other_110v[3],other_120v[3],other_130v[3],other_140v[3]])
real_b2_data3 = np.array([b2_100v[3],b2_110v[3],b2_120v[3],b2_130v[3],b2_140v[3]])
xerror1 = [(b2_100v[0] * math.sqrt(2) * math.sqrt((0.1/data_100v[0])**2+(0.01/20)**2)),(b2_110v[0] * math.sqrt(2) * math.sqrt((0.1/data_110v[0])**2+(0.01/20)**2)),(b2_120v[0] * math.sqrt(2) * math.sqrt((0.1/data_120v[0])**2+(0.01/20)**2)),(b2_130v[0] * math.sqrt(2) * math.sqrt((0.1/data_130v[0])**2+(0.01/20)**2)),(b2_140v[0] * math.sqrt(2) * math.sqrt((0.1/data_140v[0])**2+(0.01/20)**2))]
xerror2 = [(b2_100v[1] * math.sqrt(2) * math.sqrt((0.1/data_100v[1])**2+(0.01/20)**2)),(b2_110v[1] * math.sqrt(2) * math.sqrt((0.1/data_110v[1])**2+(0.01/20)**2)),(b2_120v[1] * math.sqrt(2) * math.sqrt((0.1/data_120v[1])**2+(0.01/20)**2)),(b2_130v[1] * math.sqrt(2) * math.sqrt((0.1/data_130v[1])**2+(0.01/20)**2)),(b2_140v[1] * math.sqrt(2) * math.sqrt((0.1/data_140v[1])**2+(0.01/20)**2))]
xerror3 = [(b2_100v[2] * math.sqrt(2) * math.sqrt((0.1/data_100v[2])**2+(0.01/20)**2)),(b2_110v[2] * math.sqrt(2) * math.sqrt((0.1/data_110v[2])**2+(0.01/20)**2)),(b2_120v[2] * math.sqrt(2) * math.sqrt((0.1/data_120v[2])**2+(0.01/20)**2)),(b2_130v[2] * math.sqrt(2) * math.sqrt((0.1/data_130v[2])**2+(0.01/20)**2)),(b2_140v[2] * math.sqrt(2) * math.sqrt((0.1/data_140v[2])**2+(0.01/20)**2))]
xerror4 = [(b2_100v[3] * math.sqrt(2) * math.sqrt((0.1/data_100v[3])**2+(0.01/20)**2)),(b2_110v[3] * math.sqrt(2) * math.sqrt((0.1/data_110v[3])**2+(0.01/20)**2)),(b2_120v[3] * math.sqrt(2) * math.sqrt((0.1/data_120v[3])**2+(0.01/20)**2)),(b2_130v[3] * math.sqrt(2) * math.sqrt((0.1/data_130v[3])**2+(0.01/20)**2)),(b2_140v[3] * math.sqrt(2) * math.sqrt((0.1/data_140v[3])**2+(0.01/20)**2))]
yerror1 = [(real_other_data[0] * math.sqrt((10/100)**2+(math.sqrt(2) * (0.001/2))**2)),(real_other_data[1] * math.sqrt((10/110)**2+(math.sqrt(2) * (0.001/2))**2)),(real_other_data[2] * math.sqrt((10/120)**2+(math.sqrt(2) * (0.001/2))**2)),(real_other_data[3] * math.sqrt((10/130)**2+(math.sqrt(2) * (0.001/2))**2)),(real_other_data[4] * math.sqrt((10/140)**2+(math.sqrt(2) * (0.001/2))**2))]
yerror2 = [(real_other_data1[0] * math.sqrt((10/100)**2+(math.sqrt(2) * (0.001/3))**2)),(real_other_data1[1] * math.sqrt((10/110)**2+(math.sqrt(2) * (0.001/3))**2)),(real_other_data1[2] * math.sqrt((10/120)**2+(math.sqrt(2) * (0.001/3))**2)),(real_other_data1[3] * math.sqrt((10/130)**2+(math.sqrt(2) * (0.001/3))**2)),(real_other_data1[4] * math.sqrt((10/140)**2+(math.sqrt(2) * (0.001/3))**2))]
yerror3 = [(real_other_data2[0] * math.sqrt((10/100)**2+(math.sqrt(2) * (0.001/4))**2)),(real_other_data2[1] * math.sqrt((10/110)**2+(math.sqrt(2) * (0.001/4))**2)),(real_other_data2[2] * math.sqrt((10/120)**2+(math.sqrt(2) * (0.001/4))**2)),(real_other_data2[3] * math.sqrt((10/130)**2+(math.sqrt(2) * (0.001/4))**2)),(real_other_data2[4] * math.sqrt((10/140)**2+(math.sqrt(2) * (0.001/4))**2))]
yerror4 = [(real_other_data3[0] * math.sqrt((10/100)**2+(math.sqrt(2) * (0.001/5))**2)),(real_other_data3[1] * math.sqrt((10/110)**2+(math.sqrt(2) * (0.001/5))**2)),(real_other_data3[2] * math.sqrt((10/120)**2+(math.sqrt(2) * (0.001/5))**2)),(real_other_data3[3] * math.sqrt((10/130)**2+(math.sqrt(2) * (0.001/5))**2)),(real_other_data3[4] * math.sqrt((10/140)**2+(math.sqrt(2) * (0.001/5))**2))]
b2_std,b2_std1,b2_std2,b2_std3,b2_std4 = [],[],[],[],[]
def stddevB2(b2,I,rc,sigmaI=0.1,sigmarc=0.0001):
    return math.sqrt(2)*(b2)*math.sqrt((sigmaI/I)**2 + (sigmarc/rc)**2)
for t in range(4):
    b2_std.append(stddevB2(b2_100v[t],data_100v[t],radius[t]))
    b2_std1.append(stddevB2(b2_110v[t],data_110v[t],radius[t]))
    b2_std2.append(stddevB2(b2_120v[t],data_120v[t],radius[t]))
    b2_std3.append(stddevB2(b2_130v[t],data_130v[t],radius[t]))
    b2_std4.append(stddevB2(b2_140v[t],data_140v[t],radius[t]))
def deviation(V,r,sigmaV=10,sigmar=0.001):
    r = r/100
    return ((2*V)/r) * math.sqrt((sigmaV/V)**2 + (sigmar/r)**2)
dev = []
for i in range(100,150,10):
    for t in range(2,6):
       dev.append(deviation(i,t)) 
slope,intercept,rvalue,pvalue,stderr=linregress(real_other_data,real_b2_data)
fit=np.polyfit(real_other_data,real_b2_data,1)
bfl=np.poly1d(fit)
fig = mt.figure()
ax = fig.add_subplot(111)
mt.plot(real_other_data,bfl(real_other_data),color="blue")
mt.scatter(real_other_data,real_b2_data,color="black")
mt.errorbar(real_other_data, real_b2_data,xerror1,yerror1, linestyle="None",color="red")
slope1,intercept1,rvalue1,pvalue1,stderr1=linregress(real_other_data1,real_b2_data1)
fit1=np.polyfit(real_other_data1,real_b2_data1,1)
bfl1=np.poly1d(fit1)
#mt.plot(real_other_data1,bfl1(real_other_data1),color="red")
#mt.scatter(real_other_data1,real_b2_data1,color="blue")
#mt.errorbar(real_other_data1, real_b2_data1,xerror2,yerror2, linestyle="None",color="red")
slope2,intercept2,rvalue2,pvalue2,stderr2=linregress(real_other_data2,real_b2_data2)
fit2=np.polyfit(real_other_data2,real_b2_data2,1)
bfl2=np.poly1d(fit2)
#mt.plot(real_other_data2,bfl2(real_other_data2),color="black")
#mt.scatter(real_other_data2,real_b2_data2,color="purple")
#mt.errorbar(real_other_data2, real_b2_data2,xerror3,yerror3, linestyle="None",color="red")
slope3,intercept3,rvalue3,pvalue3,stderr3=linregress(real_other_data3,real_b2_data3)
fit3=np.polyfit(real_other_data3,real_b2_data3,1)
bfl3=np.poly1d(fit3)
#mt.plot(real_other_data3,bfl3(real_other_data3),color="blue")
#mt.scatter(real_other_data3,real_b2_data3,color="orange")
#mt.errorbar(real_other_data3, real_b2_data3,xerror4,yerror4, linestyle="None",color="red")
mt.title("B^2 VS 2V / r^2")
mt.xlabel(" 2V / r^2")
mt.ylabel("B^2")
mt.grid()
mt.show()
qm1 = np.mean(np.array(real_other_data) / np.array(real_b2_data))
stqm1 = qm1 * math.sqrt((st.mean(xerror1)/st.mean(real_b2_data))**2 + (st.mean(yerror1)/st.mean(real_other_data))**2 )
