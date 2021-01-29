import pandas as pd
import numpy as np


import matplotlib.pyplot as plt



#plotting the results for 6 features


#Precision plotting
x= [0,1,2,3,4,5,6]
values = ['DDoS', 'DoS', 'Probe', 'BOTNET','WEB-ATTACK','BFA','U2R'] 
DDoS = [0.9968109648527615, 0.9985709424974482, 0.9994531410212592]
DoS = [ 0.9083764792899408, 0.8782790233275847,0.8067931905188565 ]
Probe = [0.9260551598871947,0.9282560706401766 ,0.9437337646424545 ]
Botnet = [0.0,0.0 ,0.0 ]
WebAttack = [0.0, 0.0, 0.0]
BFA = [0.0,0.8823529411764706 ,0.0 ]
U2R = [0.0, 0.0, 0.0 ]
attacks = [DDoS,DoS,Probe,Botnet,WebAttack,BFA,U2R]
df2 = pd.DataFrame(attacks, columns=['LSTM-RNN', 'SIMPLE-RNN', 'GRU-RNN'])
# df2.plot.bar(legend=None)
df2.plot.bar()
plt.legend( title='Algorithm', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.title('Precision Score For Deep Learning models with 6 features',fontsize=20,pad=15)
plt.ylabel('Precision',fontsize=20,labelpad=15)
plt.xlabel('Attack Types',fontsize=20,labelpad=15)

plt.xticks(x,values,rotation='horizontal')
plt.yticks(np.arange(0, 1, 0.05))

plt.show()












#Recall plotting


x= [0,1,2,3,4,5,6]
values = ['DDoS', 'DoS', 'Probe', 'BOTNET','WEB-ATTACK','BFA','U2R'] 
DDoS = [0.9989800081599347,  0.9978240174078608, 0.9942200462396301]
DoS = [0.9161693397985826,0.9022752704214845 ,0.9236292428198434 ]
Probe = [0.9871598899419138,0.9855803525934984 , 0.9810965046367064]
Botnet = [0.0,0.0 ,0.0 ]
WebAttack = [0.0,0.0 ,0.0 ]
BFA = [0.0, 0.05338078291814947, 0.0]
U2R = [0.0, 0.0, 0.0]
attacks = [DDoS,DoS,Probe,Botnet,WebAttack,BFA,U2R]
df2 = pd.DataFrame(attacks, columns=['LSTM-RNN', 'SIMPLE-RNN', 'GRU-RNN'])
# df2.plot.bar(legend=None)
df2.plot.bar()
plt.legend( title='Algorithm', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.title('Recall Score For Deep Learning models with 6 features',fontsize=20,pad=15)
plt.ylabel('Recall',fontsize=20,labelpad=15)
plt.xlabel('Attack Types',fontsize=20,labelpad=15)

plt.xticks(x,values,rotation='horizontal')
plt.yticks(np.arange(0, 1, 0.05))

plt.show()







#F1 Score plotting

x= [0,1,2,3,4,5,6]
values = ['DDoS', 'DoS', 'Probe', 'BOTNET','WEB-ATTACK','BFA','U2R'] 
DDoS = [0.9978943078386089,0.998197340226523 ,  0.9968297255837737]
DoS = [0.9122562674094707, 0.8901154500712939, 0.8612669014390679]
Probe = [0.9556317360100623,0.9560597073942269 , 0.9620525119288517]
Botnet = [0.0,0.0 , 0.0 ]
WebAttack = [0.0,0.0 , 0.0]
BFA = [0.0,0.10067114093959731 , 0.0]
U2R = [0.0, 0.0, 0.0]
attacks = [DDoS,DoS,Probe,Botnet,WebAttack,BFA,U2R]
df2 = pd.DataFrame(attacks, columns=['LSTM-RNN', 'SIMPLE-RNN', 'GRU-RNN'])
# df2.plot.bar(legend=None)
df2.plot.bar()
plt.legend( title='Algorithm', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.title('F1-Score For Deep Learning models with 6 features',fontsize=20,pad=15)
plt.ylabel('F1-Score',fontsize=20,labelpad=15)
plt.xlabel('Attack Types',fontsize=20,labelpad=15)

plt.xticks(x,values,rotation='horizontal')
plt.yticks(np.arange(0, 1, 0.05))

plt.show()














#Training time

x= [0,1,2,3,4,5,6]
values = ['DDoS', 'DoS', 'Probe', 'BOTNET','WEB-ATTACK','BFA','U2R'] 
DDoS = [282.36078667640686, 185.49154448509216,368.76533699035645 ]
DoS = [252.50738143920898, 132.8528151512146, 317.47396087646484]
Probe = [330.63078236579895,190.04812264442444 , 389.2078216075897]
Botnet = [136.8380331993103,77.90870451927185 , 170.26627039909363 ]
WebAttack = [136.44860792160034, 79.67743825912476 ,167.5186002254486 ]
BFA = [137.36666703224182, 85.66810393333435 , 172.2692461013794]
U2R = [117.38625311851501, 65.12616443634033, 179.1226851940155]
attacks = [DDoS,DoS,Probe,Botnet,WebAttack,BFA,U2R]
df2 = pd.DataFrame(attacks, columns=['LSTM-RNN', 'SIMPLE-RNN','GRU-RNN'])
# df2.plot.bar(legend=None)
df2.plot.bar()
plt.legend( title='Algorithm', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.title('Training time For Deep Learning models with 6 features',fontsize=20,pad=15)
plt.ylabel('Time(s)',fontsize=20,labelpad=15)
plt.xlabel('Attack Types',fontsize=20,labelpad=15)

plt.xticks(x,values,rotation='horizontal')
# plt.yticks(np.arange(0, 1530))

plt.show()

























#Accuracy Score 

x= [0,1,2,3,4,5,6]
values = ['DDoS', 'DoS', 'Probe', 'BOTNET','WEB-ATTACK','BFA','U2R'] 
DDoS = [0.9978162093621218, 0.9981331825256348, 0.9967243075370789 ]
DoS = [0.9225695729255676,0.9021262526512146 ,0.8692695498466492 ]
Probe = [0.9459938406944275,0.946624219417572 , 0.9543994665145874]
Botnet = [0.99759441614151,0.99759441614151,0.997594401516256]
WebAttack = [0.9971582889556885,0.9971582889556885,0.9971582889556885 ]
BFA = [0.9798797369003296, 0.9808105230331421,  0.9798797369003296]
U2R = [0.999707818031311,0.999707818031311 ,0.999707818031311 ]
attacks = [DDoS,DoS,Probe,Botnet,WebAttack,BFA,U2R]
df2 = pd.DataFrame(attacks, columns=['LSTM-RNN', 'SIMPLE-RNN','GRU-RNN'])
# df2.plot.bar(legend=None)
df2.plot.bar()
plt.legend( title='Algorithm', bbox_to_anchor=(1.01, 1), loc='upper left')

plt.title('Accuracy Score For Deep Learning models with 6 features',fontsize=20,pad=15)
plt.ylabel('Accuracy',fontsize=20,labelpad=15)
plt.xlabel('Attack Types',fontsize=20,labelpad=15)

plt.xticks(x,values,rotation='horizontal')
plt.yticks(np.arange(0, 1, 0.05))

plt.show()



























# #Dataset distribution 

# x= [0,1,2,3,4,5,6,7]
# values = ['Normal','DDoS', 'DoS', 'Probe', 'BOTNET','WEB-ATTACK','BFA','U2R'] 
# Normal = [68424,]
# DDoS = [73529,]
# DoS = [53616,]
# Probe = [98129,]
# Botnet = [164,]
# WebAttack = [192,]
# BFA = [1405,]
# U2R = [17,]
# attacks = [Normal,DDoS,DoS,Probe,Botnet,WebAttack,BFA,U2R]
# df2 = pd.DataFrame(attacks, columns=['Number of Samples',])
# df2.plot.bar(legend=None)
# # df2.plot.bar()
# # plt.legend( title='Algorithm', bbox_to_anchor=(1.01, 1), loc='upper left')

# plt.title('Dataset Distribution',fontsize=20,pad=15)
# plt.ylabel('Number of Samples',fontsize=20,labelpad=15)
# plt.xlabel('Samples Type',fontsize=20,labelpad=15)

# plt.xticks(x,values,rotation='horizontal')
# # plt.yticks(np.arange(0, 1, 0.05))

# plt.show()


















