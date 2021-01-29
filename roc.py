
from sklearn.metrics import  auc, roc_curve



import matplotlib.pyplot as plt


plt.rcParams['font.size'] = '20'
#Line styles for different attacks
linestyle_tuple = [
    ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),
     ('loosely dotted',        (0, (1, 10))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]










#plot for LSTM 48 Features




r_auc_ddos = 0.9998300013599891
false_pos_rate_ddos =  [0., 0., 1.]
true_pos_rate_ddos = [0., 0.99966 ,1.]


r_auc_dos = 0.9925529037815602
false_pos_rate_dos = [0., 0.00211911, 1.]
true_pos_rate_dos = [0.,0.98722492, 1.]

r_auc_probe = 0.9803574728310901
false_pos_rate_probe = [0.,0.03770552, 1.]
true_pos_rate_probe = [0.,0.99842046, 1.]

r_auc_botnet = 0.999598100109609
false_pos_rate_botnet = [0.00000000e+00, 8.03799781e-04, 1.00000000e+00]
true_pos_rate_botnet = [0., 1., 1.]

r_auc_u2r = 0.5
false_pos_rate_u2r = [0., 1.]
true_pos_rate_u2r = [0., 1.]

r_auc_webattack = 0.7307692307692308
false_pos_rate_webattack = [0., 0., 1.]
true_pos_rate_webattack = [0.,0.46153846, 1.]

r_auc_bfa = 0.6369376034492399
false_pos_rate_bfa = [0.00000000e+00, 1.46145415e-04, 1.00000000e+00]
true_pos_rate_bfa = [0.,0.27402135, 1.]

plt.plot(false_pos_rate_ddos,true_pos_rate_ddos, linestyle=linestyle_tuple[0][1], label='DDoS prediction (AUROC = %0.3f)' % r_auc_ddos,linewidth=7.0)
plt.plot(false_pos_rate_dos,true_pos_rate_dos, linestyle=linestyle_tuple[1][1], label='DoS prediction(AUROC = %0.3f)' % r_auc_dos,linewidth=7.0)
plt.plot(false_pos_rate_probe,true_pos_rate_probe, linestyle=linestyle_tuple[2][1], label='Probe prediction (AUROC = %0.3f)' % r_auc_probe,linewidth=7.0)
plt.plot(false_pos_rate_botnet,true_pos_rate_botnet, linestyle=linestyle_tuple[3][1], label='BOTNET prediction (AUROC = %0.3f)' % r_auc_botnet,linewidth=7.0)
plt.plot(false_pos_rate_webattack,true_pos_rate_webattack, linestyle=linestyle_tuple[4][1], label='Web-Attack prediction (AUROC = %0.3f)' % r_auc_webattack,linewidth=7.0)
plt.plot(false_pos_rate_bfa,true_pos_rate_bfa, linestyle=linestyle_tuple[5][1], label='BFA prediction (AUROC = %0.3f)' % r_auc_bfa,linewidth=7.0)
plt.plot(false_pos_rate_u2r,true_pos_rate_u2r, linestyle=linestyle_tuple[6][1], label='U2R prediction (AUROC = %0.3f)' % r_auc_u2r,linewidth=7.0)



# Title
#plt.title('LSTM 48 Features ROC Plot', fontsize=20)
# Axis labels
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# Show legend
plt.legend(prop={"size":20}) # 

# Show plot
plt.show()






#plot for RNN 48 Features

r_auc_ddos = 0.9998980008159934
false_pos_rate_ddos =  [0., 0., 1.]
true_pos_rate_ddos = [0., 0.999796, 1.]


r_auc_dos = 0.9887419379149095
false_pos_rate_dos = [0., 0.01132627, 1.]
true_pos_rate_dos = [0.,0.98881015 ,1.]



r_auc_probe = 0.977838830559672
false_pos_rate_probe = [0., 0.04340519, 1.]
true_pos_rate_probe = [0., 0.99908285, 1.]

r_auc_botnet = 0.9999269272926562
false_pos_rate_botnet = [0.00000000e+00, 1.46145415e-04, 1.00000000e+00]
true_pos_rate_botnet = [0., 1., 1.]

r_auc_u2r = 0.5
false_pos_rate_u2r = [0., 1.]
true_pos_rate_u2r = [0., 1.]

r_auc_webattack = 0.7307692307692308
false_pos_rate_webattack = [0., 0., 1.]
true_pos_rate_webattack = [0., 0.46153846, 1.]

r_auc_bfa = 0.8557988134136526
false_pos_rate_bfa = [0.00000000e+00, 1.46145415e-04, 1.00000000e+00]
true_pos_rate_bfa = [0., 0.71174377, 1.]


plt.plot(false_pos_rate_ddos,true_pos_rate_ddos, linestyle=linestyle_tuple[0][1], label='DDoS prediction (AUROC = %0.3f)' % r_auc_ddos,linewidth=7.0)
plt.plot(false_pos_rate_dos,true_pos_rate_dos, linestyle=linestyle_tuple[1][1], label='DoS prediction(AUROC = %0.3f)' % r_auc_dos,linewidth=7.0)
plt.plot(false_pos_rate_probe,true_pos_rate_probe, linestyle=linestyle_tuple[2][1], label='Probe prediction (AUROC = %0.3f)' % r_auc_probe,linewidth=7.0)
plt.plot(false_pos_rate_botnet,true_pos_rate_botnet, linestyle=linestyle_tuple[3][1], label='BOTNET prediction (AUROC = %0.3f)' % r_auc_botnet,linewidth=7.0)
plt.plot(false_pos_rate_webattack,true_pos_rate_webattack, linestyle=linestyle_tuple[4][1], label='Web-Attack prediction (AUROC = %0.3f)' % r_auc_webattack,linewidth=7.0)
plt.plot(false_pos_rate_bfa,true_pos_rate_bfa, linestyle=linestyle_tuple[5][1], label='BFA prediction (AUROC = %0.3f)' % r_auc_bfa,linewidth=7.0)
plt.plot(false_pos_rate_u2r,true_pos_rate_u2r, linestyle=linestyle_tuple[6][1], label='U2R prediction (AUROC= %0.3f)' % r_auc_u2r,linewidth=7.0)



# Title
#plt.title('RNN 48 Features ROC Plot', fontsize=20)
# Axis labels
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# Show legend
plt.legend(prop={"size":20}) # 

# Show plot
plt.show()
















#plot for GRU 48 Features

r_auc_ddos =  0.9994595645719306
false_pos_rate_ddos =  [0.00000000e+00, 8.76872488e-04, 1.00000000e+00]
true_pos_rate_ddos = [0.,0.999796, 1.]


r_auc_dos = 0.9885282050156877
false_pos_rate_dos =  [0., 0.00336134, 1.]
true_pos_rate_dos = [0., 0.98041775, 1.]



r_auc_probe = 0.9790836523888194
false_pos_rate_probe = [0.,0.03770552, 1.]
true_pos_rate_probe = [0., 0.99587282, 1.]

r_auc_botnet = 0.5
false_pos_rate_botnet = [0., 1.]
true_pos_rate_botnet = [0., 1.]

r_auc_u2r = 0.5
false_pos_rate_u2r = [0., 1.]
true_pos_rate_u2r = [0., 1.]

r_auc_webattack = 0.7307326944155589
false_pos_rate_webattack = [0.00000000e+00, 7.30727073e-05, 1.00000000e+00]
true_pos_rate_webattack = [0., 0.46153846, 1.]

r_auc_bfa = 0.8539463812757039
false_pos_rate_bfa = [0.00000000e+00, 2.92290829e-04, 1.00000000e+00]
true_pos_rate_bfa = [0.,0.70818505, 1.]


plt.plot(false_pos_rate_ddos,true_pos_rate_ddos, linestyle=linestyle_tuple[0][1], label='DDoS prediction (AUROC = %0.3f)' % r_auc_ddos,linewidth=7.0)
plt.plot(false_pos_rate_dos,true_pos_rate_dos, linestyle=linestyle_tuple[1][1], label='DoS prediction(AUROC = %0.3f)' % r_auc_dos,linewidth=7.0)
plt.plot(false_pos_rate_probe,true_pos_rate_probe, linestyle=linestyle_tuple[2][1], label='Probe prediction (AUROC = %0.3f)' % r_auc_probe,linewidth=7.0)
plt.plot(false_pos_rate_botnet,true_pos_rate_botnet, linestyle=linestyle_tuple[3][1], label='BOTNET prediction (AUROC = %0.3f)' % r_auc_botnet,linewidth=7.0)
plt.plot(false_pos_rate_webattack,true_pos_rate_webattack, linestyle=linestyle_tuple[4][1], label='Web-Attack prediction (AUROC = %0.3f)' % r_auc_webattack,linewidth=7.0)
plt.plot(false_pos_rate_bfa,true_pos_rate_bfa, linestyle=linestyle_tuple[5][1], label='BFA prediction (AUROC = %0.3f)' % r_auc_bfa,linewidth=7.0)
plt.plot(false_pos_rate_u2r,true_pos_rate_u2r, linestyle=linestyle_tuple[6][1], label='U2R prediction (AUROC= %0.3f)' % r_auc_u2r,linewidth=7.0)



# Title
#plt.title('GRU 48 Features ROC Plot', fontsize=20)
# Axis labels
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# Show legend
plt.legend(prop={"size":20}) # 

# Show plot
plt.show()









#plot for LSTM 6 Features




r_auc_ddos = 0.998236597958148
false_pos_rate_ddos =   [0.,0.00182682, 1.]
true_pos_rate_ddos = [0., 0.99830001, 1.]


r_auc_dos = 0.905604419086286
false_pos_rate_dos = [0., 0.14198027, 1.]
true_pos_rate_dos = [0., 0.95318911, 1.]

r_auc_probe = 0.9385595528602626
false_pos_rate_probe = [0., 0.11019364, 1.]
true_pos_rate_probe = [0., 0.98731275, 1.]

r_auc_botnet = 0.5
false_pos_rate_botnet = [0., 1.]
true_pos_rate_botnet = [0., 1.]

r_auc_u2r = 0.5
false_pos_rate_u2r = [0., 1.]
true_pos_rate_u2r = [0., 1.]

r_auc_webattack = 0.5
false_pos_rate_webattack = [0., 1.]
true_pos_rate_webattack = [0., 1.]

r_auc_bfa = 0.5
false_pos_rate_bfa = [0., 1.]
true_pos_rate_bfa = [0., 1.]


plt.plot(false_pos_rate_ddos,true_pos_rate_ddos, linestyle=linestyle_tuple[0][1], label='DDoS prediction (AUROC = %0.3f)' % r_auc_ddos,linewidth=7.0)
plt.plot(false_pos_rate_dos,true_pos_rate_dos, linestyle=linestyle_tuple[1][1], label='DoS prediction(AUROC = %0.3f)' % r_auc_dos,linewidth=7.0)
plt.plot(false_pos_rate_probe,true_pos_rate_probe, linestyle=linestyle_tuple[2][1], label='Probe prediction (AUROC = %0.3f)' % r_auc_probe,linewidth=7.0)
plt.plot(false_pos_rate_botnet,true_pos_rate_botnet, linestyle=linestyle_tuple[3][1], label='BOTNET prediction (AUROC = %0.3f)' % r_auc_botnet,linewidth=7.0)
plt.plot(false_pos_rate_webattack,true_pos_rate_webattack, linestyle=linestyle_tuple[4][1], label='Web-Attack prediction (AUROC = %0.3f)' % r_auc_webattack,linewidth=7.0)
plt.plot(false_pos_rate_bfa,true_pos_rate_bfa, linestyle=linestyle_tuple[5][1], label='BFA prediction (AUROC = %0.3f)' % r_auc_bfa,linewidth=7.0)
plt.plot(false_pos_rate_u2r,true_pos_rate_u2r, linestyle=linestyle_tuple[6][1], label='U2R prediction (AUROC= %0.3f)' % r_auc_u2r,linewidth=7.0)



# Title
#plt.title('LSTM 6 Features ROC Plot', fontsize=20)
# Axis labels
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# Show legend
plt.legend(prop={"size":20}) # 

# Show plot
plt.show()









#plot for RNN 6 Features




r_auc_ddos = 0.9979194303091539
false_pos_rate_ddos =   [0., 0.00116916, 1.]
true_pos_rate_ddos = [0., 0.99700802, 1.]


r_auc_dos = 0.922758189437655
false_pos_rate_dos = [0., 0.0487395, 1.]
true_pos_rate_dos = [0., 0.89425587, 1.]

r_auc_probe = 0.9199327255259766
false_pos_rate_probe = [0., 0.14775301, 1.]
true_pos_rate_probe = [0., 0.98761847, 1.]

r_auc_botnet = 0.5
false_pos_rate_botnet = [0., 1.]
true_pos_rate_botnet = [0., 1.]

r_auc_u2r = 0.5
false_pos_rate_u2r = [0., 1.]
true_pos_rate_u2r = [0., 1.]

r_auc_webattack = 0.5
false_pos_rate_webattack = [0., 1.]
true_pos_rate_webattack = [0., 1.]

r_auc_bfa = 0.5
false_pos_rate_bfa = [0., 1.]
true_pos_rate_bfa = [0., 1.]


plt.plot(false_pos_rate_ddos,true_pos_rate_ddos, linestyle=linestyle_tuple[0][1], label='DDoS prediction (AUROC = %0.3f)' % r_auc_ddos,linewidth=7.0)
plt.plot(false_pos_rate_dos,true_pos_rate_dos, linestyle=linestyle_tuple[1][1], label='DoS prediction(AUROC = %0.3f)' % r_auc_dos,linewidth=7.0)
plt.plot(false_pos_rate_probe,true_pos_rate_probe, linestyle=linestyle_tuple[2][1], label='Probe prediction (AUROC = %0.3f)' % r_auc_probe,linewidth=7.0)
plt.plot(false_pos_rate_botnet,true_pos_rate_botnet, linestyle=linestyle_tuple[3][1], label='BOTNET prediction (AUROC = %0.3f)' % r_auc_botnet,linewidth=7.0)
plt.plot(false_pos_rate_webattack,true_pos_rate_webattack, linestyle=linestyle_tuple[4][1], label='Web-Attack prediction (AUROC = %0.3f)' % r_auc_webattack,linewidth=7.0)
plt.plot(false_pos_rate_bfa,true_pos_rate_bfa, linestyle=linestyle_tuple[5][1], label='BFA prediction (AUROC = %0.3f)' % r_auc_bfa,linewidth=7.0)
plt.plot(false_pos_rate_u2r,true_pos_rate_u2r, linestyle=linestyle_tuple[6][1], label='U2R prediction (AUROC= %0.3f)' % r_auc_u2r,linewidth=7.0)


# Title
#plt.title('RNN 6 Features ROC Plot', fontsize=20)
# Axis labels
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# Show legend
plt.legend(prop={"size":20}) # 

# Show plot
plt.show()

















#plot for GRU 6 Features




r_auc_ddos = 0.9968177322904399
false_pos_rate_ddos =   [0.00000000e+00, 5.84581659e-04, 1.00000000e+00]
true_pos_rate_ddos = [0., 0.99422005, 1.]


r_auc_dos = 0.8662654265929325
false_pos_rate_dos = [0., 0.2136646, 1.]
true_pos_rate_dos = [0., 0.94619545, 1.]

r_auc_probe = 0.9364513386070337
false_pos_rate_probe = [0., 0.10880526, 1.]
true_pos_rate_probe = [0., 0.98170794, 1.]

r_auc_botnet = 0.5
false_pos_rate_botnet = [0., 1.]
true_pos_rate_botnet = [0., 1.]

r_auc_u2r = 0.5
false_pos_rate_u2r = [0., 1.]
true_pos_rate_u2r = [0., 1.]

r_auc_webattack = 0.5
false_pos_rate_webattack = [0., 1.]
true_pos_rate_webattack = [0., 1.]

r_auc_bfa = 0.5
false_pos_rate_bfa = [0., 1.]
true_pos_rate_bfa = [0., 1.]

plt.plot(false_pos_rate_ddos,true_pos_rate_ddos, linestyle=linestyle_tuple[0][1], label='DDoS prediction (AUROC = %0.3f)' % r_auc_ddos,linewidth=7.0)
plt.plot(false_pos_rate_dos,true_pos_rate_dos, linestyle=linestyle_tuple[1][1], label='DoS prediction(AUROC = %0.3f)' % r_auc_dos,linewidth=7.0)
plt.plot(false_pos_rate_probe,true_pos_rate_probe, linestyle=linestyle_tuple[2][1], label='Probe prediction (AUROC = %0.3f)' % r_auc_probe,linewidth=7.0)
plt.plot(false_pos_rate_botnet,true_pos_rate_botnet, linestyle=linestyle_tuple[3][1], label='BOTNET prediction (AUROC = %0.3f)' % r_auc_botnet,linewidth=7.0)
plt.plot(false_pos_rate_webattack,true_pos_rate_webattack, linestyle=linestyle_tuple[4][1], label='Web-Attack prediction (AUROC = %0.3f)' % r_auc_webattack,linewidth=7.0)
plt.plot(false_pos_rate_bfa,true_pos_rate_bfa, linestyle=linestyle_tuple[5][1], label='BFA prediction (AUROC = %0.3f)' % r_auc_bfa,linewidth=7.0)
plt.plot(false_pos_rate_u2r,true_pos_rate_u2r, linestyle=linestyle_tuple[6][1], label='U2R prediction (AUROC= %0.3f)' % r_auc_u2r,linewidth=7.0)



# Title
#plt.title('GRU 6 Features ROC Plot', fontsize=20)
# Axis labels
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
# Show legend
plt.legend(prop={"size":20}) # 

# Show plot
plt.show()


