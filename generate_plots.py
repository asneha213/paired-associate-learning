import numpy as np
from matplotlib import pyplot as plt
import pickle
import pdb
import math

## Accuracies 
f = open('accuracies_data.pkl', 'rb')
data = pickle.load(f)
data = np.array(data)

data[:,1] = 1 - data[:,1]
data[:,3] = 1 - data[:,3]

f = open('time_diff_data.pkl', 'rb')
data_times = pickle.load(f)

f1 = open('no_test_stats.pkl', 'rb')
notest = pickle.load(f1)
notest_data = []
notest_times = []
notest_gsq = []
notest_params = []
pdb.set_trace()
for i in range(15):
	if len(notest[i]):
		notest_data.append(notest[i][2])
		notest_times.append(notest[i][1])
		notest_gsq.append(notest[i][0])
		notest_params.append(notest[i][3])
notest_data = np.array(notest_data)
notest_data[:,1] = 1 - notest_data[:,1]
notest_data[:,3] = 1 - notest_data[:,3]
notest_times = np.array(notest_times)
notest_params = np.array(notest_params)

f1 = open('stochastic_learning_stats.pkl', 'rb')
uncorrtest = pickle.load(f1)
uncorrtest_data = []
uncorrtest_times = []
uncorrtest_gsq = []
uncorrtest_params = []
for i in range(15):
	if len(uncorrtest[i]):
		uncorrtest_data.append(uncorrtest[i][2])
		uncorrtest_times.append(uncorrtest[i][1])
		uncorrtest_gsq.append(uncorrtest[i][0])
		uncorrtest_params.append(uncorrtest[i][3])
uncorrtest_data = np.array(uncorrtest_data)
uncorrtest_data[:,1] = 1 - uncorrtest_data[:,1]
uncorrtest_data[:,3] = 1 - uncorrtest_data[:,3]
uncorrtest_times = np.array(uncorrtest_times)
uncorrtest_params = np.array(uncorrtest_params)

f1 = open('all_weights_learning_stats.pkl', 'rb')
corrtest = pickle.load(f1)
corrtest_data = []
corrtest_times = []
corrtest_gsq = []
corrtest_params = []
for i in range(15):
	if len(corrtest[i]):
		corrtest_data.append(corrtest[i][2])
		corrtest_times.append(corrtest[i][1])
		corrtest_gsq.append(corrtest[i][0])
		corrtest_params.append(corrtest[i][3])
corrtest_data = np.array(corrtest_data)
corrtest_data[:,1] = 1 - corrtest_data[:,1]
corrtest_data[:,3] = 1 - corrtest_data[:,3]
corrtest_times = np.array(corrtest_times)
corrtest_params = np.array(corrtest_params)

def get_t_value(times):
	'''
	sum_diff = np.sum(times[:,0] - times[:,1])
	sum_sq_diff = np.sum((times[:,0] - times[:,1])**2)
	num = sum_diff/15
	den = (sum_sq_diff - (sum_diff**2)/15)/(14*15)
	den = math.sqrt(den)
	'''
	mean_diff = np.mean(times[:,0] - times[:,1])
	std_diff = np.std(times[:,0] - times[:,1])/math.sqrt(15)
	tvalue = mean_diff/std_diff
	return tvalue

print(get_t_value(notest_times))
print(get_t_value(uncorrtest_times))
print(get_t_value(corrtest_times))


## Plot accuracies
plt.style.use('grayscale')
xind = np.arange(5)
width = 0.2
plt.bar(xind, np.mean(data,axis=0), yerr=np.std(data,axis=0)/math.sqrt(15), width=width, alpha=1, label='Data')
plt.bar(xind+width, np.mean(notest_data,axis=0), yerr=np.std(notest_data,axis=0)/math.sqrt(15), width=width, alpha=0.7, edgecolor='black', label='Model (No test)')
plt.bar(xind+2*width, np.mean(uncorrtest_data,axis=0), yerr=np.std(uncorrtest_data,axis=0)/math.sqrt(15), width=width, alpha= 0.4, edgecolor='black', label='Model (Uncorr test)')
plt.bar(xind+3*width, np.mean(corrtest_data,axis=0), yerr=np.std(corrtest_data,axis=0)/math.sqrt(15), width=width, alpha = 0.1, edgecolor='black', label='Model (Corr test)')
plt.xticks(xind + width, ('T1', 'T2_ID_COR', 'T2_ID_INC', 'T2_RV_COR', 'I2_RV_INC'))
plt.ylabel('p(correct)')
plt.legend()
plt.show()


plt.style.use('grayscale')
fig, (ax1, ax2) = plt.subplots(1, 2,  gridspec_kw={'width_ratios': [1, 2]})
width = 0.1
xind = np.arange(1)
identical_time = [np.mean(notest_times, axis=0)[0], np.mean(uncorrtest_times, axis=0)[0], np.mean(corrtest_times, axis=0)[0]]

identical_time_err = [np.std(notest_times, axis=0)[0]/math.sqrt(15), np.std(uncorrtest_times, axis=0)[0]/math.sqrt(15), np.std(corrtest_times, axis=0)[0]/math.sqrt(15)]
reverse_time = [np.mean(notest_times, axis=0)[1], np.mean(uncorrtest_times, axis=0)[1], np.mean(corrtest_times, axis=0)[1]]
reverse_time_err = [np.std(notest_times, axis=0)[1]/math.sqrt(15), np.std(uncorrtest_times, axis=0)[1]/math.sqrt(15), np.std(corrtest_times, axis=0)[1]/math.sqrt(15)]

ax1.bar(xind, [data_times[0]] , yerr=[data_times[2]/math.sqrt(15)], width=width, label='Identical')
ax1.bar(xind+width, [data_times[1]], yerr=[data_times[3]/math.sqrt(15)], width=width, label='Reverse')
ax1.set_xticks([0,0.1])
ax1.set_xticklabels(['ID', 'RV'])
#ax1.set_title('Reaction times: Data')
ax1.set_ylabel('time difference in milliseconds')
ax1.set_xlabel('Data')

xind = np.arange(3)
width = 0.2
ax2.bar(xind, identical_time, yerr=identical_time_err, width=width, label='Identical')
ax2.bar(xind+width, reverse_time, yerr=reverse_time_err, width=width, label='Reverse')
ax2.set_xticks([0,1,2])
ax2.set_xticklabels(['No-test', 'Uncorr-test', 'Corr-test'])
#ax2.set_title('Reaction times: Model')
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('time difference in simulated iterations')
ax2.set_xlabel('Models')
ax2.legend()
plt.show()


plt.style.use('grayscale')
fig, (ax1, ax2) = plt.subplots(1, 2)
width = 0.2
xind = np.arange(1)
gsq1 = [np.mean(notest_gsq)]; gsq2 = [np.mean(uncorrtest_gsq)]; gsq3 = [np.mean(corrtest_gsq)]
gsq_err1 = [np.std(notest_gsq)/math.sqrt(15)]; gsq_err2 = [np.std(uncorrtest_gsq)/math.sqrt(15)]; gsq_err3 = [np.std(corrtest_gsq)/math.sqrt(15)]
rho1 = [np.mean(notest_params, axis=0)[2]]; rho2 = [np.mean(uncorrtest_params, axis=0)[2]]; rho3 =[np.mean(corrtest_params, axis=0)[2]]
rho_err1 = [np.std(notest_params, axis=0)[2]/math.sqrt(15)]; rho_err2 = [np.std(uncorrtest_params, axis=0)[2]/math.sqrt(15)]; rho_err3 = [np.std(corrtest_params, axis=0)[2]/math.sqrt(15)]
ax1.bar(xind, gsq1, yerr=gsq_err1, width=width, label='M1: No test')
ax1.bar(xind+width, gsq2, yerr=gsq_err2, width=width, label='M2: Uncorr test')
ax1.bar(xind+2*width, gsq3, yerr=gsq_err3, width=width, label='M3: Corr test')
ax1.set_xticks([0,0.2,0.4])
ax1.set_xticklabels(['No-test', 'Uncorr-test', 'Corr-test'])
ax1.set_ylabel('Goodness-of-fit (G-squared)')
ax2.bar(xind, rho1, yerr=rho_err1, width=width, label='M1: No test')
ax2.bar(xind+width, rho2, yerr=rho_err2, width=width, label='M2: Uncorr test')
ax2.bar(xind+2*width, rho3, yerr=rho_err3, width=width, label='M3: Corr test')
ax2.set_xticks([0,0.2,0.4])
ax2.set_xticklabels(['No-test', 'Uncorr-test', 'Corr-test'])
ax2.yaxis.set_label_position("right")
ax2.set_ylabel(r'Correlation ($\rho$)')
#ax2.legend()
plt.show()


