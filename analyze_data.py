import numpy as np
np.set_printoptions(precision=4)
import pickle
import pdb
import math
import csv


def get_data_stats():
    data = pickle.load(open('data.pkl','rb'))

    stats = []
    avg_stats = []
    
    time_identical = {}
    time_reverse = {}

    for sub in data:
        time_identical[sub] = []
        time_reverse[sub] = []

    time = []
    for sub in data:
        
        total = {'1':0 , '3': 0, '5': 0}
        correct_first = {'1':0 , '3': 0, '5': 0}
        correct_first_i = {'1':0 , '3': 0, '5': 0}
        correct_first_r = {'1':0 , '3': 0, '5': 0}
        incorrect_first_i = {'1':0 , '3': 0, '5': 0}
        incorrect_first_r = {'1':0 , '3': 0, '5': 0}
        incorrect_second_correct_first_i = {'1':0 , '3': 0, '5': 0}
        incorrect_second_correct_first_r = {'1':0 , '3': 0, '5': 0}
        correct_second_incorrect_first_i = {'1':0 , '3': 0, '5': 0}
        correct_second_incorrect_first_r = {'1':0 , '3': 0, '5': 0}
        
        
        for trial in data[sub]:
            for pair in data[sub][trial]:

                datum = data[sub][trial][pair]
                reps = datum['xREPS']
                total[reps] += 1

                if datum['ANSWER1'] == 'C':
                    correct_first[reps] += 1

                if ('B1' in datum and 'B2' in datum) or ('F1' in datum and 'F2' in datum):
                    identical = 1
                else:
                    identical = 0


                if datum['ANSWER1'] == 'C' and identical:
                    correct_first_i[reps] += 1
                elif datum['ANSWER1'] == 'C' and not identical:
                    correct_first_r[reps] += 1
                elif datum['ANSWER1'] != 'C' and identical:
                    incorrect_first_i[reps] += 1
                elif datum['ANSWER1'] != 'C' and not identical:
                    incorrect_first_r[reps] += 1
                
                if datum['ANSWER1'] == 'C' and datum['ANSWER2'] != 'C':
                    if identical:
                        incorrect_second_correct_first_i[reps] += 1
                    else:
                        incorrect_second_correct_first_r[reps] += 1

                if datum['ANSWER1'] != 'C' and datum['ANSWER2'] == 'C':
                    if identical:
                        correct_second_incorrect_first_i[reps] += 1
                    else:
                        correct_second_incorrect_first_r[reps] += 1

                if datum['ANSWER1'] == 'C' and datum['ANSWER2'] == 'C':
                    if identical:
                        if 'F1' in datum:
                            if int(datum['F1']) > 6000 or int(datum['F2'])> 6000:
                                continue
                            time_identical[sub].append(int(datum['F1'])-int(datum['F2']))
                        else:
                            if int(datum['B1']) > 6000 or int(datum['B2'])> 6000:
                                continue
                            time_identical[sub].append(int(datum['B1'])-int(datum['B2']))
                    else:
                        if 'F1' in datum:
                            if int(datum['F1']) > 6000 or int(datum['B2'])> 6000:
                                continue
                            time_reverse[sub].append(int(datum['F1'])-int(datum['B2']))
                        else:
                            if int(datum['B1']) > 6000 or int(datum['F2'])> 6000:
                                continue
                            time_reverse[sub].append(int(datum['B1'])-int(datum['F2']))
        
        stats.append(list(correct_first.values()) + list(incorrect_second_correct_first_i.values()) + list(correct_second_incorrect_first_i.values()) + list(incorrect_second_correct_first_r.values()) + list(correct_second_incorrect_first_r.values()) +  list(correct_first_i.values()) + list(correct_first_r.values()) +  list(incorrect_first_i.values()) + list(incorrect_first_r.values() ))

        avg_stats.append([sum(list(correct_first.values())), \
        sum(list(incorrect_second_correct_first_i.values())), \
        sum(list(correct_second_incorrect_first_i.values())), \
        sum(list(incorrect_second_correct_first_r.values())), \
        sum(list(correct_second_incorrect_first_r.values())), \
        sum(list(correct_first_i.values())), \
        sum(list(correct_first_r.values())), \
        sum(list(incorrect_first_i.values())), \
        sum(list(incorrect_first_r.values()))])


    avg_stats = np.array(avg_stats).astype(float)
    count_stats = avg_stats.copy()

    incorrect_2_correct_1_i = avg_stats[:,1]/avg_stats[:,-4]
    correct_2_incorrect_1_i = avg_stats[:,2]/(avg_stats[:,-2])
    incorrect_2_correct_1_r = avg_stats[:,3]/avg_stats[:,-3]
    correct_2_incorrect_1_r = avg_stats[:,4]/(avg_stats[:,-1])

    avg_stats[:,1] = avg_stats[:,1]/avg_stats[:,-4]
    avg_stats[:,2] = avg_stats[:,2]/(avg_stats[:,-2])
    avg_stats[:,3] = avg_stats[:,3]/avg_stats[:,-3]
    avg_stats[:,4] = avg_stats[:,4]/(avg_stats[:,-1])
    avg_stats[:,0] = avg_stats[:,0]/72
    avg_stats[:,5] = avg_stats[:,5]/36
    avg_stats[:,6] = avg_stats[:,6]/36
    avg_stats[:,7] = avg_stats[:,7]/36
    avg_stats[:,8] = avg_stats[:,8]/36

    difference_i = incorrect_2_correct_1_i - correct_2_incorrect_1_i
    difference_r = incorrect_2_correct_1_r - correct_2_incorrect_1_r

    t_r = np.mean(difference_r)/(np.std(difference_r)/math.sqrt(15))
    t_i = np.mean(difference_i)/(np.std(difference_i)/math.sqrt(15))

    time_diff_identical = []
    time_diff_reverse = []
    for sub in data:
        time_diff_identical.append(np.mean(time_identical[sub]))
        time_diff_reverse.append(np.mean(time_reverse[sub]))

    time_data = [np.mean(time_diff_identical), np.mean(time_diff_reverse), np.std(time_diff_identical), np.std(time_diff_reverse)]
    pickle.dump(avg_stats, open('accuracies_data.pkl', 'wb'))
    pickle.dump(time_data, open('time_diff_data.pkl', 'wb'))
    with open('data_acc.csv', 'a+') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',')
        datawriter.writerow(['Data'])
        datawriter.writerow(['correct_first', 'incorrect_second_correct_first_i', 'correct_second_incorrect_first_i', 'incorrect_second_correct_first_r', 'correct_second_incorrect_first_r', 'correct_first_i', 'correct_first_r', 'incorrect_first_i', 'incorrect_first_r'])

    for i in range(len(avg_stats)): 
        with open('data_acc.csv', 'a+') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',')
            avg_stats[i] = [round(x,2) for x in avg_stats[i]]
            datawriter.writerow(avg_stats[i])

    return avg_stats, time_data


if __name__ == "__main__" :
    stats = get_data_stats()
    print(stats)



