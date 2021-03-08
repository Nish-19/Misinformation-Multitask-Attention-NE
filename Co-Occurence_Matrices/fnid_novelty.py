import pandas as pd

def coocurance_matrix(col1, col2):
	co_mat = pd.crosstab(col1, col2)
	return co_mat
train_df = pd.read_csv('../FNID_Dataset/fnn_train.csv')
test_df = pd.read_csv('../FNID_Dataset/fnn_test.csv')
dev_df = pd.read_csv('../FNID_Dataset/fnn_dev.csv')
quora_df_train = pd.read_csv('train_fnid_nv_best.csv')
quora_df_test = pd.read_csv('test_fnid_nv_best.csv')
quora_df_dev = pd.read_csv('dev_fnid_nv_best.csv')

quora_df_train.rename(columns = {'0':'Quora_Labels'}, inplace=True)
quora_df_test.rename(columns = {'0':'Quora_Labels'}, inplace=True)
quora_df_dev.rename(columns = {'0':'Quora_Labels'}, inplace=True)

with open('FNID_Novelty_Quora_Best_Characterestics.txt', 'w') as infile:
	co_mat_train = coocurance_matrix(train_df.label_fnn, quora_df_train.best_novelty)
	print('##############Train Novelty-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_train, file=infile)
	# Test
	co_mat_test = coocurance_matrix(test_df.label_fnn, quora_df_test.best_novelty)
	print('##############Test Novelty-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
	# Test
	co_mat_dev = coocurance_matrix(dev_df.label_fnn, quora_df_dev.best_novelty)
	print('##############Dev Novelty-Quora Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_dev, file=infile)