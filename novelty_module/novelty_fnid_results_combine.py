import pandas as pd 

train_df = pd.read_csv('../FNID_Dataset/fnn_train.csv')
dev_df = pd.read_csv('../FNID_Dataset/fnn_dev.csv')
test_df = pd.read_csv('../FNID_Dataset/fnn_test.csv')

# Quora-0
train_quora_0 = pd.read_csv('train_fnid_quora_0.csv')
test_quora_0 = pd.read_csv('test_fnid_quora_0.csv')
dev_quora_0 = pd.read_csv('dev_fnid_quora_0.csv')
train_quora_0.rename(columns = {'0':'Quora_Labels'}, inplace=True)
test_quora_0.rename(columns = {'0':'Quora_Labels'}, inplace=True)
dev_quora_0.rename(columns = {'0':'Quora_Labels'}, inplace=True)

# Quora-1
train_quora_1 = pd.read_csv('train_fnid_quora_1.csv')
test_quora_1 = pd.read_csv('test_fnid_quora_1.csv')
dev_quora_1 = pd.read_csv('dev_fnid_quora_1.csv')
train_quora_1.rename(columns = {'0':'Quora_Labels'}, inplace=True)
test_quora_1.rename(columns = {'0':'Quora_Labels'}, inplace=True)
dev_quora_1.rename(columns = {'0':'Quora_Labels'}, inplace=True)

# Quora-2
train_quora_2 = pd.read_csv('train_fnid_quora_2.csv')
test_quora_2 = pd.read_csv('test_fnid_quora_2.csv')
dev_quora_2 = pd.read_csv('dev_fnid_quora_2.csv')
train_quora_2.rename(columns = {'0':'Quora_Labels'}, inplace=True)
test_quora_2.rename(columns = {'0':'Quora_Labels'}, inplace=True)
dev_quora_2.rename(columns = {'0':'Quora_Labels'}, inplace=True)

# combine nv results
def combine_results(ip_df, qdf0, qdf1, qdf2, name):
	nv_results = []
	for i, row in ip_df.iterrows():
		label = row['label_fnn']
		nv0 = qdf0.loc[i, 'Quora_Labels']
		nv1 = qdf1.loc[i, 'Quora_Labels']
		nv2 = qdf2.loc[i, 'Quora_Labels']
		if label == 'fake':
			if nv0 == 0 or nv1 == 0 or nv2 == 0:
				nv_results.append(0)
			else:
				nv_results.append(1)
		elif label == 'real':
			if nv0 == 1 or nv1 == 1 or nv2 == 1:
				nv_results.append(1)
			else:
				nv_results.append(0)
	new_df = pd.DataFrame()
	new_df['best_novelty'] = nv_results
	new_df.to_csv(name, index = False)

combine_results(train_df, train_quora_0, train_quora_1, train_quora_1, 'train_fnid_nv_best.csv')
combine_results(test_df, test_quora_0, test_quora_1, test_quora_1, 'test_fnid_nv_best.csv')
combine_results(dev_df, dev_quora_0, dev_quora_1, dev_quora_1, 'dev_fnid_nv_best.csv')