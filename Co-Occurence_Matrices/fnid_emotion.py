import pandas as pd 

def coocurance_matrix(col1, col2):
	co_mat = pd.crosstab(col1, col2)
	return co_mat

train_df = pd.read_csv('../FNID_Dataset/fnn_train.csv')
dev_df = pd.read_csv('../FNID_Dataset/fnn_dev.csv')
test_df = pd.read_csv('../FNID_Dataset/fnn_test.csv')

train_em_df = pd.read_csv('fnd_train_emocom.csv')
test_em_df = pd.read_csv('fnd_test_emocom.csv')
dev_em_df = pd.read_csv('fnd_dev_emocom.csv')

train_em_df.rename(columns = {'0':'Emotion_Label'}, inplace=True)
test_em_df.rename(columns = {'0':'Emotion_Label'}, inplace=True)
dev_em_df.rename(columns = {'0':'Emotion_Label'}, inplace=True)

with open('FNID_Combine_Emotion_KlingGo_Characterestics.txt', 'w') as infile:
	# Train set
	co_mat_train = coocurance_matrix(train_df.label_fnn, train_em_df.best_emotion)
	print('##############Train FND-Emotion-KlingGo Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_train, file=infile)
	# Test Set
	co_mat_test = coocurance_matrix(test_df.label_fnn, test_em_df.best_emotion)
	print('##############Test FND-Emotion-KlingGo Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_test, file=infile)
	# Dev set
	co_mat_dev = coocurance_matrix(dev_df.label_fnn, dev_em_df.best_emotion)
	print('##############Dev FND-Emotion-KlingGo Co-Occurance Matrix##############\n', file=infile)
	print(co_mat_dev, file=infile)