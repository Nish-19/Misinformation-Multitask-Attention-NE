import pandas as pd 

train_df = pd.read_csv('../../FNID_Dataset/fnn_train.csv')
dev_df = pd.read_csv('../../FNID_Dataset/fnn_dev.csv')
test_df = pd.read_csv('../../FNID_Dataset/fnn_test.csv')

train_em_df = pd.read_csv('../bert_based_klinger/fnn_em_train.tsv_k_numb_predictions_bin.csv')
test_em_df = pd.read_csv('../bert_based_klinger/fnn_em_test.tsv_k_numb_predictions_bin.csv')
dev_em_df = pd.read_csv('../bert_based_klinger/fnn_em_dev.tsv_k_numb_predictions_bin.csv')

train_em_df_go = pd.read_csv('fnd_goemotion_train.csv')
test_em_df_go = pd.read_csv('fnd_goemotion_test.csv')
dev_em_df_go = pd.read_csv('fnd_goemotion_dev.csv')

def combine_emotion(ip_df, kg_df, go_df, name):
	em_lst = []
	for i, row in ip_df.iterrows():
		label = row['label_fnn']
		kling_label = kg_df.loc[i, 'emotion_label']
		go_label = go_df.loc[i, 'statement_go']
		# checking
		if label == 'fake':
			if kling_label == 1 or go_label == 1:
				em_lst.append(1)
			else:
				em_lst.append(0)
		elif label == 'real':
			if kling_label == 0 or go_label == 0:
				em_lst.append(0)
			else:
				em_lst.append(1)

	em_new = pd.DataFrame()
	em_new['best_emotion'] = em_lst
	em_new.to_csv(name, index = False)

combine_emotion(train_df, train_em_df, train_em_df_go, 'fnd_train_emocom.csv')
combine_emotion(test_df, test_em_df, test_em_df_go, 'fnd_test_emocom.csv')
combine_emotion(dev_df, dev_em_df, dev_em_df_go, 'fnd_dev_emocom.csv')