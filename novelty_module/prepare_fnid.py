import pandas as pd

def create_dataset(ip_df, name):
	lst = []
	for i, row in ip_df.iterrows():
		if row['label_fnn'] == 'fake':
			label = 0
		elif row['label_fnn'] == 'real':
			label = 1
		lst.append([row['statement'].encode('utf-8'), row['paragraph_based_content'].encode('utf-8'), label])
	# Writing to a txt file
	with open(name.split('.')[0]+".txt", 'w', newline = '') as outfile:
		for obj in lst:
			outfile.write(str(obj[0]) + '\t' + str(obj[1]) + '\t' + str(obj[2]) + '\n')

train_df = pd.read_csv('../FNID_Dataset/fnn_train.csv')
dev_df = pd.read_csv('../FNID_Dataset/fnn_dev.csv')
test_df = pd.read_csv('../FNID_Dataset/fnn_test.csv')

create_dataset(train_df, 'fnn_train.csv')
create_dataset(test_df, 'fnn_test.csv')
create_dataset(dev_df, 'fnn_dev.csv')

data = data2 = "" 
  
# Reading data from file1 
with open('data/quora/train.txt', encoding='utf-8') as fp: 
    data = fp.read()
  
# Reading data from file2 
with open('../FNID_Dataset/train_fnn_mt1st.txt', encoding='utf-8') as fp: 
    data2 = fp.read()
  
# Merging 2 files 
# To add the data of file2 
# from next line 
data += "\n"
data += data2 
  
with open ('combined_train_fnn.txt', 'w', encoding='utf-8') as fp: 
    fp.write(data) 