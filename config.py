TRAIN_DATAPATH = "RSV22_Pochta/train_dataset_train.csv"
TEST_DATAPATH = "RSV22_Pochta/test_dataset_test.csv"
cat_features = ['oper_type', 'oper_attr', 'oper_type + oper_attr', 'type',
 'class', 'mailctg',  'name_mfi', 'index_oper', 'postmark']
fit_model = False
MODEL_PATH = 'RSV22_Pochta/models'
ss_path = 'RSV22_Pochta/sample_solution.csv'
drop_cols = ["mailrank", "mailtype", "is_privatecategory"]
tresh = 0.1
