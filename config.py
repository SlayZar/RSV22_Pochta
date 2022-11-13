TRAIN_DATAPATH = "../pochta/train_dataset_train.csv"
TEST_DATAPATH = "../pochta/test_dataset_test.csv"
cat_features = ['oper_type', 'oper_attr', 'oper_type + oper_attr', 'type',
 'class', 'mailctg',  'name_mfi', 'index_oper', 'postmark']
fit_model = True
MODEL_PATH = 'models'
ss_path = 'sample_solution.csv'
drop_cols = ["mailrank", "mailtype", "is_privatecategory"]
