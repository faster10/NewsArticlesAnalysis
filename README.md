#NEWSArticleAnalysis
A model is trained on news articles and then these articles were evaluated to write a comparison on how satire articles are differently articulated than serious articles.
Trainnn.py file trains the model while pre_train.py loads the pre-trained model (trained on 25k articles in total, 12.5 k serious articles while 12.5k satiric)
Pre_train_testSet locates test articles (satire and serious) in the test set and calculates precision, recall and f-score scores for the test set.
