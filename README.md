# spam-classification
Using sklearn Bayes and Support Vector Machines to predict spam/ham mails

# dataset
Download link for entire Ling-spam corpus
http://www.aueb.gr/users/ion/data/lingspam_public.tar.gz

The emails in this dataset have went through lemmatization and removal of unnecessary words such as 'of', 'in', 'and', etc

For this project, train set size is 702 (301 ham, 301 spam), test set size is 260 (130 ham, 130 spam)

# classifcation model used
sklearn Multinomial Naive Bayes and SVC

# accuracy achieved
- Multinomial Naive Bayes model has an error of 3/126 for ham and 7/134 for spam
- SVC model has an error of 6/122 for ham and 14/138 for spam
