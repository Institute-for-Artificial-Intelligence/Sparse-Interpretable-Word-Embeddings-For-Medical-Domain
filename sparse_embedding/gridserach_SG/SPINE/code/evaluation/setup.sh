
cd extrinsic_downstream

cd newsgroups
mkdir data
# download data and prepare pickle dumps
python3 get_newsgroup_data.py


cd ../np_bracketing
mkdir data
# download data and prepare pickle dumps
wget http://angelikilazaridou.github.io/resourses/parsing/NP_dataset.tar.gz
tar -xvzf NP_dataset.tar.gz
python3 get_data.py


cd ../sentiment_analysis/
mkdir data
# download data and prepare pickle dumps
cd data
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip stanfordSentimentTreebank.zip
cd ..
python3 get_data.py


cd ../TREC/
mkdir data
# download data and prepare pickle dumps
cd data
wget http://cogcomp.org/Data/QA/QC/train_5500.label
wget http://cogcomp.org/Data/QA/QC/TREC_10.label
cd ..
python3 create_data.py

