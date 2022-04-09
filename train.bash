log_file=exp/logs/NUCLA/exp_11062020_T0_v1t-v3t.log
date >> $log_file
python actionRecognition.py >> $log_file
date >> $log_file