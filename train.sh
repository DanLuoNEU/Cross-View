VERSION=v12
LOGFILE=logs/exp_${VERSION}.log

python3 trainClassifier_Multi.py > "$LOGFILE" 2>&1 &