VERSION=v21
LOGFILE=logs/exp_${VERSION}_test.log

python3 testClassifier_Multi.py > "$LOGFILE" 2>&1 &
