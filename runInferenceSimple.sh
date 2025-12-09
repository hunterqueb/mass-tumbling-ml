#!/bin/bash


python scripts/ss-tumbling-inference-avg.py --save --simpleGains
python scripts/ss-tumbling-inference-avg.py --save --OOD --simpleGains
python scripts/ss-tumbling-inference-avg.py --save --controlSat --simpleGains
python scripts/ss-tumbling-inference-avg.py --save --OOD --controlSat --simpleGains
