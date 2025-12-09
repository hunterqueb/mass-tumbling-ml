#!/bin/bash


python scripts/ss-tumbling-inference-avg.py --save
python scripts/ss-tumbling-inference-avg.py --save --OOD
python scripts/ss-tumbling-inference-avg.py --save --controlSat
python scripts/ss-tumbling-inference-avg.py --save --OOD --controlSat
