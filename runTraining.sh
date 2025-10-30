#!/bin/bash

python scripts/ss-tumbling.py --save --noise 0.1 --force --batch 8 --epochs 300 --T 3
python scripts/ss-tumbling.py --save --noise 0.1 --force --batch 8 --epochs 300 --T 10
python scripts/ss-tumbling.py --save --noise 0.1 --force --batch 8 --epochs 300 --T 20
python scripts/ss-tumbling.py --save --noise 0.1 --force --batch 8 --epochs 300 --T 60




