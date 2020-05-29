#!/bin/bash


python train_InsightFace_pallel.py  --dist-url 'tcp://172.17.51.1:23415' --dist-backend 'nccl' --multiprocess --world-size 1 --rank 0
