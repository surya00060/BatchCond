# BatchCond Framework: Efficient Batched Inference in Conditional Neural Networks
This repository contains code for the paper Efficient Batched Inference in Conditional Neural Networks, accepted to ESWeek CASES 2024.

## Dependencies
To install the necessary packages, use `pip install -r requirements.txt `

This code was tested on RTX 2080 Ti GPU with CUDA Version 12.2

## Commands for Conditional-Depth NNs: Early Exit CNNs
`EarlyExitCNNs` folder contains all the code for perfoming batched inference. 

To measure batch size 1 performance,
```
python batching.py --measure_batch_size_1_inference
```
To measure batched inference performance with padding,
```
python batching.py --measure_batched_inference --batch_size 128
```
To run BatchCond: SimBatch + ABR, run the following command
```
python batching.py --measure_simbatch_abr --batch_size 128
```