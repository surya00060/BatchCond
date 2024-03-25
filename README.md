# BatchCond Framework: Efficient Batched Inference in Conditional Neural Networks
Code will be open-sourced after publication. 

## BatchCond Framework: Instructions for Conditional-Depth NNs: Early Exit CNNs
`EarlyExitCNNs` folder contains all the code for training the model from scratch, collecting the training data for predictor, training the predictor and running Batched Conditional NN inference. 

To measure batch size 1 performance,
```
python batching.py --measure_batch_size_1_inference
```
To measure batched inference performance,
```
python batching.py --measure_batched_inference
```
To run BatchCond: SimBatch + ABR, run the following command
```
python batching.py --measure_simbatch_abr
```
