# BatchCond Framework: Efficient Batched Inference in Conditional Neural Networks
Code will be open-sourced after publication. 

## BatchCond Instructions for Conditional-Depth NNs: Early Exit CNNs
`EarlyExitCNNs` folder contains all the code for training the model from scratch, collecting the training data for predictor, training the predictor and running Batched Conditional NN inference. 


To run BatchCond: SimBatch + HW_Aware ReOrg, run the following command
```
python batching.py --measure_batched_inference
```

To evaluate the effectiveness of SimBatch, run the following command
```
python batching.py ----eval_SimBatch
```
