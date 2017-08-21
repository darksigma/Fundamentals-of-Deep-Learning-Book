## slim_seq2seq
A distillation of the official Tensorflow seq2seq model and tutorial.

## Files

`seq2seq.py`: Contains the implementations of the encoder and decoder for the basic RNN seq2seq model and for the attention-based RNN seq2seq model.

`seq2seq_model.py`: Contains the code for processing minibatches with bucketing and accumulating gradients.

`translate.py`: End-to-end implementation of the NMT problem; defines global variables, loads data, and contains code for running training and decoding.

`data_utils.py`: Utility methods for loading, tokenizing, and preprocessing English-to-French corpus.

## Tutorial

The main logic for starting the training, as well as decoding a particular sentence, lies in `translate.py`. All flags are passed to this file, where the user has a choice between training the model, or decoding a provided sentence (translating it to French) using a trained model. The specifics of this are detailed in the first set of comments in `translate.py` in lines X-Y.

#### `translate.py`

There are four functions in `translate.py`: `read_data()`, `create_model()`, `train()`, `decode()`; the first two are convenience functions to prepare the data and model, the train function contains the end-to-end training code, and the decode function is used for interactive translation. Each method is explained in more detail here:

`read_data`: Reads training data (English and French sentences) for the source language and the target language; used in the main training loop.

`create_model`: Create attention-based seq2seq model based on user-defined and/or default FLAGS or loads a checkpointed model from a previous run from a specified training directory, `train_dir`.

`train`: Runs forward and backward passes with data to train the model and reports statistics; this method is reviewed in more detail in the next immediate section.

`decode`: Loads a saved model and runs a single forward pass of the model through a provided English sentence, outputting the greedily decoded French translation; this method is reviewed later.

###### `train()`

The `train()` method arguably the most important function in the code, and is therefore presented with a high level overview, and then a detailed investigation of the function's internals for those interested.

###### High-Level Overview

The pipeline used for this application is very similar to most machine learning pipelines, and is presented here. More specific details are described in the following section.

1. Load the training and development sets for both the English and French pre-processed sentences.
2. Instantiate the TensorFlow session and create the model using `create_model()`.
3. Read the appropriate data in batches using `read_data()` and perform one training update based on the data from the batch.
4. Continue training and after every epoch, checkpoint the model and save statistics. Repeat indefinitely.

###### Detailed Walkthrough with Code Snippets

1) First, we choose the appropriate bucket according to the distribution of the data. Recall that buckets contain sequences that have similar length, and this technique is used for further optimization of the training procedure. Using buckets are not necessary in a sequence-to-sequence implementation, but it has been found to greatly improve training speed. The corresponding bucket is selected:
```
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])
```
The data is fetched by `get_batch` and fed into the model via the placeholders:
```
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
```
In this line, the next batch is extracted, and each of the `encoder_inputs`, `decoder_inputs`, and `target_weights` are the placeholders for the input data.

2) The model takes a single step forward in processing this batch, and the loss for this step is returned and later accumulated.

```
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      ...
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1
```
Take careful note of the abstraction at this level -- the loss being returned for the batch is not being accumulated in order to perform gradient-based optimization. The loss being returned here is simply used for logging and checkpointing purposes; the optimization is already done within the `step()` method. Along these lines, another consideration to be aware of is the following: unless the boolean `forward_only` is set to false, the gradients computed in the `step()` method are not used to update the model's parameters. This update procedure is done in `step()`, but is skipped if only forward propagation is specified.

3) Repeat Steps 1 and 2 for `n=FLAGS.steps_per_checkpoint` updates, or in other words, one epoch. When training deep models on large sets of data, an epoch does not necessarily have to complete one pass through the training data and can instead be completed by a fixed number of updates to the model, as is done here. This is often much more feasible when reporting intermediate metrics and usually supplies a reasonable estimate of performance on our training and developmental sets, assuming that the data trained on thus far is uniformly sampled.

4) Checkpoint the model:
  * Compute perplexity for the previous training epoch, which we compute by simply exponentiating the most recent loss statistic.
  * Anneal learning rate if there are no improvements over last three epochs.
  * Save the model to `train_dir`.
  * Evaluate model thus far on development set and report perplexity.

5) Repeat Steps 1-4.

###### `decode()`

Decode follows the exact same flow as `train()` does; for brevity, we do not reiterate the same steps. The session is instantiated, and the model is created using `create_model()`. The second argument passed is the `forward_only` flag, and in this case, is set to `true` since we only decode a sentence but we freeze the weights and do not update them according to a computed loss.

The batch size is set to 1 since we only process one sentence at a time (note that this would be both very computationally inefficient as well as lead to noisy gradient samples and unstable training). The sentence is tokenized, bucketed, and fed to the main computation graph using the same `step()` method. In this case, because we are seeking the explicit output activations of the decoder and the model is in `forward_only` mode, we are onlyr eturned with `output_logits`. As the name suggests, these are the unnormalized log-probabilities of the output layers in the decoder.

```
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
```
The logits are then passed through an argmax layer  and then to mapped to tokens from the French vocabulary, where end-of-sequence tokens are used to delimit the sentence or phrase.
```
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
```
Finally, the translated French phrase or sentence is outputted, and the system is now ready for the next sentence:
```
      print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()
```


##### `seq2seq_model.py`

This file the main seq2seq model itself; this implementation uses attention. The class Seq2SeqModel is defined, each it has three main functions: `__init__`, `step()`, and `get_batch()`, which we discuss in more detail below. Note that while the model presented is of a certain form, it is easily amenable to using different RNN schemes.

###### `__init__()`
This method defines the model and consumes all input parameters.Most are self explanatory, but some of particular note are `max_gradient_norm`, which is a hyperparameter denoting at what point we clip gradients, the `use_lstm` flag, which will let us pick between using LSTM cells or GRU cells (default), and `forward_only` flag, which will avoid constructing the backward pass part of the model (useful for pure decoding).

###### `sampled_loss()`
This is an important feature of the model. The sampled softmax is a special type of softmax that was specifically designed for NMT models to try to handle a large target vocabulary.

##### `seq2seq.py`

##### `data_utils.py`
