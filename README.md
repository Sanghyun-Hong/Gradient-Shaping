# On the Effectiveness of Mitigating Data Poisoning Attacks with Gradient Shaping

This repository contains the code for the paper <br>
[_On the Effectiveness of Mitigating Data Poisoning Attacks with Gradient Shaping_](https://arxiv.org/abs/2002.11497).

**Authors:** [Sanghyun Hong](http://sanghyun-hong.com), [Varun Chandrasekaran](http://pages.cs.wisc.edu/~chandrasekaran/), [Yiǧitcan Kaya](http://www.cs.umd.edu/~yigitcan), [Tudor Dumitraș](http://users.umiacs.umd.edu/~tdumitra/), and [Nicolas Papernot](https://www.papernot.fr/) <br>
**Contact:** [Sanghyun Hong](mailto:shhong@cs.umd.edu)


## Abstract

Machine learning algorithms are vulnerable to data poisoning attacks. Prior taxonomies that focus on specific scenarios, _e.g._, indiscriminate or targeted, have enabled defenses for the corresponding subset of known attacks. Yet, this introduces an inevitable arms race between adversaries and defenders. In our work, we study the feasibility of an attack-agnostic defense relying on artifacts that are common to all poisoning attacks. Specifically, we focus on a common element between all attacks: they modify gradients computed to train the model. We identify two main artifacts of gradients computed in the presence of poison: (1) their l2-norms have significantly higher magnitudes than those of clean gradients, and (2) their orientation differs from clean gradients. Based on these observations, we propose two prerequisites for a generic poisoning defense: it must bound  gradient magnitudes and minimize differences in orientation. We call this _gradient shaping_. As an exemplar tool to evaluate the feasibility of gradient shaping, we use differentially private stochastic gradient descent (DP-SGD), which clips and perturbs individual gradients during training to obtain privacy guarantees. We find that DP-SGD, even in configurations that do not result in meaningful privacy guarantees, increases the model's robustness to indiscriminate attacks. It also mitigates worst-case targeted attacks and increases the adversary's cost in multi-poison scenarios. The only attack we find DP-SGD to be ineffective against is a strong, yet unrealistic, indiscriminate attack. Our results suggest that, while we currently lack a generic poisoning defense, gradient shaping is a promising direction for future research.

**[Notice]:** Reproducing our results may take more time than you expected; thus, we recommend skipping the reproducing steps and exploring the results from our evaluations in the [``results``](results) directory.


---


## Install Dependencies

You can install the required Python packages by running the following command:

```
  $ pip install -r requirements.txt
```

You can also install the specific version of [TensorFlow-Privacy](https://github.com/tensorflow/privacy) that we used by running the following commands:

```
  $ git submodule update --init --recursive --remote
  $ cd privacy
  $ pip install -e .
```

---


## Analyzing the Impact of Poisoning Mechanisms on Gradients

To reproduce the analysis results in Sec. 3 and 4 of our paper.

### Gradient Analysis: Feature Collision

To cause the impact of feature collision on gradients during training, we use the watermarking technique to cause the impact of feature collision on gradients during training. You can reproduce the results by running the following scripts:


**[Collision (LR, Scratch)]:** an LR model is trained from scratch on the FashionMNIST-3/4 training set containing multiple poisons:
```
  $ ./analyze_collision.sh subtask multi
```

**[Collision (MLP, Re-train)]:** when we update a trained MLP model on the entire FashionMNIST containing multiple poisons:
```
  $ ./analyze_collision.sh fashion_mnist multi
```


### Gradient Analysis: Feature Insertion

To exploit the feature insertion by poisons, we utilize backdooring mechanisms. You can reproduce our results by running the following scripts:

**[Insertion (MLP, Re-train)]:** when we update a trained MLP model on the entire FashionMNIST containing multiple poisons:
```
  $ ./analyze_insertion.sh fashion_mnist 0
```


### Gradient Analysis: Magnitude and Orientation Differences in Gradients

Running the above scripts will store the averaged gradients from clean and poison samples to the ``param_updates`` folder; for instance, ``results/analysis/collision/a_pair_0_1/fashion_mnist_3_4.1.0/vanilla_lr_300_40_0.01``.

To compute the magnitude and orientation differences between the clean and poison gradients in each epoch, you can run the following script. You need to specify the location containing the ``param_updates`` folder at the ``results_base`` variable:

```
  $ python3 analyze_gradients.py
```

The script will create ``mextracts_epoch.csv`` and ``aextracts_epoch.csv`` in the same location.


### Gradient Analysis: Results

**[Collision]** <br>
Our results are in [``results/analysis/collison``](results/analysis/collison) folder. All the subdirectory that stores each analysis uses the following naming convention, ``<analysis-type>/<dataset>/<intensity>/<model>``:

- ``analysis-type``: The prefix ``a_pair_`` indicates that we used a single poison, but ``pairs_of_`` means we used multiple-poisons. Also, the suffix ``_retrain`` implies that we consider the re-training scenario.

- ``dataset``: ``fashion_mnist_3_4.1.0`` means FashionMNIST-3/4 (the binary dataset) is used whereas ``fashion_mnist`` indicates that we use the entire FashionMNIST dataset.

- ``intensity``: The name starts with ``alpha_``, and the following number indicates the intensity—_i.e._, the interpolation ratio.

- ``model``: The details about the training of a model ``<vanilla/dp>_<model>_<batch-size>_<epochs>_<learning-rate>``.


**[Insertion]:** <br>
Our results are in [``results/analysis/insertion``](results/analysis/insertion) folder. All the subdirectory that stores each analysis uses the following naming convention, ``<dataset>/<intensity>/<model>``:

- ``dataset``: We only consider the entire FashionMNIST dataset ``fashion_mnist``.

- ``intensity``: The name means ``<backdoor-label>/<poison-ratio>/<patch-size>/``.

- ``model``: The details about the training of a model ``<vanilla/dp>_<model>_<batch-size>_<epochs>_<learning-rate>``.


**[Interpreting Results]:** <br>

``mextracts_epoch.csv`` stores the magnitudes of the averaged gradients computed from the clean and poison samples—_i.e._, the estimation of an individual gradient—over an epoch.

```
  0.08842676881862006,0.9355205916848285
  0.01853436994802128,0.11387253561197228
  0.012392445541291722,0.07633205789016119
  0.13739815486495846,0.10181093015223937
  0.1650391350165304,0.0076037653113433775
  ...
```

``aextracts_epoch.csv`` stores the cosine similarity scores between the averaged gradients computed from the clean and poison samples:


```
  0.11206721417779844
  -0.3868968795311831
  -0.13791345133168953
  0.6948125421567697
  -0.8147651988123773
  ...
```


---


## Mitigating Indiscriminate Poisoning Attacks with Gradient Shaping

To reproduce the analysis results in Sec. 6.2 of our paper.

### Mitigation

Here, we consider two indiscriminate poisoning attacks: (1) the random label-flipping and (2) the attack formulated by Steinhardt et al. For the simplicity, we provide the poisoning samples crafted in each attack in [datasets/poisons/indiscriminate](datasets/poisons/indiscriminate). You can conduct the poisoning attacks by running the following script:

```
  $ ./do_ipoisoning.sh subtask 0
```

We use DP-SGD to realize gradient shaping. You can consider a set of clipping norms and noise multipliers by specifying them in the ``NORMCLP`` and ``NOISEML`` in the bash script.

**[Notice]:** Running this script takes a lot of time, so we highly recommend not to run and look at the results that the repo contains. Each result is stored in ``results/ipoisoning/<attack>/<dataset>/<model>/attack_results.csv`` file.


### Gradient Analysis

You can analyze the impact of gradient shaping during training by running the following scripts:

```
  $ ./analyze_ipoisoning.sh subtask
  $ python3 analyze_gradients.py
```

Running this script will create the same output files as what we observed in our feature collision and insertion analysis. The results will be stored in ``results/analysis/ipoisoning/<attack>/<dataset>/<model>``.


---


## Mitigate Targeted Poisoning Attacks with Gradient Shaping

To reproduce the analysis results in Sec. 6.3 and 6.4 of our paper.

### Mitigation

Here, we consider the clean-label targeted poisoning attack formulated by Shafahi et al. For the simplicity, we provide the poisoning samples crafted in [datasets/poisons/targeted](datasets/poisons/targeted). To run the attack, you first need to run the following scripts:

```
  $ ./do_tpoisoning.sh purchases oneshot
  $ ./do_tpoisoning.sh purchases multipoison
```

Those scripts split the entire task of targeted poisoning attacks for each of 100 targets using 100 poisons into the number of task and processes specified in the scripts. You can modify ``TOTTASKS`` and ``TOTPROCS`` for that.


Running them will create multiple scripts with the name ``<dataset>-net-<model-count>-tpoisoning-<attack-type>-<task-idx>-of-<total-tasks>.sh``. You need to run all of them one by one or across multiple machines:

```
  $ ./purchases-net-0-tpoisoning-one-1-of-10.sh
  ...
  $ ./purchases-net-10-tpoisoning-multi-1-of-10.sh
```

**[Notice]:** Running this script takes a lot of time, so we recommend not to run and look at the results that the repo contains. Each result is stored in ``results/tpoisoning/clean-labels/<attack-type>/<dataset>/<model>/<attack-budget>/attack_w_<target>.csv`` file.


### Gradient Analysis

You can analyze the impact of gradient shaping during training by running the following scripts:

```
  $ ./analyze_tpoisoning.sh purchases
  $ python3 analyze_gradients.py
```

Running this script will create the same output files as what we observed in our feature collision and insertion analysis. The results will be stored in ``results/analysis/tpoisoning/clean-labels/<dataset>/<model>``.


---


## Cite This Work

You are encouraged to cite our paper if you use this code for academic research.

```
@article{Hong20GS,
    author        = {Sanghyun Hong and
                     Varun Chandrasekaran and
                     Yiğitcan Kaya and
                     Tudor Dumitraş and
                     Nicolas Papernot},
    title         = {On the Effectiveness of Mitigating Data Poisoning Attacks with Gradient Shaping},
    journal       = {CoRR},
    volume        = {abs/2002.11497},
    year          = {2020},
    url           = {https://arxiv.org/abs/2002.11497},
    archivePrefix = {arXiv},
    eprint        = {2002.11497},
    timestamp     = {Wed, 26 Feb 2020 14:04:16 +0100},
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


**Fin.**
