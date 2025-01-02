<div align="center">
<h1>MambaJSCC </h1>
<h3>MambaJSCC: Adaptive Deep Joint Source-Channel Coding with Generalized State Space Model</h3>


Tong Wu<sup>1</sup>, Zhiyong Chen<sup>1</sup>, Meixia Tao<sup>1</sup>, Fellow, IEEE, Yaping Sun<sup>3</sup>, Xiaodong Xu<sup>2,3</sup>,
Wenjun Zhang<sup>1</sup>, Fellow, IEEE, and Ping Zhang<sup>2,3</sup>, Fellow, IEEE  

1. Cooperative Medianet Innovation Center (CMIC),Shanghai Jiao Tong University, Shanghai Key Laboratory of Digital Media Processing and Transmission

2.   Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications 

3.	 Department of Broadband Communication, Peng Cheng Laboratory  


Paper : https://arxiv.org/abs/2409.16592

</div>

# Overview

This is the official deployment of the paper "MambaJSCC: Adaptive Deep Joint Source-Channel Coding with Generalized State Space Model"

<p align="center">
  <img src="./figs/structure.png" alt="System Structure" width="95%">
</p>

<p align="center">
  <img src="./figs/structure.png" alt="System Structure" width="95%">
</p>


# Getting start

	The [main.py]() file contain the main experiments of our paper. 
	
	For the first step, you should change the path in the [main.py]() file according to your environment.
	
	We provide some checkpoints of our model and you can download them [here](https://drive.google.com/drive/folders/103Shcs7Gh5LKoz2smJXVt85ok1m_e2mL?usp=sharing).
	
	After download the checkpoints, you can directly run the [main.py]() file to evaluate the performance of the JSCC system and the joint JSCC and CDDM system at an SNR of 10 dB under the AWGN channel.
	
	The code about training has been annotated, but the related code has been contained in the project, you can run the function directly.



# Citation

```
@misc{wu2024mambajsccadaptivedeepjoint,
      title={MambaJSCC: Adaptive Deep Joint Source-Channel Coding with Generalized State Space Model}, 
      author={Tong Wu and Zhiyong Chen and Meixia Tao and Yaping Sun and Xiaodong Xu and Wenjun Zhang and Ping Zhang},
      year={2024},
      eprint={2409.16592},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2409.16592}, 
}

```
