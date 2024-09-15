# Avg_FT
Codes for our paper to 2025ICASSP: Two Heads Are Better Than One: Averaging along Fine-Tuning to Improve Targeted Transferability.

As shown in below, the proposed AaF steers AE towards a more central region than FFT.

<img src="results/loss_surface/919_01.png" width="250"><img src="results/loss_surface/919_02.png" width="250"><img src="results/loss_surface/919_03.png" width="250">  

<img src="results/loss_surface/919_04.png" width="250"><img src="results/loss_surface/919_05.png" width="250"><img src="results/loss_surface/919_06.png" width="250">  

<img src="results/loss_surface/919_07.png" width="250"><img src="results/loss_surface/919_08.png" width="250"><img src="results/loss_surface/919_09.png" width="250">

In the supplementary file 'supp.pdf', we provide more detailed results:

- Ablation study on the decaying factor gamma;
- Visualization of FFT and AaF in a 2D subspace;
- Attack performance on Swin;
- Attack performance in the most difficult-target scenario; 
- Visual comparison. 

## Usage
Please run main_Avg_FT.py to see the targeted transferability improvement by the proposed AaF method.




