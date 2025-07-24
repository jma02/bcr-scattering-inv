# Code to implement ...
- Solving inverse wave scattering with deep learning, Yuwei Fan, Lexing Ying
- BCR-Net: A neural network based on the nonstandard wavelet
form, Yuwei Fan, Cindy Orozco Bohorquez, Lexing Ying.
This code itself was gifted to us by Dr. Yuwei Fan. 

Some notes.
On modern GPUs, I have experienced extreme weight collapse or extreme weight explosion.
This code works fine on CPUs and on older GPUs (2020 era).

Use conda to manage cuda tooling. 
`conda env create -f environment_gpu.yml`

Have fun.