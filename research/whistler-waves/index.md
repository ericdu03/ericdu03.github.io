---
layout: post
title: Whistler Waves in the Solar Wind
advisor: "Dr. Ivan Vasko"
permalink: /research/whistler-waves/
tags: ["MATLAB"]
author: Eric Du
date: 2026-06-26
---

# Overview 
This project centers around investigating the presence of whistler waves emanating from the sun, detected by
space probes orbiting the moon (~1 AU). The reason they are interesting to us is because they are a potential
candidate for explaining a long-standing problem observed in the solar wind: the heat flux instability
problem.  

In short, the problem is as follows: because the solar wind radiates away from the sun, its temperature
follows an inverse power law as a function of distance:

$$
    T \propto r^{-\alpha}
$$

Depending on the theoretical model you choose, the value of $\alpha$ is different. If the solar wind expands
purely adiabatically, then classical thermodynamics predicts $\alpha = 4/3$, but if were to introduce
collisions into the picture, then $\alpha = 2 / 7$. In practice, however, what we observe is $\alpha \approx
0.3$ to $0.6$, which fits neither of these models very well. This motivates the conclusion that the solar
wind expands mostly adiabatically, while certain instabilities within the solar wind release energy and slow
down the temperature decay.   

One such instability thought to play a significant role in generating this missing energy are whistler waves,
which are low-frequency waves typically in the 1 to 30 Hz range. According to leading models, these waves
occur frequently enough and with enough energy to meaningfully add kinetic energy into the solar wind,
possibly providing a partial explanation for the observed decay curve.

To analyze these waves, we utilize data collected from the THEMIS-ARTEMIS mission launched by NASA in
2007.[^themis] This mission consists of five spacecraft -- named THEMIS A, B, C, D, and E -- equipped with
sensitive instruments designed to measure the local electromagnetic and particle dynamics around the Earth.
However, in 2008 two of these spacecraft, THEMIS B and C, were launched to orbit the moon instead, and it's
these two spacecraft that this project focused on.[^artemis]

The reason we chose these two spacecraft is both due to their on board instrumentation and also their
location. Firstly, whistler waves primarily show up in the magnetic field data, which can be detected using
THEMIS's magnetometers. Secondly, THEMIS is also equipped with other instruments that measure quantities like
the ion and electron velocities, which I also made use of in my analysis. Finally, and perhaps most
importantly, the spacecraft's orbit around the moon means that our magnetic field measurements are not
impacted by terrestrial activity which have the potential to introduce false positives in our data. These
three factors combined make THEMIS B and C the ideal spacecraft for this kind of analysis.   

There's one final motivating factor I have yet to mention: in 2019, NASA made a software upgrade to this
spacecraft, allowing data to be sent at a rate of 512 Hz instead of the previous 128 Hz, allowing us to
perform this analysis using data of the highest-available quality. 

# Extracting the Data

The raw data files are stored as `.cdf` files at `themis.ssl.berkeley.edu`, and is completely free to
access. Using the data visualizer on the website, we can pick out dates which potentially have whistler
waves of interest, then navigate to the raw data source and download the corresponding `.cdf` file. In
particular, for any given day there are two sets of data that we need to download: the fluxgate magnetometer
and the search coil magnetometer. 

While both of these are magnetometer measurements, they serve extremely different functions. The fluxgate
magnetometer operates at a frequency of 64 Hz, and specializes in low-frequency background magnetic field
measurements. By contrast, the search coil magnetometer captures data at a rate between 0.1 and 4 kHz, which is
significantly more sensitive and allows it to pick up whistler waves that live in the 10-100 Hz range with
exceptional accuracy. Both of these magnetometers provide crucial data for our analysis, as the flux-gate
magnetometer is used to calibrate a background, and the search-coil is used to pick up the whistler wave
signal.    

# Analysis

Once the data is downloaded, we perform a fast Fourier transform (FFT) to convert the raw data into frequency
space, and plot the data using a power spectral density plot to identify the window of time over which a
whistler is observed. Then, we compute other metrics of the wave such as the polarization (i.e.
whether the wave is linearly or circularly polarized), and use thresholds on these values to form a set of
selection criteria for what constitutes a whistler wave. With these waves selected, we then extract data
from the electrostatic analyzer to then compute statistics such as the ion flow velocity, ion temperature,
and other relevant statistics.  
   
Overall, throughout this analysis I analyzed approximately 40 million points of data spanning across nearly
three years. While this project didn't end up in a publication and therefore I don't have much to show for
it, this was the very first research project I conducted, and I greatly attribute my love for data science
and passion for research to the work done here.    



[^themis]: Yes, there was an ARTEMIS mission before the modern moon landing one.  
[^artemis]: These two spacecraft were also simultaneously renamed to ARTEMIS P1 and P2, but I will continue
    calling them THEMIS B and C. 
 
