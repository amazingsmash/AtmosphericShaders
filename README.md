# AtmosphericShaders

Analytical and numerical analysis and rendering of the problem of computing efficient atmospheric scattering. Matlab and Mathematica code is provided, along with GLSL shaders for producing a spherical atmosphere-like effect on virtual globes.

Shading is based on the most prominent atmospheric raycasting using a simplified linear model of the atmospheric density to calculate how much light is scattered our way on each fragment. The numerical calculations are heavily simplified by exploiting the spherical symmetry of the atmosphere.

## Analytical

Contains a series of Mathematica Notebooks which present the Rayleigh Scattering integrations and derive the final expressions. 

## Simple Model

Contains a simplification of the model for fast rendering on GLSL (tested on OpenGL ES 2.0).

## Numerical Analysis

AtmosphereRenderer.m is a matlab parametric renderer of the atmosphere scattering.
AtmCodeGenerator.m is matlab code that generates the the code of a GLSL shader.

## Full Model

Contains a GLSL implementation of the full scattering model achieved by the analytical integration. 
