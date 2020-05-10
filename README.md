# AtmosphericShaders
GLSL shaders for producing an spherical atmosphere-like effect on virtual globes.

Shading is based on atmospheric raycasting using a simplified linear model of the atmospheric density to calculate how much light is scattered our way on each fragment.

## Analytical

Contains a series of Mathematica Notebooks which present the Rayleigh Raycacast integrations and derive the final expressions. 

## Simple Model

Contains a simplification of the model for fast rendering on GLSL (tested on OpenGL ES 2.0).

## Numerical Analysis

AtmosphereRenderer.m is a matlab parametric renderer of the atmosphere scattering.
AtmCodeGenerator.m is matlab code that generates the the code of a GLSL shader.

## Full Model

Contains a GLSL implementation of the full scattering model achieved by the analytical integration. 
