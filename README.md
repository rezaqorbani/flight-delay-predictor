# US Flight delay predictor

## Table of Contents

- [US Flight delay predictor](#us-flight-delay-predictor)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Features](#features)
  - [Model](#model)
  - [Results](#results)
  - [Inference](#inference)
  - [Montioring](#montioring)
  - [Appendix](#appendix)
    - [Airports](#airports)

## Introduction

This is a serveless service for predicting the delay (in minutes) of any flight in any of top 18 busiest airports (listed in [Airports](#airports) section) in US. Our model is trained on historical flight and weather data from 2021 and 2022 but uses real-time weather data during inference. Using GitHub actions we schedule a run of data pipeline to add fresh data to our training dataset as well as monitor the performance of our model on the latest data. The model can be trained on-demand. There are seperate serverless UI for inference and monitoring. [Hopsworks](https://www.hopsworks.ai/) is used to store extracted features as well the trained model.

## Data

## Features

## Model

## Results

## Inference

## Montioring

## Appendix

### Airports
