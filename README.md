# Amazon CAPTCHA Solver

## Overview
This project aims to develop a robust and efficient CAPTCHA solver specifically designed for Amazon's CAPTCHA system. Our goal is to create a model that can accurately decode Amazon CAPTCHAs with high speed and low resource usage.

## Data Source
We are using the Amazon CAPTCHA Database provided by a-maliarov as our primary data source. This repository contains a large collection of Amazon CAPTCHA images.

Data Repository: [amazon-captcha-database](https://github.com/a-maliarov/amazon-captcha-database)

The repository includes:
- 13,000 unique CAPTCHA patterns in the 'unique' folder
- Backup training data in the 'backup' folder
- Accuracy test logs in the 'accuracy' folder

## Project Goals
1. Develop a minimal CNN model that can achieve high accuracy (aiming for 99%+) in solving Amazon CAPTCHAs.
2. Optimize the model for fast inference time to support real-time applications.
3. Create a lightweight solution that can be easily deployed and scaled across multiple devices or servers.

## Current Progress
- Implemented a MinimalCNN model with promising results
- Achieved ~97% accuracy with the initial model
- Exploring ways to increase accuracy without significantly impacting inference time

## Next Steps
- Fine-tune the model architecture to improve accuracy
- Implement data augmentation techniques to enhance model generalization
- Optimize the model for deployment in a distributed environment
- Develop an API for easy integration with existing systems