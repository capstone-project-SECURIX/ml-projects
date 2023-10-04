# Ass3 - Association Algorithm

# Pose Estimation with Local Joint - Association Algorithm

## Table of Contents
- [Introduction](#introduction)
- [Research Paper](#research-paper)
- [Code (Colab) Link](#code-colab-link)
- [Output Screenshots](#output-screenshots)
- [Association Algorithm](#association-algorithm)
  - [Description](#description)
  - [Algorithm](#algorithm)
  - [Advantages](#advantages)
  - [Limitations](#limitations)
  - [Applications](#applications)
  - [Future Scope](#future-scope)
- [Pose Estimation with Local Joint](#pose-estimation-with-local-joint)
  - [Usage of Association Algorithm](#usage-of-association-algorithm)
  - [Major Tuning Parameters](#major-tuning-parameters)
- [Conclusion](#conclusion)

## Introduction
This README provides an overview of the "Pose-estimation-with-local-joint" project, focusing on the use of the Association Algorithm in pose estimation, its advantages, limitations, applications, and future scope.

## Research Paper
- [Multi-Person Pose Estimation with Local Joint](https://paperswithcode.com/paper/multi-person-pose-estimation-with-local-joint)

## Research Paper Summary
The paper titled "Multi-Person Pose Estimation with Local Joint-to-Person Associations" addresses the challenge of estimating the poses of multiple individuals in a single image, particularly in situations where people may be occluding each other or may be partially outside the frame arxiv.org, paperswithcode.com.

The authors propose a solution that treats multi-person pose estimation as a joint-to-person association problem. They construct a fully connected graph from a set of detected joint candidates in an image and resolve the joint-to-person association and outlier detection using integer linear programming arxiv.org.

However, since solving joint-to-person association jointly for all persons in an image is an NP-hard problem and even approximations are expensive, the authors propose to solve the problem locally for each person arxiv.org.

The authors tested their approach on the challenging MPII Human Pose Dataset for multiple persons. Their method achieved the accuracy of a state-of-the-art method but was 6,000 to 19,000 times faster arxiv.org, paperswithcode.com.

This approach can handle situations where humans are in groups or crowds, and where a person can be occluded by another person or might be truncated. This makes it particularly useful in real-world applications where such scenarios are common.

## Code (Colab) Link
- [Google Colab Code](https://colab.research.google.com/drive/14aLspmK2MbTUmECW-GvT5M4kQBaAvbKn?usp=sharing)

## Output Screenshots
- ![Output Screenshot 1](https://github.com/capstone-project-SECURIX/ml-projects/blob/main/Ass3-Association/output/s1.png)
- ![Output Screenshot 2](https://github.com/capstone-project-SECURIX/ml-projects/blob/main/Ass3-Association/output/s2.png)

## Association Algorithm
### Description
The Association Algorithm is a technique used in multi-person pose estimation. It aims to associate body joints or keypoints with specific individuals in a scene. This algorithm plays a crucial role in understanding the spatial relationships between body parts and individuals, which is vital for accurately estimating poses in crowded or multi-person scenarios.

### Algorithm
The Association Algorithm typically involves the following steps:
1. Keypoint Detection: Detect body keypoints in an image using a pose estimation model.
2. Keypoint Clustering: Cluster detected keypoints to group them by individual.
3. Association: Associate each cluster of keypoints with a specific person by considering spatial proximity and movement patterns.
4. Pose Estimation: Finally, estimate the pose of each individual based on the associated keypoints.

### Advantages
- Enables accurate multi-person pose estimation even in crowded scenes.
- Helps in understanding human spatial relationships and interactions.
- Useful for applications like action recognition, sports analysis, and surveillance.

### Limitations
- Computationally intensive, especially in scenarios with a large number of people.
- Sensitive to occlusions and overlapping individuals.
- Requires fine-tuning and parameter optimization for different scenarios.

### Applications
- Human pose estimation in images and videos.
- Action recognition in videos.
- Sports analysis for player tracking and movement analysis.
- Surveillance systems for abnormal behavior detection.

### Future Scope
The Association Algorithm continues to evolve with advancements in computer vision and deep learning. Future research may focus on:
- More robust association methods to handle occlusions and complex scenes.
- Efficient implementation for real-time applications.
- Integration with 3D pose estimation for enhanced accuracy.

## Pose Estimation with Local Joint
### Usage of Association Algorithm
The "Pose-estimation-with-local-joint" project utilizes the Association Algorithm to accurately estimate poses in images or videos containing multiple individuals. It leverages the algorithm's ability to associate keypoints with specific people, allowing for precise pose estimation in various scenarios.

### Major Tuning Parameters
Some major tuning parameters used in the project include:
- Keypoint Detection Model: The choice of the underlying pose estimation model affects accuracy.
- Keypoint Clustering Parameters: Parameters for grouping keypoints into clusters.
- Association Thresholds: Thresholds for spatial proximity and movement patterns.
- Post-processing Techniques: Techniques to refine pose estimation results.

## Conclusion
The "Pose-estimation-with-local-joint" project showcases the application of the Association Algorithm in multi-person pose estimation. Understanding the algorithm's working, advantages, limitations, and tuning parameters is crucial for achieving accurate pose estimation results in complex scenarios.
