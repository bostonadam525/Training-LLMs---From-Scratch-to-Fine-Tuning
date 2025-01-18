# Quantization in PyTorch
* This repo goes over principles of quantization mostly in PyTorch.
* Understanding the mathematics and technical principles of quantization is paramount when fine-tuning LLMs.


## What is Quantization?
* Basically the conversion from a higher memory format to lower memory format.

### Important concepts
1. Full precision/half-precision 
2. Calibration 
  * squeezing higher values to a scaled lower value. 
3. Modes of Quantization
  * a. Post training quantization - PQT
  * b. Quantization aware training - QAT

* Example: Let's say we have a neural network
  * All networks have Weights represented in matrices.
    * Let's say the weights are in a 3x3 matrix.
    * Each value is stored in 32 bits or FP32.
    * Full precision or single precision (stored as floating point numbers)
      * Example: FP 32 bit —> FP 16 bit (called “half precision”)
      * Example in TensorFlow: `TF32 bit —> TF16 bit`


* In a large LLM like llama2 with 70 billion params, if you have limited RAM with 32 GB, you cant download this model to your RAM due to limited cost, resources, memory! 

## What can we do to work with LLMs when we have limited cost and memory?
* Convert higher memory to lower memory!
  * example: 32 bit to int 8 
    * 32 bits will now be stored in 8 bits


## Quantization allows FAST INFERENCE!!
* Just a few examples/use-cases where this is most helpful
  1. mobile devices
  2. edge devices
  3. cloud
  4. fine-tuning


## Advantages of Quantization
1. Less memory consumption when loading LLMs on to basically any device. 
2. Less inference time due to simpler data types
3. Less energy consumption, because inference takes less computation overall



## Disadvantages of Quantization?
1. loss of information 
2. loss of accuracy


# How to Perform Quantization  - Mathematical Concepts and Theory
* There are 2 different types of quantization that we will focus on:

1. **Symmetric quantization**

2. **Asymmetric quantization**


# 1. Symmetric Quantization
* Batch normalization commonly used with neural network optimization, is a technique of symmetric quantization.
* In symmetric quantization, the range of original floating-point values is mapped to a symmetric range around zero in the quantized space. What this means is that the quantized value for zero in the floating-point space is **exactly zero in the quantized space.**



## What is Symmetric uint8 quantization (unsigned int8 quantization)?
* Lets say i have a FP number between 0.0 and 1000.0 as weights for an LLM. 
* Weights are usually stored in 32 bits.
  * We would aim to convert this into uint8 which is 2^8
  * The "u" before "int" is "Unsigned" which tells us the value range is from 0 to 255.
  * So what are Unsigned Integers of 8 bits? A uint8 data type contains all whole numbers from 0 to 255.
      * As with all unsigned numbers, the values must be **non-negative**.
      * Uint8's are mostly used in graphics (colors are always non-negative).

### How do we represent floating point numbers?
* In computer science, a floating point number (or floats) are positive or negative numbers with a decimal point.
* These values are represented by “ bits ”, or binary digits.
  * The IEEE-754 standard describes how bits can represent one of three functions to represent the value:
    1. the sign
    2. exponent
    3. fraction (or "mantissa" )

#### FP16: 
* Let's say we have the number 7.32 (see chart below, source: [Defining Floating Point Precision - What is FP64, FP32, FP16?](https://www.exxactcorp.com/blog/hpc/what-is-fp64-fp32-fp16)
  * FP16 Half Precision uses:
    * 1 bit for the positive/negative **sign**
    * 5 bits for representing the **exponent (range)** with base 2.
    * 10 bits for the **fraction/precision/mantissa**, i.e. value after the decimal point.
    * **Total: 16**
  * So for the number 7.32 we would have:
    * Sign —> 7
    * 5 bits —> exponent memory --> 2^5
    * rest of bits —> saved for mantissa

![image](https://github.com/user-attachments/assets/e93ec3e4-82f0-4f74-8efd-c763ce5d1a9a)


#### Bfloat16
* This is a custom floating point format called **“Brain Floating Point Format"**,” or **“bfloat16”** for short.
* The name comes from “Google Brain”, which is the artificial intelligence research group at Google where the idea for this format was invented.
* Bfloat16 is a custom 16-bit floating point format for ML that’s comprised of:
    * 1 sign bit
    * 8 exponent bits
    * 7mantissa bits
* **This is different from the industry-standard IEEE 16-bit floating point, which was not designed with deep learning applications in mind.**
* Figure 1 below from Google Cloud shows the internals of 3 floating point formats: (a) FP32: IEEE single-precision, (b) FP16: IEEE half-precision, and (c) bfloat16. (source: [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)


![image](https://github.com/user-attachments/assets/f3bb1b60-1ac1-4a19-8610-0030c0b2d217)


#### Floating Point 32
* FP32 Single Precision uses:
  * 1 bit for the positive/negative **sign**
  * 8 bits for representing the **exponent (range)** with base 2.
  * 23 bits for the **fraction/precision/mantissa**, i.e. value after the decimal point.
  * **Total: 32**

![image](https://github.com/user-attachments/assets/2544cc9d-816d-4b14-8a61-e0b3c369e2eb)


#### Floating Point 64
* FP64 or double precision 64-bit floating point precision utilizes 64 bits of binary to represent numbers in calculations performed on your system. This format offers the highest precision among mainstream options making it ideal for application that require precision and accuracy to the tee.

* FP64 Double precision uses:
  * 1 bit for the positive/negative **sign**
  * 11 bits for representing the **exponent (range)** with base 2.
  * 52 bits for the **fraction/precision/mantissa**, i.e. value after the decimal point.


![image](https://github.com/user-attachments/assets/4bdd6606-8818-4248-9743-0371bf51b591)



# How do we convert from FP to uint8?
* Let’s say we wish to map the floating point range `[0.0 .. 1000.0]` to the quantized range `[0 .. 255]`.
* The range `[0 .. 255]` is the set of values that can fit in an unsigned 8-bit integer.
* We assume a symmetric distribution (evenly distributed)
* This is the transformation we want to do:

![image](https://github.com/user-attachments/assets/e0beccb7-3998-4205-9cbd-34d732fbfb8b)

## Scaling Equation
* To do this conversion what we must do is similar to using a Min-Max Scalar in machine learning. 
* This is the equation:

![image](https://github.com/user-attachments/assets/c8dfd343-7ac5-4135-a2ba-4988fa05727b)


* So using the mapping we want this is what it looks like:
  * 0.0 maps to -->  0 (quantized)
  * 1000 maps to --> 255 (quantized)

* The equation breakdown is this:
  * `scale  = (Xmax - Xmin / Qmax - Qmin)`
  * `scale = (1000 - 0 / 255 - 0 ) = 3.92`
  * `scale=3.92`, so everything is scaled by this!!!

### Another Scaling example
* To convert from a floating point value to a quantized value, we can simply divide the floating point value by the scale.
* For example, the floating point value 500.0 corresponds to the quantized value
* The equation looks like this:
  * `round(500/3.9215) = round(127.5) = 128`

### Another Scaling exampmle 
* `round(250/3.92)= round(63.77) = 64`


# 2. Affine (or asymmetric) quantization
* This is the one that has a zero-point that is non-zero in value!
* Now let’s say we wish to map the floating point range `[-20.0 .. 1000.0]` to the quantized range `[0 .. 255].`
* This is the transformation:

![image](https://github.com/user-attachments/assets/682b04a8-9096-4e1c-9797-e333347e49dd)

* This is the equation we want:

![image](https://github.com/user-attachments/assets/a7db96d1-b45c-471b-b658-08ee189d8b71)

* This gives us: `1000 + 20 / 255 = 4.0` —> and 4.0 is the scale factor we need.

## Affine/Assymetric example:
* `round(-20/4) = (-5.0) + 5 = 0`
  * Here the 5 is called **zero point**

* **The zero-point acts as a bias for shifting the scaled floating point value and corresponds to the value in the quantized range that represents the floating point value 0.0.**
* The zero point is always the negative of the representation of the minimum floating point value since the minimum will always be negative or zero. 
* The equation looks like this:

![image](https://github.com/user-attachments/assets/c4ba525b-e4d9-4cfb-818b-4763112c8da2)

* The transformation mapping:

![image](https://github.com/user-attachments/assets/60389bd0-bcac-4028-a854-7d353e6ead65)





# Modes of Quantization
* There are 2 modes of quantization we will cover:

## 1. **Post training quantization (aka “PTQ”)**
  * This is one of the most common quantization techniques which involves quantizing a model’s parameters (both weights and activations) **after training the model.**
  * * Problem: loss of data —> accuracy decreases 
  * This is the general flow:
      * `Pre-trained model (FIXED weights) ——> calibration of weights DATA —> quantized model —> use cases`
  * Quantization of the weights is performed using either **symmetric or asymmetric quantization.**
  * Quantization of the activations requires inference of the model to get their potential distribution since we do not know their range.
  * There are 2 forms of quantization of the activations:
    1. **Dynamic Quantization**
    2. **Static Quantization**

### Most popular PTQ methods
* GPTQ, one-shot weight quantization method
* GGML
* QLORA’s 4bits ( bitsandbytes )

## 2. **Quantization aware training (aka “QAT”)**
  * The general concept here is: new training data to use for fine-tuning model —> then we quantize the model!! 
  * Solution to PTQ is that with this method there is generally **no loss of data and no loss of accuracy**
  * The general flow with this method is:
    * `TRAINED MODEL —> QUANTIZATION —> FINE-TUNING`

### QAT methods
1. LLM-QAT (Key-Value Data-Free Quantization Aware Training)
2. KV-QAT

* These specialized quantization techniques are designed to reduce quantization errors in the Key-Value (KV) caches of Large Language Models (LLMs).
* KV caches store critical intermediate outputs from attention layers, ensuring the model does not need to recompute information for each token.
* Unlike other quantization methods that primarily focus on activations or weights, KV-QAT targets these caches, which play a crucial role in every token’s processing. 

## Comparing PTQ vs. QAT


![image](https://github.com/user-attachments/assets/37b458f3-8b02-4595-bd6d-dd61b02b62a1)


## When to Choose QAT over PTQ?
* When the model architecture is sensitive to quantization (activations) and retraining is feasible.
* **Accuracy sensitive: When high accuracy is paramount, regardless of higher training costs.**
* Low bit quantization: Low bit quantization like 2/4 bit.

## When to Choose PTQ over QAT?
* **Large model size: When retraining is impractical or computational resources are limited.**
* **Accuracy sensitivity: When a slight reduction in accuracy is acceptable.**
* A well-tuned PTQ model can be a good base for subsequent QAT fine-tuning.

## Best of both worlds (PTQ with QAT Fine-tuning):
  * Combining PTQ and QAT to get best of both worlds is an area of active research.






# References
* [A Comprehensive Study on Quantization Techniques for Large Language Models](https://arxiv.org/html/2411.02530v1#S1)
* [A Visual Guide to Quantization](https://www.maartengrootendorst.com/blog/quantization/#the-realm-of-4-bit-quantization)
* [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)
* [Defining Floating Point Precision - What is FP64, FP32, FP16?](https://www.exxactcorp.com/blog/hpc/what-is-fp64-fp32-fp16)
* [Diving deeper into Quantization Realm : Introduction to PTQ and QAT](https://iprathore71.medium.com/diving-deeper-into-quantization-realm-9c73e3172a3c)
* [Hugging Face Quantization Blog Post](https://huggingface.co/docs/optimum/en/concept_guides/quantization#quantization-to-int8)
* [Quantization Aware Training (QAT) vs. Post-Training Quantization (PTQ)](https://medium.com/better-ml/quantization-aware-training-qat-vs-post-training-quantization-ptq-cd3244f43d9a)
* [The Power of Quantization in ML: A PyTorch Tutorial Part 1](https://medium.com/@sayedebad.777/the-power-of-quantization-in-ml-a-pytorch-tutorial-part-1-8d0c1bf8b679)
* [Tensor Quantization: The Untold Story](https://towardsdatascience.com/tensor-quantization-the-untold-story-d798c30e7646)
