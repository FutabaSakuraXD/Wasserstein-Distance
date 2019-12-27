# Wasserstein-Distance
Loss of two distribution(# Adapted from https://github.com/gpeyre/SinkhornAutoDiff)
This code is used calculate the Wasserstein distance of two distribution. Remember to apply a softmax layer before call it.
Since the original code has no default values and learning its background knowledge is time consuming, I simply asked my mentor and added some default values. So, you can just initialize it this way:
w_dist = WassersteinDistance()
