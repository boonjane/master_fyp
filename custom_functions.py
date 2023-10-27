#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Dice coefficient function
def dice_coefficient(y_true, y_pred):
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    
    y_true_flat = K.reshape(y_true_flat, K.shape(y_true))
    y_pred_flat = K.reshape(y_pred_flat, K.shape(y_pred))
    
    intersection = K.sum(y_true_flat * y_pred_flat)
    dice = (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)
    return dice

# Dice loss function
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)

# Intersection over Union (IoU) function
def iou(y_true, y_pred):
    y_true_shape = K.shape(y_true)
    y_pred_shape = K.shape(y_pred)
    
    if len(y_true_shape) == len(y_pred_shape):
        # Both tensors have the same number of dimensions
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    else:
        # One of the tensors has a different number of dimensions
        intersection = K.sum(K.abs(y_true * y_pred))
        union = K.sum(y_true) + K.sum(y_pred) - intersection
    
    iou = K.mean((intersection + 1e-15) / (union + 1e-15))
    return iou


# In[ ]:




