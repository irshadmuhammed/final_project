import tensorflow as tf

class AsymmetricLoss(tf.keras.losses.Loss):
    """
    Implementation of Asymmetric Loss for Multi-Label Classification in TensorFlow.
    Reference: https://arxiv.org/abs/2009.14119
    """

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8, reduction=tf.keras.losses.Reduction.AUTO, name='asymmetric_loss', from_logits=False):
        super(AsymmetricLoss, self).__init__(reduction=reduction, name=name)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Forward pass for the Asymmetric Loss.
        
        Args:
            y_true: Ground truth labels (multi-hot), shape (N, C).
            y_pred: Model logits (N, C) if from_logits=True, else probabilities.
            
        Returns:
            Computed loss.
        """
        # Ensure correct types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculating probabilities
        if self.from_logits:
            xs_pos = tf.sigmoid(y_pred)
        else:
            xs_pos = y_pred
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip > 0:
            xs_neg = tf.add(xs_neg, self.clip)
            xs_neg = tf.clip_by_value(xs_neg, clip_value_min=0.0, clip_value_max=1.0)

        # Basic Cross-Entropy calculation
        los_pos = y_true * tf.math.log(tf.maximum(xs_pos, self.eps))
        los_neg = (1 - y_true) * tf.math.log(tf.maximum(xs_neg, self.eps))

        # Asymmetric Focusing
        pt = y_true * xs_pos + (1 - y_true) * xs_neg
        one_sided_gamma = self.gamma_pos * y_true + self.gamma_neg * (1 - y_true)
        one_sided_w = tf.pow(1 - pt, one_sided_gamma)

        # Final loss calculation
        loss = -one_sided_w * (los_pos + los_neg)
        
        # Sum over classes (axis=-1) to get loss per sample
        return tf.reduce_sum(loss, axis=-1)
