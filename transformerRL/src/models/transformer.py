import tensorflow as tf
import numpy as np

def positional_encoding(inputs, max_len, d_model):
    """
    生成位置编码并添加到输入中。
    Args:
        inputs: [batch_size, seq_len, d_model]
        max_len: 序列最大长度 (History Size)
        d_model: Embedding 维度
    """
    # 生成位置索引 [0, 1, ... max_len-1]
    position = tf.range(max_len, dtype=tf.float32)
    
    # 计算分母项
    div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(np.log(10000.0) / d_model))
    
    # 扩展维度以便广播
    pos_enc = tf.expand_dims(position, 1) * tf.expand_dims(div_term, 0)
    
    # 计算 sin 和 cos
    pos_enc = tf.concat([tf.sin(pos_enc), tf.cos(pos_enc)], axis=1) # [max_len, d_model]
    
    # 扩展 batch 维度 [1, max_len, d_model]
    pos_enc = tf.expand_dims(pos_enc, 0)
    
    # 将位置编码加到输入上 (截取实际序列长度)
    # 注意：inputs 的维度是 [batch, seq_len, d_model]
    return inputs + pos_enc[:, :tf.shape(inputs)[1], :]

def scaled_dot_product_attention(Q, K, V, d_model, mask=None):
    """计算注意力权重"""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)  # [..., seq_len, seq_len]
    
    # 缩放
    dk = tf.cast(d_model, tf.float32)
    scaled_attention_logits = matmul_qk / tf.sqrt(dk)

    # 如果有 mask (用于屏蔽 padding 或未来信息)，在这里添加
    if mask is not None:
        # mask 为 1 的位置是 padding，乘以负无穷
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    
    return output, attention_weights

# 【重要】确保这个函数名是 transformer_encoder_layer
def transformer_encoder_layer(inputs, d_model, num_heads, dropout_rate=0.1, is_training=True, mask=None, scope="transformer_layer"):
    """
    标准的 Transformer Encoder Layer (TF 1.x 风格)
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # 确保 d_model 能被 num_heads 整除
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            
        depth = d_model // num_heads

        # --- 1. Multi-Head Attention ---
        # LayerNorm (Pre-Norm 结构)
        norm1 = tf.contrib.layers.layer_norm(inputs)
        
        # 线性映射生成 Q, K, V
        Q = tf.layers.dense(norm1, d_model, name="dense_Q")
        K = tf.layers.dense(norm1, d_model, name="dense_K")
        V = tf.layers.dense(norm1, d_model, name="dense_V")

        # Split heads: [batch, seq_len, num_heads, depth]
        Q_ = tf.reshape(Q, [-1, tf.shape(inputs)[1], num_heads, depth])
        K_ = tf.reshape(K, [-1, tf.shape(inputs)[1], num_heads, depth])
        V_ = tf.reshape(V, [-1, tf.shape(inputs)[1], num_heads, depth])

        # Transpose for matmul: [batch, num_heads, seq_len, depth]
        Q_ = tf.transpose(Q_, perm=[0, 2, 1, 3])
        K_ = tf.transpose(K_, perm=[0, 2, 1, 3])
        V_ = tf.transpose(V_, perm=[0, 2, 1, 3])

        # Attention calculation
        attn_out, _ = scaled_dot_product_attention(Q_, K_, V_, depth, mask=mask)

        # Transpose back: [batch, seq_len, num_heads, depth]
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])
        
        # Concat heads: [batch, seq_len, d_model]
        attn_out = tf.reshape(attn_out, [-1, tf.shape(inputs)[1], d_model])
        
        # Output projection
        attn_out = tf.layers.dense(attn_out, d_model, name="dense_output")
        attn_out = tf.layers.dropout(attn_out, rate=dropout_rate, training=is_training)
        
        # Residual Connection 1
        out1 = inputs + attn_out

        # --- 2. Feed Forward Network (FFN) ---
        norm2 = tf.contrib.layers.layer_norm(out1)
        
        ffn = tf.layers.dense(norm2, d_model * 4, activation=tf.nn.relu, name="ffn_1")
        ffn = tf.layers.dense(ffn, d_model, name="ffn_2")
        ffn = tf.layers.dropout(ffn, rate=dropout_rate, training=is_training)
        
        # Residual Connection 2
        final_output = out1 + ffn
        
        return final_output