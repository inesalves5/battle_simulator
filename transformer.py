import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_nlp.layers import TransformerEncoder

INPUT_DIM = 190+64+1 # 190 for state, 64 for action, 1 for action type

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Multi-head self attention
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        
        # Feedforward
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        
        # Normalização
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Atenção com residual
        attn_output = self.att(inputs, inputs)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  

        # Feedforward com residual
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerNetwork(tf.keras.Model):
    def __init__(self, input_dim=INPUT_DIM, embed_dim=256, num_heads=4, ff_dim=512, num_blocks=2):
        super(TransformerNetwork, self).__init__()
        
        # Projeta entrada densa para embedding
        self.embedding = layers.Dense(embed_dim)

        # Vários blocos encoder
        self.encoders = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim) 
                         for _ in range(num_blocks)]
        
        # Pooling para reduzir sequência → vetor fixo
        self.global_pool = layers.GlobalAveragePooling1D()

        # Saída final
        self.output_layer = layers.Dense(1, activation="linear")

    def call(self, inputs, training=False):
        # (batch, features) → (batch, seq_len=1, embed_dim)
        x = self.embedding(inputs)
        x = tf.expand_dims(x, axis=1)

        # Passa pelos encoders
        for encoder in self.encoders:
            x = encoder(x, training=training)

        # Pooling para vetor
        x = self.global_pool(x)
        return self.output_layer(x)
