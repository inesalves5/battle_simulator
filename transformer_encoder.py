import tensorflow as tf
from tensorflow.keras import layers

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        
        # MLP
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

    def call(self, inputs, training=False, mask=None):
        # Atenção com residual
        attn_output = self.att(inputs, inputs, attention_mask=mask)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feedforward com residual
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class GameTransformer(tf.keras.Model):
    def __init__(self, n_player_pieces=64, n_opponent_pieces=126, n_actions=64,
                 embed_dim=128, num_heads=4, ff_dim=256, num_blocks=2):
        super().__init__()

        self.n_player_pieces = n_player_pieces
        self.n_opponent_pieces = n_opponent_pieces
        self.n_actions = n_actions
        self.total_tokens = n_player_pieces + n_opponent_pieces + n_actions + 1  # +1 tipo de jogo

        # Embeddings
        self.embed_player = layers.Dense(embed_dim)     
        self.embed_opponent = layers.Dense(embed_dim)   
        self.embed_actions = layers.Dense(embed_dim)    
        self.embed_game_type = layers.Dense(embed_dim) 

        self.encoders = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim) 
                         for _ in range(num_blocks)]
        
        # Pooling
        self.pool_player = layers.GlobalAveragePooling1D()
        self.pool_opponent = layers.GlobalAveragePooling1D()
        self.pool_actions = layers.GlobalAveragePooling1D()

        self.final_dense = layers.Dense(128, activation="relu")
        self.output_layer = layers.Dense(1, activation="linear")  

    def call(self, inputs, training=False, mask=None):
  
        player_tokens = inputs[:, :self.n_player_pieces]                 
        opponent_tokens = inputs[:, self.n_player_pieces:self.n_player_pieces+self.n_opponent_pieces]  
        actions_tokens = inputs[:, self.n_player_pieces+self.n_opponent_pieces:
                                   self.n_player_pieces+self.n_opponent_pieces+self.n_actions]        
        game_type_token = inputs[:, -1:]  

        player_emb = tf.expand_dims(self.embed_player(player_tokens), 1)     
        opponent_emb = tf.expand_dims(self.embed_opponent(opponent_tokens), 1)
        actions_emb = tf.expand_dims(self.embed_actions(actions_tokens), 1)
        game_type_emb = tf.expand_dims(self.embed_game_type(game_type_token), 1)

        x = tf.concat([player_emb, opponent_emb, actions_emb, game_type_emb], axis=1)  

        for encoder in self.encoders:
            x = encoder(x, training=training, mask=mask)

        player_out = self.pool_player(x[:, :1, :])     
        opponent_out = self.pool_opponent(x[:, 1:2, :])  
        actions_out = self.pool_actions(x[:, 2:3, :])    
        game_type_out = x[:, -1, :]  

        combined = tf.concat([player_out, opponent_out, actions_out, game_type_out], axis=-1)

        combined = self.final_dense(combined)
        return self.output_layer(combined)

    def predict_rewards(self, game_state, action):
        encoded = tf.convert_to_tensor(game_state.encode(action), dtype=tf.float32)
        encoded = tf.expand_dims(encoded, axis=0)  
        reward = self(encoded, training=False)
        return float(reward.numpy()[0][0])
