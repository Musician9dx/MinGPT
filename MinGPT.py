import tensorflow as tf
from tensorflow.keras.layers import Dense, Permute, Multiply, LayerNormalization, Embedding, Layer, Add
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras import Model
self.pEmbedding=PositionEmbedding(vocabSize,embedSize)


tf.random.set_seed(1337)


class AttentionBlock(Layer):

    def __init__(self, embed_size):
        super(AttentionBlock, self).__init__()
        self.embed_size = embed_size
        self.key = Dense(embed_size, activation="linear")
        self.query = Dense(embed_size, activation="linear")
        self.value = Dense(embed_size, activation="linear")
        self.lnm = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # Assuming input_shape is [batch_size, sequence_length, embed_size]
        seq_len = input_shape[1]
        one_matrix = tf.ones((seq_len, seq_len))
        self.mask = tf.linalg.band_part(one_matrix, -1, 0)

    def call(self, inputs):
        key = self.key(inputs)
        query = self.query(inputs)
        value = self.value(inputs)

        attention_scores = tf.matmul(query, key, transpose_b=True)
        # Expanding mask dimensions to [1, seq_len, seq_len] for broadcasting
        mask = tf.expand_dims(self.mask, 0)
        attention_scores *= mask  # Applying mask
        attention_scores = softmax(attention_scores, axis=-1)

        attention_output = tf.matmul(attention_scores, value)
        attention_output = self.lnm(attention_output)

        return attention_output


class MultiHeadAttention(Layer):

    def __init__(self, numHeads, embed_size):
        super(MultiHeadAttention, self).__init__()

        self.headSize = embed_size // numHeads
        self.numHeads = numHeads
        self.embed_size = embed_size

    def build(self):
        self.head = [AttentionBlock(self.headSize) for i in range(self.numHeads)]

    def call(self, inputs):
        c = []

        for layer in range(self.numHeads):
            op = self.head[layer](inputs)
            c.append(op)

        op = tf.concat(c, axis=-1)

        return op


class BigramLanguageModel(Model):

    def __init__(self, vocab_size, embed_size):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = Embedding(vocab_size, embed_size)
        self.self_attention1 = AttentionBlock(embed_size)
        self.self_attention2 = AttentionBlock(embed_size)
        self.self_attention3 = AttentionBlock(embed_size)
        self.self_attention4 = AttentionBlock(embed_size)
        self.mha = MultiHeadAttention(4, embed_size)
        self.pEmbedding=PositionEmbedding(vocab_size,embed_size)

        self.dense1 = Dense(embed_size, activation="linear")
        self.dense2 = Dense(embed_size, activation="linear")

        self.adc = Add()

    def call(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        logits=logits+seld.pEmbedding(logits)
        
        lg1 = logits
        logits = self.mha(logits)
        logits = self.adc([relu(logits), lg1])

        lg1 = logits
        logits = self.self_attention1(logits)
        logits = self.adc([relu(logits), lg1])

        lg1 = logits
        logits = self.self_attention2(logits)
        logits = self.adc([relu(logits), lg1])

        lg1 = logits
        logits = self.self_attention3(logits)
        logits = self.adc([relu(logits), lg1])

        lg1 = logits
        logits = self.self_attention4(logits)
        logits = self.adc([relu(logits), lg1])

        logits = self.dense1(logits)
        logits = self.dense2(logits)

        if targets is None:
            return logits, None
        else:
            B, T, C = tf.shape(logits)
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            return logits, loss

    def generate(self, idx, new_tokens):
        for _ in range(new_tokens):
            logits, _ = self.call(idx)
            logits = logits[:, -1, :]
            probs = softmax(logits, axis=-1)
            idx_next = tf.reshape(tf.argmax(probs, axis=-1), (-1, 1))
            idx = tf.concat([idx, tf.cast(idx_next, tf.int32)], axis=1)
        return idx

    def fitM(self, xb, yb, steps=100):
        optimizer = tf.keras.optimizers.Adam()

        for step in range(steps):
            with tf.GradientTape() as tape:
                logits, loss = self.call(xb, yb)
                print(f"Step: {step}", float(loss))
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

