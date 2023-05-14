 
class GraphImageScaler(tfk.layers.Layer):
    def __init__(self, 
        mlp_width, 
        mlp_depth, 
        dropout=0.0, 
        hidden_units=None,
        layer_norm=False,
        activation="ReLU",
        kernel_initializer='glorot_normal',
        stop_f_grad=True,
        num_heads=8,
        **kwargs, 
        ):
        super().__init__(**kwargs)


        if hidden_units is None:
            hidden_units = 2 * image_mpl_width

        self.input_image   = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)
        self.input_scale  = tfk.layers.Dense(mlp_width, kernel_initializer=kernel_initializer)

        self.image_network = tfk.models.Sequential(
            [
                #Need a non-linearity between the input layer and transformer block
                FeedForward(
                    hidden_units=hidden_units, 
                    normalize=layer_norm, 
                    kernel_initializer=kernel_initializer, 
                    activation=activation, 
                    ),

            ] + [
                Transformer(
                    num_heads=num_heads, key_dim=hidden_units,
                    dropout=dropout,
                    hidden_units=hidden_units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    normalize=layer_norm,
                ) for i in range(mlp_depth)
            ] + [
                ConvexCombination(
                    kernel_initializer=kernel_initializer,
                    dropout=dropout
                )
        ])

        self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=hidden_units, 
                    normalize=layer_norm, 
                    kernel_initializer=kernel_initializer, 
                    activation=activation, 
                    ) for i in range(mlp_depth)
            ] + [
                tfk.layers.Dense(1, kernel_initializer=kernel_initializer),
        ])

        self.stop_f_grad = stop_f_grad

    @staticmethod
    def ragged_to_dense(tensor):
        """ Convert a ragged tensor to dense with an attention mask """
        mask = tf.ones_like(tensor[...,0])
        mask_1d = mask.to_tensor()
        mask = tf.einsum("...a,...b->...ab", mask_1d, mask_1d)
        mask = mask + (1. - mask_1d[...,None])
        mask = tf.cast(mask, 'bool')
        return mask


    def call(self, inputs, mc_samples=1, **kwargs):
        metadata, iobs, sigiobs, imodel = inputs
        
        if self.stop_f_grad:
            imodel = tf.stop_gradient(imodel)

        scale = metadata
        image = tf.concat((metadata, iobs, sigiobs, imodel), axis=-1)

        image = self.input_image(image)
        scale = self.input_scale(scale)

        image = self.image_network(image)
        scale = scale + image

        z = self.scale_network(scale)
        #z = tf.RaggedTensor.from_row_splits(tf.transpose(z), params.row_splits)
        #z = tf.exp(z)
        return z

