

@tfk.saving.register_keras_serializable(package="abismal")
class BGImageScaler(tfk.models.Model):
    bijector_dict = {
        'softplus' : tf.nn.softplus,
        'elup1' : lambda x : tf.nn.elu(x) + 1.,
        'exp' : tf.math.exp,
        'elog' : elog,
    }
    posterior_dict = {
        'normal' : normal_posterior,
        'gamma' : gamma_posterior,
        'foldednormal' : folded_normal_posloc_posterior,
        'rice' : rice_posterior,
        'lognormal' : log_normal_posterior,
        #'delta' : delta_posterior,
        'posnormal' : pos_normal_posterior,
    }
    prior_dict = {
        'cauchy' : cauchy_prior,
        'laplace' : laplace_prior,
        'normal' : normal_prior,
        'cennormal' : cen_normal_prior,
        'cenlaplace' : cen_laplace_prior,
        'halfnormal' : halfnormal_prior,
        'halfcauchy' : halfcauchy_prior,
        'exponential' : exponential_prior, 
        'lognormal' : lognormal_prior, 
        'foldednormal' : foldednormal_prior,
        'gamma' : gamma_prior,
    }
    def __init__(
            self, 
            mlp_width=32, 
            mlp_depth=20, 
            hidden_units=None,
            activation="relu",
            kl_weight=1.,
            epsilon=1e-12,
            ff_epsilon=None,
            normalizer_gain=0.1,
            num_image_samples=None,
            share_weights=True,
            prior_name='exponential',
            background_prior_name='laplace',
            posterior_name='foldednormal',
            background_posterior_name='normal',
            bijector_name='softplus',
            normalizer_name=None,
            hkl_to_imodel=False,
            gated=False,
            output_bias=True,
            dropout=None,
            random_seed=1234,
            batch_normalize=True,
            **kwargs, 
        ):
        """
        This function has a lot of overrides, but comes with sensible defaults built in. 

        Parameters
        ----------
        mlp_width : int (optional)
            Default 32 neurons. This is referred to as d-model in the CLI / paper
        mlp_depth : int (optional)
            Default 20 layers
        hidden_units : int (optional)
            This defaults to 2*mlp_width
        activation : str (optional)
            This is a Keras activation function with the default being "relu". 
        kl_weight : float (optional)
            The importance of the prior distribution on scales. This parameter is ignored if the posterior is a delta distribution. 
        epsilon : float (optional)
            A small constant for numerical stability defaults to 1e-12.
        ff_epsilon : float (optional)
            Epsilon used in the feed forward layers' pre-normalization (see `normalizer_name`).
            Defaults to None, in which case it falls back to `epsilon`.
        normalizer_gain : float (optional)
            Gain applied in the numerator of the 'l2'/'rms' feed forward normalizers. The default is 0.1.
        num_image_samples : int (optional)
            The number of reflections to sample in order to create the image representation vectors. 
            No subsampling will be done if this is set to None which is the default. 
        share_weights : bool (optional)
            Whether or not share neural network weights between the image model and the scale model. 
            The default is True. 
        prior_name : str (optional)
            The name of the prior distribution to use. 
        posterior_name : str (optional)
            The posterior parameterization to use
        bijector_name : str (optional)
            The bijector to use for parameters that need to be constrained positive
        normalizer_name : str (optional)
            The name of the normalizing function to use in the neural network
        hkl_to_imodel : bool (optional)
            Optionally allow the neural network to access the miller indices while computing the image representation vector. 
        gated : bool (optional)
            Optionally use a gated architecture instead of a vanilla residual multilayer perceptron
        output_bias : bool (optional)
            Use bias in the output layer. 
        dropout : float (optional)
            Use dropout when pooling reflections. 
        """
        super().__init__(**kwargs)
        self.kl_weight = kl_weight
        self.num_image_samples = num_image_samples
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.epsilon = epsilon
        self.ff_epsilon = ff_epsilon
        self.normalizer_gain = normalizer_gain
        self.activation = activation
        self.share_weights = share_weights
        self.prior_name = prior_name.lower()
        self.posterior_name = posterior_name.lower()
        self.bijector_name = bijector_name
        self.normalizer_name = normalizer_name
        self.hkl_to_imodel = hkl_to_imodel
        self.gated = gated
        self.output_bias = output_bias
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.random_seed = random_seed
        self.batch_normalize = batch_normalize
        self.background_posterior_name = background_posterior_name
        self.background_prior_name = background_prior_name

        input_bias=False

        if self.hidden_units is None:
            self.hidden_units = 2 * mlp_width
        ffepsilon = epsilon if ff_epsilon is None else ff_epsilon

        #kernel_initializer = 'glorot_normal'
        kernel_initializer = tfk.initializers.VarianceScaling(scale=mlp_depth**-1.0, mode='fan_avg', seed=random_seed) #FixUp init for early layers

        self.input_image = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=input_bias)
        self.input_scale = tfk.layers.Dense(
                mlp_width, kernel_initializer=kernel_initializer, use_bias=input_bias) #Should use_bias?

        self.pool = Average(axis=-2, dropout=dropout)
        #self.pool = ConvexCombination()

        if gated:
            from abismal.layers import GLUFeedForward as FeedForward
        else:
            from abismal.layers import FeedForward 

        self.image_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units,
                    activation=self.activation,
                    kernel_initializer=kernel_initializer,
                    use_bias=True,
                    normalizer=normalizer_name,
                    epsilon=ffepsilon,
                    normalizer_gain=normalizer_gain,
                ) for _ in range(mlp_depth)])
        if share_weights:
            self.scale_network = self.image_network
        else:
            self.scale_network = tfk.models.Sequential([
                FeedForward(
                    hidden_units=self.hidden_units, 
                    kernel_initializer=kernel_initializer, 
                    activation=self.activation, 
                    use_bias=True,
                    normalizer=normalizer_name,
                    epsilon=ffepsilon,
                    normalizer_gain=normalizer_gain,
                    ) for _ in range(mlp_depth)
            ]) 

        self.output_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer, use_bias=output_bias)
        self.background_dense = tfk.layers.Dense(2, kernel_initializer=kernel_initializer, use_bias=output_bias)
        #self.built = True

    def get_config(self):
        config = super().get_config()
        config.update({
            'mlp_width' : self.mlp_width,
            'mlp_depth' : self.mlp_depth, 
            'hidden_units': self.hidden_units,
            'epsilon' : self.epsilon,
            'ff_epsilon' : self.ff_epsilon,
            'normalizer_gain' : self.normalizer_gain,
            'activation' : self.activation,
            'kl_weight' : self.kl_weight,
            'num_image_samples' : self.num_image_samples,
            'share_weights' : self.share_weights,
            'prior_name' : self.prior_name,
            'posterior_name' : self.posterior_name,
            'background_prior_name' : self.background_prior_name,
            'background_posterior_name' : self.background_posterior_name,
            'bijector_name' : self.bijector_name,
            'normalizer_name' : self.normalizer_name,
            'hkl_to_imodel' : self.hkl_to_imodel,
            'gated' : self.gated,
            'output_bias' : self.output_bias, 
            'dropout': self.dropout,
            'random_seed': self.random_seed,
            'batch_normalize' : self.batch_normalize,
        })
        return config

    @staticmethod
    def sample_refls(tensor, mc_samples):
        n = tf.shape(tensor)[0]
        l = tf.reduce_sum(tf.ones_like(tensor[...,0]), axis=-1)
        idx = tf.random.uniform(
            (n, mc_samples),
            minval=-0.5,
            maxval=l[:,None]-0.5,
        )
        idx = tf.cast(tf.round(idx), 'int32')
        out = tf.gather(tensor, idx, axis=1, batch_dims=1)
        return out

    #def prior_function(self):
    #    return self.prior_dict[self.prior_name]()

    def bijector_function(self, x):
        return self.bijector_dict[self.bijector_name](x) + self.epsilon

    def distribution_function(self, output):
        return self.posterior_dict[self.posterior_name](output, self.bijector_function)

    def background_distribution_function(self, output):
        return self.posterior_dict[self.background_posterior_name](output, self.bijector_function)

    def build(self, shapes):
        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = shapes

        dimage = metadata[-1] + 2
        if self.hkl_to_imodel:
            dimage = dimage + 3
        self.input_image.build(
            metadata[:-1] + [dimage] #add columns for iobs/sigiobs
        )

        self.input_scale.build(metadata)
        self.image_network.build(metadata[:-1] + [self.mlp_width])
        if not self.share_weights:
            self.scale_network.build(metadata[:-1] + [self.mlp_width])
        self.pool.build(metadata[:-1] + [self.mlp_width])
        self.output_dense.build(metadata[:-1] + [self.mlp_width])
        self.built = True

    def compute_kl_terms(self, q, p, z, name='KL_Σ', reduction='mean'):
        try: #Attempt to calculate this analytically
            kl_terms = q.kl_divergence(p)
        except NotImplementedError:
            q_z = q.log_prob(z)
            #z = z + self.epsilon #<-- why?
            p_z = p.log_prob(z)
            kl_terms = q_z - p_z

        if reduction == 'mean':
            kl_div = tf.reduce_mean(kl_terms)
        elif reduction == 'sum':
            kl_div = tf.reduce_mean(kl_terms)
        else:
            kl_div = reduction(kl_terms)

        self.add_loss(self.kl_weight * kl_div)
        self.add_metric(kl_div, name=name)
        return kl_terms

    def call(self, inputs, mc_samples=32, training=None, **kwargs):
        (
            asu_id,
            hkl,
            resolution,
            wavelength,
            metadata,
            iobs,
            sigiobs,
        ) = inputs

        image = [metadata, iobs, sigiobs]
        n = 2
        if self.hkl_to_imodel:
            image.append(0.02 * tf.cast(hkl, metadata.dtype))
            n = n + 3

        image = tf.concat(image, axis=-1)
        if self.batch_normalize:
            image = instance_normalize(image)
        metadata = image[...,:-n]

        scale = metadata

        if self.num_image_samples is not None:
            #Subsample reflections per image 
            image = ImageScaler.sample_refls(image, self.num_image_samples)

        unpooled_image = self.input_image(image)
        image = tf.ragged.map_flat_values(self.image_network, unpooled_image)
        image = self.pool(image)

        scale_in = tf.ragged.map_flat_values(self.input_scale, scale) + image
        q_latent = tf.ragged.map_flat_values(self.scale_network, scale_in)
        q_params = tf.ragged.map_flat_values(self.output_dense, q_latent)

        b_params = tf.ragged.map_flat_values(self.background_dense, q_latent)

        if ragged_tensor.is_ragged(q_params):
            q = self.distribution_function(q_params.flat_values)
            b = self.background_distribution_function(b_params.flat_values)
        else:
            q = self.distribution_function(q_params)
            b = self.background_distribution_function(b_params)

        z = q.sample(mc_samples) 
        w = b.sample(mc_samples)

        if self.kl_weight > 0.:
            p = self.prior_dict[self.prior_name]()
            r = self.prior_dict[self.background_prior_name]()
            self.compute_kl_terms(q, p, z, name='KL_Σ')
            self.compute_kl_terms(b, r, w, name='KL_B')

        z = tf.RaggedTensor.from_row_splits(
            tf.transpose(z), metadata.row_splits
        )
        w = tf.RaggedTensor.from_row_splits(
            tf.transpose(w), metadata.row_splits
        )

        self.add_metric(tf.math.reduce_mean(z), name='Σ_mean')
        self.add_metric(tf.math.reduce_std(z), name='Σ_std')
        self.add_metric(tf.math.reduce_mean(w), name='B_mean')
        self.add_metric(tf.math.reduce_std(w), name='B_std')
        return z, w

