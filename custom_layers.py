# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K
from keras import activations, initializers, regularizers
from keras.engine import Layer, InputSpec
from keras.layers import Dense, Recurrent
from keras.layers.merge import _Merge
from keras.engine.topology import Layer

import theano
import theano.tensor as T


class DenseNonNegW(Dense):
    """Equivalent to a Dense layer with a differential elementwise 
       nonnegative constraint on the kernel by using K.exp(kernel)
       during forward pass.
       Thus, to initialize the kernel to a known nonnegative matrix
       A, the kernel should be initialized with log(eps + A), where
       eps is a small value like 1e-7 to prevent NaNs.
    """
    def call(self, inputs):
        output = K.dot(inputs, K.exp(self.kernel))
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output 



class DivideAbyAplusB(_Merge):
    """Layer that divides (element-wise) the first input by the 
    elementwise sum of the first input and second input.
    It takes as input a list of tensors of len 2, all of the 
    same shape, and returns a single tensor (also of the same 
    shape).
    """

    def _merge_function(self, inputs):
        A = inputs[0]
        B = inputs[1]
        output = K.exp( K.log(1e-7 + A) - K.log(1e-7 + A + B) )
        return output


def divide_A_by_AplusB(inputs, **kwargs):
    """Functional interface to the `DivideAbyAplusB` layer.
    # Arguments
        inputs: A list of input tensors (length exactly 2).
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, equal to A/(A+B).
    """
    return DivideAbyAplusB(**kwargs)(inputs)


def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x


class SimpleDeepRNN(Recurrent):
    '''Fully-connected RNN where the output is to be fed back to input, and
    each time step has a K_layer-deep network at each time step.
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializers](../initializers.md)).
        inner_init: initialization function of the inner cells.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        K_layers: depth of deep network at each time step
        alt_params: dictionary of alternate parameters
        maps_from_alt: dictionary of maps from alternate parameters to RNN parameters. May only contain keys 'U','W','S','h0', or 'b'. If the value corresponding to a key is a list, then the list must be of length K_layers and specifies layer-dependent maps.
        flag_connect_input_to_layers: if set, add residual connections from input to every deep layer in each time step
    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 activation='tanh',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., 
                 K_layers=1,
                 alt_params=None,
                 keys_trainable=None,
                 maps_from_alt=None,
                 flag_connect_input_to_layers=False,
                 flag_nonnegative=False,
                 flag_return_all_hidden=False,
                 **kwargs):
        self.units = output_dim
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.inner_init = initializers.get(inner_init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self.K_layers = K_layers
        if alt_params is None:
            alt_params = {}
        self.alt_params=alt_params
        if keys_trainable is None:
            self.keys_trainable = alt_params.keys()
        else:
            self.keys_trainable = keys_trainable
        if maps_from_alt is None:
            maps_from_alt={}
        self.maps_from_alt=maps_from_alt
        self.flag_connect_input_to_layers = flag_connect_input_to_layers
        self.flag_nonnegative = flag_nonnegative
        self.flag_return_all_hidden = flag_return_all_hidden

        self.consume_less='gpu'

        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        super(SimpleDeepRNN, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        if self.flag_return_all_hidden:
            units = self.K_layers*self.units
        else:
            units = self.units
        if self.return_sequences:
            return (input_shape[0], input_shape[1], units)
        else:
            return (input_shape[0], units)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim


        if self.flag_return_all_hidden:
            output_dim_h0 = self.K_layers*self.output_dim
        else:
            output_dim_h0 = self.output_dim
        if self.flag_nonnegative:
            self.log_h0 = self.add_weight((self.output_dim,),
                                          initializer='uniform',
                                          name='{}_log_h0'.format(self.name))
            self.h0_last = K.softplus(self.log_h0)
        else:
            self.h0_last = self.add_weight((self.output_dim,),
                                     initializer='zero',
                                     name='{}_h0'.format(self.name))
        if self.flag_return_all_hidden:
            self.h0 = K.tile(self.h0_last, [self.K_layers,])
        else:
            self.h0 = self.h0_last

        for key in self.alt_params:
            param=self.alt_params[key]
            if key in self.keys_trainable:
                flag_trainable=True
            else:
                flag_trainable=False
            pcur = self.add_weight(param.shape,
                                   initializer='zero',
                                   trainable=flag_trainable,
                                   name=('{}_%s' % key).format(self.name))
            pcur.set_value(param)
            #setattr(self, key, pcur)
            self.alt_params[key]=pcur

        self.Wk=[]
        self.Uk=[]
        self.bk=[]
        self.Sk=[]
        for k in range(self.K_layers):
            if ('W' in self.maps_from_alt):
                if isinstance(self.maps_from_alt['W'],list):
                    map_cur=self.maps_from_alt['W'][k]
                else:
                    map_cur=self.maps_from_alt['W']
                Wcur = map_cur(self.alt_params)
            else:
                Wcur = self.add_weight((input_dim, self.output_dim),
                                         initializer=self.init,
                                         name=('{}_W_%d' % k).format(self.name),
                                         regularizer=self.W_regularizer)
            
            if ('U' in self.maps_from_alt):
                if isinstance(self.maps_from_alt['U'],list):
                    map_cur=self.maps_from_alt['U'][k]
                else:
                    map_cur=self.maps_from_alt['U']
                Ucur = map_cur(self.alt_params)
            else:
                Ucur = self.add_weight((self.output_dim, self.output_dim),
                                         initializer=self.inner_init,
                                         name=('{}_U_%d' % k).format(self.name),
                                         regularizer=self.U_regularizer)
            
            if ('b' in self.maps_from_alt):
                if isinstance(self.maps_from_alt['b'],list):
                    map_cur=self.maps_from_alt['b'][k]
                else:
                    map_cur=self.maps_from_alt['b']
                bcur = map_cur(self.alt_params)
            else:
                bcur = self.add_weight((self.output_dim,),
                                         initializer='zero',
                                         name=('{}_b_%d' % k).format(self.name),
                                         regularizer=self.b_regularizer)
            
            self.Wk.append(Wcur)
            self.Uk.append(Ucur)
            self.bk.append(bcur)
            
            if k>0:
                if ('S' in self.maps_from_alt):
                    if isinstance(self.maps_from_alt['S'],list):
                        map_cur=self.maps_from_alt['S'][k-1]
                    else:
                        map_cur=self.maps_from_alt['S']
                    Scur = map_cur(self.alt_params)
                else:
                    Scur = self.add_weight((self.output_dim, self.output_dim),
                                             initializer=self.inner_init,
                                             name=('{}_S_%dto%d' % (k-1,k)).format(self.name))
                
                self.Sk.append(Scur)
        
        """
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        """
        self.built = True

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a '
                             '`batch_shape` argument to your Input layer.')
        if hasattr(self, 'states'):
            if self.flag_return_all_hidden:
                output_dim = self.K_layers*self.output_dim
            else:
                output_dim = self.output_dim
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], output_dim))]
 
    def preprocess_input(self, x, training=None):
        """
        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.Wk[0], self.bk[0], self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
        else:
            return x
        """
        return x

    # override Recurrent's get_initial_states function to load the trainable
    # initial hidden state
    def get_initial_states(self, x):
            initial_state = K.expand_dims(self.h0, 0) # (1, output_dim)
            initial_state = K.tile(initial_state, [x.shape[0], 1])  # (samples, output_dim)
            #initial_states = [initial_state for _ in range(len(self.states))]
            initial_states = [initial_state]
            return initial_states
    
    def step(self, x, states):
        if self.flag_return_all_hidden:
            # use the last hidden state in the stack from previous time step:
            prev_output = states[0][:, -self.output_dim:]
        else:
            prev_output = states[0]
        B_U = states[1]
        B_W = states[2]

        """
        if self.consume_less == 'cpu':
            Wx = x
        else:
            Wx = K.dot(x * B_W, self.W) + self.b
        """

        preact=[]
        hidden=[]
        for k in range(self.K_layers):
            preact.append( K.dot(prev_output * B_U, self.Uk[k]) )
            if k>0:
                # add in the output from layer k-1
                preact[k] += K.dot(hidden[k-1], self.Sk[k-1])
            if self.flag_connect_input_to_layers:
                # add in the transformed input (residual connection)
                preact[k] += K.dot(x * B_W, self.Wk[k])
            hidden.append( self.activation(preact[k] + self.bk[k]) )

        if self.flag_return_all_hidden:
            output = K.concatenate(hidden, axis=1)
        else:
            output = hidden[-1]
        return output, [output]

    def get_constants(self, x, training=None):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = K.int_shape(x)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'K_layers' : self.K_layers,
                  #'alt_params' : self.alt_params,
                  #'maps_from_alt' : self.maps_from_alt,
                  'flag_connect_input_to_layers' : self.flag_connect_input_to_layers}
        base_config = super(SimpleDeepRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
