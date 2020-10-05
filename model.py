import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPool2D, UpSampling2D, Add, Concatenate, Conv2DTranspose, AveragePooling2D, Multiply, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, ResNet50V2, ResNet101V2, MobileNetV2, EfficientNetB6, EfficientNetB7
from configs import *
from loss import *
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow.keras.metrics import Recall, Precision, AUC, Accuracy

def aspp_block(x):
	filters = x.shape[-1]
	rates = [1, 2, 3, 4]  ###
	kernels = [1]+ [3]*(len(rates)-1)
	pyramid = []
	for i in range(len(rates)):
		p = Conv2D(filters=filters, kernel_size=kernels[i], padding='same', dilation_rate=rates[i])(x)
		p = BatchNormalization()(p)
		p = Activation('relu')(p)
		pyramid.append(p)

	ave = AveragePooling2D(pool_size=2)(x)
	ave = Conv2D(filters=filters, kernel_size=1)(ave)
	ave = BatchNormalization()(ave)
	ave = Activation('relu')(ave)
	ave_features = UpSampling2D(interpolation='bilinear')(ave)

	aspp = Concatenate()(pyramid + [ave_features])
	aspp = Conv2D(filters=filters, kernel_size=1, padding='same')(aspp)
	aspp = BatchNormalization()(aspp)
	aspp = Activation('relu')(aspp)

	return Add()([aspp, x])


def atrous_block(x):
	filters = x.shape[-1]
	rates = [1, 2, 4, 8]  #####
	kernels = [1, 3, 3, 3]
	pyramid = []
	for i in range(len(rates)):
		p = Conv2D(filters=filters, kernel_size=kernels[i], padding='same', dilation_rate=rates[i])(x)
		p = BatchNormalization()(p)
		p = Activation('relu')(p)
		pyramid.append(p)

	atrs = Concatenate()(pyramid)
	atrs = Conv2D(filters=filters, kernel_size=1, padding='same')(atrs)
	atrs = BatchNormalization()(atrs)
	atrs = Activation('relu')(atrs)

	return Add()([x, atrs])


def dense_atrous_block(x):
	kernels = [3, 3, 3, 1]
	rates = [1, 3, 5, 1]
	filters = x.shape[-1]
	d = []

	for i in range(len(kernels)):
		dilate = Conv2D(filters=filters, kernel_size=kernels[i], padding='same', dilation_rate=rates[i])
		d.append(dilate)

	d1 = d[0](x)
	d1 = BatchNormalization()(d1)
	d1 = Activation('relu')(d1)

	d2 = d[3](d[1](x))
	d2 = BatchNormalization()(d2)
	d2 = Activation('relu')(d2)

	d3 = d[3](d[1](d[0](x)))
	d3 = BatchNormalization()(d3)
	d3 = Activation('relu')(d3)

	d4 = d[3](d[2](d[1](d[0](x))))
	d4 = BatchNormalization()(d4)
	d4 = Activation('relu')(d4)

	out = Add()([x, d1, d2, d3, d4])
	return out


def se_block(x, ratio=8):  # Squeeze-and-Excitation
	filters = x.shape[-1]
	se_shape = (1, 1, filters)
	se = GlobalAveragePooling2D()(x)
	se = Reshape(se_shape)(se)
	se = Dense(filters // ratio)(se)
	se = Activation('relu')(se)
	se = Dense(filters)(se)
	se = Activation('sigmoid')(se)
	se = Multiply()([x, se])
	return se


def cbam_block(x, ratio=8):
	# channel_attention

	filters = x.shape[-1]

	avg_pool_ch = GlobalAveragePooling2D()(x)
	avg_pool_ch = Reshape((1, 1, filters))(avg_pool_ch)
	avg_pool_ch = Dense(filters // ratio)(avg_pool_ch)
	avg_pool_ch = Activation('relu')(avg_pool_ch)
	avg_pool_ch = Dense(filters)(avg_pool_ch)

	max_pool_ch = GlobalMaxPooling2D()(x)
	max_pool_ch = Reshape((1, 1, filters))(max_pool_ch)
	max_pool_ch = Dense(filters // ratio)(max_pool_ch)
	max_pool_ch = Activation('relu')(max_pool_ch)
	max_pool_ch = Dense(filters)(max_pool_ch)

	cbam_ch = Add()([avg_pool_ch, max_pool_ch])
	cbam_ch = Activation('sigmoid')(cbam_ch)

	cbam_ch = Multiply()([x, cbam_ch])

	# spatial_attention
	kernel_size = 7
	avg_pool_s = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_ch)
	max_pool_s = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_ch)
	concat = Concatenate(axis=3)([avg_pool_s, max_pool_s])
	cbam_s = Conv2D(filters=1, kernel_size=kernel_size, strides=1, padding='same')(concat)
	cbam_s = Activation('sigmoid')(cbam_s)

	cbam = Multiply()([cbam_ch, cbam_s])

	return cbam


def scSE_block(x, ratio=8):  # Concurrent Spatial and Channel Squeeze & Excitation
	filters = x.shape[-1]
	shape = (1, 1, filters)
	# cSE (Spatial Squeeze and Channel Excitation Block)
	cSE = GlobalAveragePooling2D()(x)
	cSE = Reshape(shape)(cSE)
	cSE = Dense(filters // ratio)(cSE)
	cSE = Activation('relu')(cSE)
	cSE = Dense(filters)(cSE)
	cSE = Activation('sigmoid')(cSE)
	cSE = Multiply()([x, cSE])

	# sSE (Channel Squeeze and Spatial Excitation)
	sSE = Conv2D(1, 1, padding='same')(x)
	sSE = Activation('sigmoid')(sSE)
	sSE = Multiply()([x, sSE])

	# scSE (Concurrent Spatial and Channel Squeeze & Excitation)
	scSE = Add()([cSE, sSE])
	return scSE


# skip connection blocks
def down_skip(skip, down_scale, filters=64):
	dn = MaxPool2D(pool_size=down_scale, padding='same')(skip)
	dn = Conv2D(filters=filters, kernel_size=3, padding='same')(dn)
	dn = BatchNormalization()(dn)
	dn = Activation('relu')(dn)
	return dn


def direct_skip(skip, filters=64, kernel_size=3):
	dr = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same')(skip)
	dr = BatchNormalization()(dr)
	dr = Activation('relu')(dr)
	return dr


	# decoder blocks
def conv3_block(x, filters):
	x = Conv2D(filters, 3, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def conv1_block(x, filters):
	x = Conv2D(filters, 1, padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def convtranspose_block(x, filters):
	x = Conv2DTranspose(filters, 3, 2, 'same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x


def upscore_block(x, filters, scale):
	x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
	x = UpSampling2D(interpolation='bilinear', size=scale)(x)
	return x


def build_model(encoder='efficientnetb7', center='dac', full_skip=True, attention='sc', upscore='upall'):

	MODEL_NAME = encoder
	if center is not None:
		MODEL_NAME = MODEL_NAME+'_'+center
	if attention is not None:
		MODEL_NAME = MODEL_NAME+'_'+attention
	if full_skip:
		MODEL_NAME = MODEL_NAME + '_fullskip'
	if upscore is not None:
		MODEL_NAME = MODEL_NAME + '_'+upscore


	if encoder == 'resnet50':
		encoder = ResNet50(input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data'), weights='imagenet', include_top=False)
		skip_names = ['data', 'conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
		encoder_output = encoder.get_layer('conv5_block3_out').output
		# data    320x320x3
		# conv1_relu    160x160x64
		# conv2_block3_out     80x80x256
		# conv3_block4_out    40x40x512
		# conv4_block6_out    20x20x1024
		# conv5_block3_out    10x10x2048  --> encoder output

	elif encoder == 'resnet101':
		encoder = ResNet101(input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data'), weights='imagenet', include_top=False)
		skip_names = ['data', 'conv1_relu', 'conv2_block3_out', 'conv3_block4_out']
		encoder_output = encoder.get_layer('conv4_block23_out').output
		#data   320x320x3
		#conv1_relu   160x160x64
		#conv2_block3_out   80x80x256
		#conv3_block4_out    40x40x512
		#conv4_block23_out   20x20x1024 --> encoder output
		#conv5_block3_out  10x10x2048

	elif encoder == 'resnet50v2':
		encoder = ResNet50V2(input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data'), weights='imagenet', include_top=False)
		skip_names = ['data', 'conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu']
		encoder_output = encoder.get_layer('post_relu').output
		# data   320x320x3
		# conv1_conv   160x160x64
		# conv2_block3_1_relu   80x80x64
		# conv3_block4_1_relu   40x40x128
		# conv4_block6_1_relu   20x20x256
		# post_relu   10x10x2048  --> encoder output

	elif encoder == 'resnet101v2':
		encoder = ResNet101V2(input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data'), weights='imagenet', include_top=False)
		skip_names = ['data', 'conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu']
		encoder_output = encoder.get_layer('post_relu').output
		#data   320x320x3
		#conv1_conv   160x160x64
		#conv2_block3_1_relu   80x80x64
		#conv3_block4_1_relu    40x40x128
		#conv4_block23_1_relu   20x20x256 
		#post_relu  10x10x2048 --> encoder output

	elif encoder == 'vgg19':
		encoder = VGG19(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), weights='imagenet', include_top=False)
		skip_names = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4']
		encoder_output = encoder.get_layer('block5_pool').output
		# block1_conv2   320x320x64
		# block2_conv2   160x160x128
		# block3_conv4   80x80x256
		# block4_conv4   40x40x512
		# block5_conv4   20x20x512
		# block5_pool   10x10x512   --> encoder output

	elif encoder == 'efficientnetb6':
		encoder = EfficientNetB6(input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data'), weights='imagenet', include_top=False)
		skip_names = ['data', 'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation']
		encoder_output = encoder.get_layer('block6a_expand_activation').output
		#data   320x320x3
		#block2a_expand_activation   160x160x192
		#block3a_expand_activation   80x80x240
		#block4a_expand_activation    40x40x432
		#block6a_expand_activation   20x20x1200 --> encoder output
		#top_activation   10x10x2304

	elif encoder == 'efficientnetb7':
		encoder = EfficientNetB7(input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data'), weights='imagenet', include_top=False)
		skip_names = ['data', 'block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation']
		encoder_output = encoder.get_layer('block6a_expand_activation').output
		#data   320x320x3
		#block2a_expand_activation   160x160x192
		#block3a_expand_activation   80x80x288
		#block4a_expand_activation    40x40x480
		#block6a_expand_activation   20x20x1344 --> encoder output
		#top_activation   10x10x

	elif encoder == 'mobilenetv2':
		encoder = MobileNetV2(input_tensor=Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='data'), weights='imagenet', include_top=False)
		skip_names = ['data', 'block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu']
		encoder_output = encoder.get_layer('out_relu').output
		# data   320x320x3
		# block_1_expand_relu   160x160x96
		# block_3_expand_relu   80x80x144
		# block_6_expand_relu    40x40x192
		# block_13_expand_relu   20x20x576
		# out_relu   10x10x1248   --> encoder output

	skip_layers = [encoder.get_layer(i).output for i in skip_names]
	# Center --------------
	if center == 'atrous':
		x = atrous_block(encoder_output)
	elif center == 'dac':
		x = dense_atrous_block(encoder_output)
	elif center == 'aspp':
		x = aspp_block(encoder_output)
	elif center is None:
		x = encoder_output

    # Decoder --------------
	if attention == 'se':
		attn_block = se_block
	elif attention == 'cbam':
		attn_block = cbam_block
	elif attention == 'sc':
		attn_block = scSE_block

	filters = [i.shape[-1] for i in skip_layers]
	filters[0] = 64

	scales = [2 ** i for i in range(1, len(filters))][::-1]
	X = []
	for i in range(1, len(filters) + 1):
		X.append(x)

		down = []
		if full_skip:
			for j in range(len(scales) - (i - 1)):
				d = down_skip(skip_layers[j], scales[j + (i - 1)], filters[-1]//4)
				if attention is not None:
					d = attn_block(d) 
				down.append(d)


		direct = direct_skip(skip_layers[-i], filters[-1]//4)
		if attention is not None:
			direct = attn_block(direct)


		x = convtranspose_block(x, filters[-1]//4)
		if attention is not None:
			x = attn_block(x)

		x = Concatenate()([x] + [direct] + down)
		
		x = conv3_block(x, x.shape[-1])

	if upscore is not None:
		if upscore=='upall':
			up_scales=[2 ** i for i in range(1, len(filters)+1)][::-1]
			UP = [upscore_block(x, 32, up_scales[i]) for i, x in enumerate(X)]
			if attention is not None:
				UP = [attn_block(x) for x in UP]

			up = Concatenate()(UP)
     
		elif upscore=='upcenter':
			up = upscore_block(X[0], 64, 2 ** len(filters))
			if attention is not None:
				up = attn_block(up)

		x = Concatenate()([x, up])


	x = Conv2D(1, 1, padding='same')(x)
	x = Activation('sigmoid')(x)

	model = Model(encoder.input, x)

	metrics = [dice_coef, Recall(), Precision()]
	opt = Nadam(LR)
	model.compile(loss=bce_dice_loss, optimizer=opt, metrics=metrics)

	return model, MODEL_NAME

############# UNET #################
def unet():
	model_name = 'U-Net'
	def conv_block(x, filters):
		x = Conv2D(filters, 3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

		x = Conv2D(filters, 3, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

		return x

	f = [64, 128, 256, 512]
	inputs = Input((IMAGE_SIZE, IMAGE_SIZE, 3))

	skips = []
	x = inputs

	for i in f:
		x = conv_block(x, i)
		skips.append(x)
		x = MaxPool2D(2)(x)

	x = conv_block(x, 1024)

	f.reverse()
	skips.reverse()
	for i, fi in enumerate(f):
		x = UpSampling2D(2)(x)
		skip = skips[i]
		x = Concatenate()([x, skip])
		x = conv_block(x, fi)

	x = Conv2D(1, 1, padding='same')(x)
	x = Activation('sigmoid')(x)

	model = Model(inputs, x)
	metrics = [dice_coef, Recall(), Precision(), AUC()]
	opt = Nadam(LR)
	model.compile(loss=bce_dice_loss, optimizer=opt, metrics=metrics)
	return model, model_name


