??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8??
~
dense_9004/kernelVarHandleOp*
shape
:@*"
shared_namedense_9004/kernel*
dtype0*
_output_shapes
: 
w
%dense_9004/kernel/Read/ReadVariableOpReadVariableOpdense_9004/kernel*
dtype0*
_output_shapes

:@
v
dense_9004/biasVarHandleOp*
shape:@* 
shared_namedense_9004/bias*
dtype0*
_output_shapes
: 
o
#dense_9004/bias/Read/ReadVariableOpReadVariableOpdense_9004/bias*
dtype0*
_output_shapes
:@
~
dense_9005/kernelVarHandleOp*
shape
:@@*"
shared_namedense_9005/kernel*
dtype0*
_output_shapes
: 
w
%dense_9005/kernel/Read/ReadVariableOpReadVariableOpdense_9005/kernel*
dtype0*
_output_shapes

:@@
v
dense_9005/biasVarHandleOp*
shape:@* 
shared_namedense_9005/bias*
dtype0*
_output_shapes
: 
o
#dense_9005/bias/Read/ReadVariableOpReadVariableOpdense_9005/bias*
dtype0*
_output_shapes
:@
~
dense_9006/kernelVarHandleOp*
shape
:@*"
shared_namedense_9006/kernel*
dtype0*
_output_shapes
: 
w
%dense_9006/kernel/Read/ReadVariableOpReadVariableOpdense_9006/kernel*
dtype0*
_output_shapes

:@
v
dense_9006/biasVarHandleOp*
shape:* 
shared_namedense_9006/bias*
dtype0*
_output_shapes
: 
o
#dense_9006/bias/Read/ReadVariableOpReadVariableOpdense_9006/bias*
dtype0*
_output_shapes
:
h

Nadam/iterVarHandleOp*
shape: *
shared_name
Nadam/iter*
dtype0	*
_output_shapes
: 
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
dtype0	*
_output_shapes
: 
l
Nadam/beta_1VarHandleOp*
shape: *
shared_nameNadam/beta_1*
dtype0*
_output_shapes
: 
e
 Nadam/beta_1/Read/ReadVariableOpReadVariableOpNadam/beta_1*
dtype0*
_output_shapes
: 
l
Nadam/beta_2VarHandleOp*
shape: *
shared_nameNadam/beta_2*
dtype0*
_output_shapes
: 
e
 Nadam/beta_2/Read/ReadVariableOpReadVariableOpNadam/beta_2*
dtype0*
_output_shapes
: 
j
Nadam/decayVarHandleOp*
shape: *
shared_nameNadam/decay*
dtype0*
_output_shapes
: 
c
Nadam/decay/Read/ReadVariableOpReadVariableOpNadam/decay*
dtype0*
_output_shapes
: 
z
Nadam/learning_rateVarHandleOp*
shape: *$
shared_nameNadam/learning_rate*
dtype0*
_output_shapes
: 
s
'Nadam/learning_rate/Read/ReadVariableOpReadVariableOpNadam/learning_rate*
dtype0*
_output_shapes
: 
|
Nadam/momentum_cacheVarHandleOp*
shape: *%
shared_nameNadam/momentum_cache*
dtype0*
_output_shapes
: 
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
?
Nadam/dense_9004/kernel/mVarHandleOp*
shape
:@**
shared_nameNadam/dense_9004/kernel/m*
dtype0*
_output_shapes
: 
?
-Nadam/dense_9004/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_9004/kernel/m*
dtype0*
_output_shapes

:@
?
Nadam/dense_9004/bias/mVarHandleOp*
shape:@*(
shared_nameNadam/dense_9004/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_9004/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_9004/bias/m*
dtype0*
_output_shapes
:@
?
Nadam/dense_9005/kernel/mVarHandleOp*
shape
:@@**
shared_nameNadam/dense_9005/kernel/m*
dtype0*
_output_shapes
: 
?
-Nadam/dense_9005/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_9005/kernel/m*
dtype0*
_output_shapes

:@@
?
Nadam/dense_9005/bias/mVarHandleOp*
shape:@*(
shared_nameNadam/dense_9005/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_9005/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_9005/bias/m*
dtype0*
_output_shapes
:@
?
Nadam/dense_9006/kernel/mVarHandleOp*
shape
:@**
shared_nameNadam/dense_9006/kernel/m*
dtype0*
_output_shapes
: 
?
-Nadam/dense_9006/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_9006/kernel/m*
dtype0*
_output_shapes

:@
?
Nadam/dense_9006/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_9006/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_9006/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_9006/bias/m*
dtype0*
_output_shapes
:
?
Nadam/dense_9004/kernel/vVarHandleOp*
shape
:@**
shared_nameNadam/dense_9004/kernel/v*
dtype0*
_output_shapes
: 
?
-Nadam/dense_9004/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_9004/kernel/v*
dtype0*
_output_shapes

:@
?
Nadam/dense_9004/bias/vVarHandleOp*
shape:@*(
shared_nameNadam/dense_9004/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_9004/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_9004/bias/v*
dtype0*
_output_shapes
:@
?
Nadam/dense_9005/kernel/vVarHandleOp*
shape
:@@**
shared_nameNadam/dense_9005/kernel/v*
dtype0*
_output_shapes
: 
?
-Nadam/dense_9005/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_9005/kernel/v*
dtype0*
_output_shapes

:@@
?
Nadam/dense_9005/bias/vVarHandleOp*
shape:@*(
shared_nameNadam/dense_9005/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_9005/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_9005/bias/v*
dtype0*
_output_shapes
:@
?
Nadam/dense_9006/kernel/vVarHandleOp*
shape
:@**
shared_nameNadam/dense_9006/kernel/v*
dtype0*
_output_shapes
: 
?
-Nadam/dense_9006/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_9006/kernel/v*
dtype0*
_output_shapes

:@
?
Nadam/dense_9006/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_9006/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_9006/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_9006/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
?)
ConstConst"/device:CPU:0*?)
value?)B?) B?)
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate
+momentum_cachemPmQmRmS mT!mUvVvWvXvY vZ!v[
 
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
?
regularization_losses

,layers
	variables
-metrics
.layer_regularization_losses
	trainable_variables
/non_trainable_variables
 
 
 
 
?
regularization_losses

0layers
	variables
1metrics
2layer_regularization_losses
trainable_variables
3non_trainable_variables
][
VARIABLE_VALUEdense_9004/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9004/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

4layers
	variables
5metrics
6layer_regularization_losses
trainable_variables
7non_trainable_variables
][
VARIABLE_VALUEdense_9005/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9005/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

8layers
	variables
9metrics
:layer_regularization_losses
trainable_variables
;non_trainable_variables
 
 
 
?
regularization_losses

<layers
	variables
=metrics
>layer_regularization_losses
trainable_variables
?non_trainable_variables
][
VARIABLE_VALUEdense_9006/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9006/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
?
"regularization_losses

@layers
#	variables
Ametrics
Blayer_regularization_losses
$trainable_variables
Cnon_trainable_variables
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUENadam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUENadam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUENadam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

D0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
	Etotal
	Fcount
G
_fn_kwargs
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

E0
F1
 
?
Hregularization_losses

Llayers
I	variables
Mmetrics
Nlayer_regularization_losses
Jtrainable_variables
Onon_trainable_variables
 
 
 

E0
F1
?
VARIABLE_VALUENadam/dense_9004/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_9004/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/dense_9005/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_9005/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/dense_9006/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_9006/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/dense_9004/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_9004/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/dense_9005/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_9005/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUENadam/dense_9006/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_9006/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
?
 serving_default_dense_9004_inputPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_9004_inputdense_9004/kerneldense_9004/biasdense_9005/kerneldense_9005/biasdense_9006/kerneldense_9006/bias*/
_gradient_op_typePartitionedCall-15308947*/
f*R(
&__inference_signature_wrapper_15308711*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:?????????
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_9004/kernel/Read/ReadVariableOp#dense_9004/bias/Read/ReadVariableOp%dense_9005/kernel/Read/ReadVariableOp#dense_9005/bias/Read/ReadVariableOp%dense_9006/kernel/Read/ReadVariableOp#dense_9006/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Nadam/dense_9004/kernel/m/Read/ReadVariableOp+Nadam/dense_9004/bias/m/Read/ReadVariableOp-Nadam/dense_9005/kernel/m/Read/ReadVariableOp+Nadam/dense_9005/bias/m/Read/ReadVariableOp-Nadam/dense_9006/kernel/m/Read/ReadVariableOp+Nadam/dense_9006/bias/m/Read/ReadVariableOp-Nadam/dense_9004/kernel/v/Read/ReadVariableOp+Nadam/dense_9004/bias/v/Read/ReadVariableOp-Nadam/dense_9005/kernel/v/Read/ReadVariableOp+Nadam/dense_9005/bias/v/Read/ReadVariableOp-Nadam/dense_9006/kernel/v/Read/ReadVariableOp+Nadam/dense_9006/bias/v/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-15308995**
f%R#
!__inference__traced_save_15308994*
Tout
2**
config_proto

CPU

GPU 2J 8*'
Tin 
2	*
_output_shapes
: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9004/kerneldense_9004/biasdense_9005/kerneldense_9005/biasdense_9006/kerneldense_9006/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/dense_9004/kernel/mNadam/dense_9004/bias/mNadam/dense_9005/kernel/mNadam/dense_9005/bias/mNadam/dense_9006/kernel/mNadam/dense_9006/bias/mNadam/dense_9004/kernel/vNadam/dense_9004/bias/vNadam/dense_9005/kernel/vNadam/dense_9005/bias/vNadam/dense_9006/kernel/vNadam/dense_9006/bias/v*/
_gradient_op_typePartitionedCall-15309086*-
f(R&
$__inference__traced_restore_15309085*
Tout
2**
config_proto

CPU

GPU 2J 8*&
Tin
2*
_output_shapes
: ԝ
?/
?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308754

inputs-
)dense_9004_matmul_readvariableop_resource.
*dense_9004_biasadd_readvariableop_resource-
)dense_9005_matmul_readvariableop_resource.
*dense_9005_biasadd_readvariableop_resource-
)dense_9006_matmul_readvariableop_resource.
*dense_9006_biasadd_readvariableop_resource
identity??!dense_9004/BiasAdd/ReadVariableOp? dense_9004/MatMul/ReadVariableOp?!dense_9005/BiasAdd/ReadVariableOp? dense_9005/MatMul/ReadVariableOp?!dense_9006/BiasAdd/ReadVariableOp? dense_9006/MatMul/ReadVariableOp?
 dense_9004/MatMul/ReadVariableOpReadVariableOp)dense_9004_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@
dense_9004/MatMulMatMulinputs(dense_9004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_9004/BiasAdd/ReadVariableOpReadVariableOp*dense_9004_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_9004/BiasAddBiasAdddense_9004/MatMul:product:0)dense_9004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_9004/ReluReludense_9004/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
 dense_9005/MatMul/ReadVariableOpReadVariableOp)dense_9005_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
dense_9005/MatMulMatMuldense_9004/Relu:activations:0(dense_9005/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_9005/BiasAdd/ReadVariableOpReadVariableOp*dense_9005_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_9005/BiasAddBiasAdddense_9005/MatMul:product:0)dense_9005/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_9005/ReluReludense_9005/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@^
dropout_3591/dropout/rateConst*
valueB
 *  ?>*
dtype0*
_output_shapes
: g
dropout_3591/dropout/ShapeShapedense_9005/Relu:activations:0*
T0*
_output_shapes
:l
'dropout_3591/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_3591/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
1dropout_3591/dropout/random_uniform/RandomUniformRandomUniform#dropout_3591/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:?????????@?
'dropout_3591/dropout/random_uniform/subSub0dropout_3591/dropout/random_uniform/max:output:00dropout_3591/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
'dropout_3591/dropout/random_uniform/mulMul:dropout_3591/dropout/random_uniform/RandomUniform:output:0+dropout_3591/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@?
#dropout_3591/dropout/random_uniformAdd+dropout_3591/dropout/random_uniform/mul:z:00dropout_3591/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@_
dropout_3591/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_3591/dropout/subSub#dropout_3591/dropout/sub/x:output:0"dropout_3591/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_3591/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_3591/dropout/truedivRealDiv'dropout_3591/dropout/truediv/x:output:0dropout_3591/dropout/sub:z:0*
T0*
_output_shapes
: ?
!dropout_3591/dropout/GreaterEqualGreaterEqual'dropout_3591/dropout/random_uniform:z:0"dropout_3591/dropout/rate:output:0*
T0*'
_output_shapes
:?????????@?
dropout_3591/dropout/mulMuldense_9005/Relu:activations:0 dropout_3591/dropout/truediv:z:0*
T0*'
_output_shapes
:?????????@?
dropout_3591/dropout/CastCast%dropout_3591/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????@?
dropout_3591/dropout/mul_1Muldropout_3591/dropout/mul:z:0dropout_3591/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
 dense_9006/MatMul/ReadVariableOpReadVariableOp)dense_9006_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
dense_9006/MatMulMatMuldropout_3591/dropout/mul_1:z:0(dense_9006/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_9006/BiasAdd/ReadVariableOpReadVariableOp*dense_9006_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_9006/BiasAddBiasAdddense_9006/MatMul:product:0)dense_9006/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_9006/ReluReludense_9006/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_9006/Relu:activations:0"^dense_9004/BiasAdd/ReadVariableOp!^dense_9004/MatMul/ReadVariableOp"^dense_9005/BiasAdd/ReadVariableOp!^dense_9005/MatMul/ReadVariableOp"^dense_9006/BiasAdd/ReadVariableOp!^dense_9006/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_9005/MatMul/ReadVariableOp dense_9005/MatMul/ReadVariableOp2F
!dense_9006/BiasAdd/ReadVariableOp!dense_9006/BiasAdd/ReadVariableOp2F
!dense_9005/BiasAdd/ReadVariableOp!dense_9005/BiasAdd/ReadVariableOp2D
 dense_9004/MatMul/ReadVariableOp dense_9004/MatMul/ReadVariableOp2F
!dense_9004/BiasAdd/ReadVariableOp!dense_9004/BiasAdd/ReadVariableOp2D
 dense_9006/MatMul/ReadVariableOp dense_9006/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?
i
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308570

inputs
identity?Q
dropout/rateConst*
valueB
 *  ?>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:?????????@?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:?????????@a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?	
?
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308605

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
-__inference_dense_9006_layer_call_fn_15308891

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308611*Q
fLRJ
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308605*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308813

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308623
dense_9004_input-
)dense_9004_statefulpartitionedcall_args_1-
)dense_9004_statefulpartitionedcall_args_2-
)dense_9005_statefulpartitionedcall_args_1-
)dense_9005_statefulpartitionedcall_args_2-
)dense_9006_statefulpartitionedcall_args_1-
)dense_9006_statefulpartitionedcall_args_2
identity??"dense_9004/StatefulPartitionedCall?"dense_9005/StatefulPartitionedCall?"dense_9006/StatefulPartitionedCall?$dropout_3591/StatefulPartitionedCall?
"dense_9004/StatefulPartitionedCallStatefulPartitionedCalldense_9004_input)dense_9004_statefulpartitionedcall_args_1)dense_9004_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308511*Q
fLRJ
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308505*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9005/StatefulPartitionedCallStatefulPartitionedCall+dense_9004/StatefulPartitionedCall:output:0)dense_9005_statefulpartitionedcall_args_1)dense_9005_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308539*Q
fLRJ
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308533*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
$dropout_3591/StatefulPartitionedCallStatefulPartitionedCall+dense_9005/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-15308581*S
fNRL
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308570*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9006/StatefulPartitionedCallStatefulPartitionedCall-dropout_3591/StatefulPartitionedCall:output:0)dense_9006_statefulpartitionedcall_args_1)dense_9006_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308611*Q
fLRJ
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308605*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity+dense_9006/StatefulPartitionedCall:output:0#^dense_9004/StatefulPartitionedCall#^dense_9005/StatefulPartitionedCall#^dense_9006/StatefulPartitionedCall%^dropout_3591/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2L
$dropout_3591/StatefulPartitionedCall$dropout_3591/StatefulPartitionedCall2H
"dense_9004/StatefulPartitionedCall"dense_9004/StatefulPartitionedCall2H
"dense_9005/StatefulPartitionedCall"dense_9005/StatefulPartitionedCall2H
"dense_9006/StatefulPartitionedCall"dense_9006/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9004_input: : 
?
h
/__inference_dropout_3591_layer_call_fn_15308868

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*/
_gradient_op_typePartitionedCall-15308581*S
fNRL
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308570*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
?
K
/__inference_dropout_3591_layer_call_fn_15308873

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-15308589*S
fNRL
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308577*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308639
dense_9004_input-
)dense_9004_statefulpartitionedcall_args_1-
)dense_9004_statefulpartitionedcall_args_2-
)dense_9005_statefulpartitionedcall_args_1-
)dense_9005_statefulpartitionedcall_args_2-
)dense_9006_statefulpartitionedcall_args_1-
)dense_9006_statefulpartitionedcall_args_2
identity??"dense_9004/StatefulPartitionedCall?"dense_9005/StatefulPartitionedCall?"dense_9006/StatefulPartitionedCall?
"dense_9004/StatefulPartitionedCallStatefulPartitionedCalldense_9004_input)dense_9004_statefulpartitionedcall_args_1)dense_9004_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308511*Q
fLRJ
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308505*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9005/StatefulPartitionedCallStatefulPartitionedCall+dense_9004/StatefulPartitionedCall:output:0)dense_9005_statefulpartitionedcall_args_1)dense_9005_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308539*Q
fLRJ
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308533*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
dropout_3591/PartitionedCallPartitionedCall+dense_9005/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-15308589*S
fNRL
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308577*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9006/StatefulPartitionedCallStatefulPartitionedCall%dropout_3591/PartitionedCall:output:0)dense_9006_statefulpartitionedcall_args_1)dense_9006_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308611*Q
fLRJ
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308605*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity+dense_9006/StatefulPartitionedCall:output:0#^dense_9004/StatefulPartitionedCall#^dense_9005/StatefulPartitionedCall#^dense_9006/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2H
"dense_9004/StatefulPartitionedCall"dense_9004/StatefulPartitionedCall2H
"dense_9005/StatefulPartitionedCall"dense_9005/StatefulPartitionedCall2H
"dense_9006/StatefulPartitionedCall"dense_9006/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9004_input: : 
?	
?
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308884

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308831

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
-__inference_dense_9005_layer_call_fn_15308838

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308539*Q
fLRJ
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308533*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
h
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308577

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308656

inputs-
)dense_9004_statefulpartitionedcall_args_1-
)dense_9004_statefulpartitionedcall_args_2-
)dense_9005_statefulpartitionedcall_args_1-
)dense_9005_statefulpartitionedcall_args_2-
)dense_9006_statefulpartitionedcall_args_1-
)dense_9006_statefulpartitionedcall_args_2
identity??"dense_9004/StatefulPartitionedCall?"dense_9005/StatefulPartitionedCall?"dense_9006/StatefulPartitionedCall?$dropout_3591/StatefulPartitionedCall?
"dense_9004/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_9004_statefulpartitionedcall_args_1)dense_9004_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308511*Q
fLRJ
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308505*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9005/StatefulPartitionedCallStatefulPartitionedCall+dense_9004/StatefulPartitionedCall:output:0)dense_9005_statefulpartitionedcall_args_1)dense_9005_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308539*Q
fLRJ
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308533*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
$dropout_3591/StatefulPartitionedCallStatefulPartitionedCall+dense_9005/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-15308581*S
fNRL
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308570*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9006/StatefulPartitionedCallStatefulPartitionedCall-dropout_3591/StatefulPartitionedCall:output:0)dense_9006_statefulpartitionedcall_args_1)dense_9006_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308611*Q
fLRJ
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308605*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity+dense_9006/StatefulPartitionedCall:output:0#^dense_9004/StatefulPartitionedCall#^dense_9005/StatefulPartitionedCall#^dense_9006/StatefulPartitionedCall%^dropout_3591/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2L
$dropout_3591/StatefulPartitionedCall$dropout_3591/StatefulPartitionedCall2H
"dense_9004/StatefulPartitionedCall"dense_9004/StatefulPartitionedCall2H
"dense_9005/StatefulPartitionedCall"dense_9005/StatefulPartitionedCall2H
"dense_9006/StatefulPartitionedCall"dense_9006/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
2__inference_sequential_1812_layer_call_fn_15308666
dense_9004_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_9004_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-15308657*V
fQRO
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308656*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9004_input: : 
?
i
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308858

inputs
identity?Q
dropout/rateConst*
valueB
 *  ?>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:?????????@?
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@?
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@R
dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: ?
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:?????????@a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:?????????@o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????@i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?	
?
2__inference_sequential_1812_layer_call_fn_15308694
dense_9004_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_9004_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-15308685*V
fQRO
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308684*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9004_input: : 
?	
?
2__inference_sequential_1812_layer_call_fn_15308802

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-15308685*V
fQRO
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308684*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308533

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?'
?
#__inference__wrapped_model_15308488
dense_9004_input=
9sequential_1812_dense_9004_matmul_readvariableop_resource>
:sequential_1812_dense_9004_biasadd_readvariableop_resource=
9sequential_1812_dense_9005_matmul_readvariableop_resource>
:sequential_1812_dense_9005_biasadd_readvariableop_resource=
9sequential_1812_dense_9006_matmul_readvariableop_resource>
:sequential_1812_dense_9006_biasadd_readvariableop_resource
identity??1sequential_1812/dense_9004/BiasAdd/ReadVariableOp?0sequential_1812/dense_9004/MatMul/ReadVariableOp?1sequential_1812/dense_9005/BiasAdd/ReadVariableOp?0sequential_1812/dense_9005/MatMul/ReadVariableOp?1sequential_1812/dense_9006/BiasAdd/ReadVariableOp?0sequential_1812/dense_9006/MatMul/ReadVariableOp?
0sequential_1812/dense_9004/MatMul/ReadVariableOpReadVariableOp9sequential_1812_dense_9004_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
!sequential_1812/dense_9004/MatMulMatMuldense_9004_input8sequential_1812/dense_9004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1sequential_1812/dense_9004/BiasAdd/ReadVariableOpReadVariableOp:sequential_1812_dense_9004_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
"sequential_1812/dense_9004/BiasAddBiasAdd+sequential_1812/dense_9004/MatMul:product:09sequential_1812/dense_9004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_1812/dense_9004/ReluRelu+sequential_1812/dense_9004/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
0sequential_1812/dense_9005/MatMul/ReadVariableOpReadVariableOp9sequential_1812_dense_9005_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
!sequential_1812/dense_9005/MatMulMatMul-sequential_1812/dense_9004/Relu:activations:08sequential_1812/dense_9005/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1sequential_1812/dense_9005/BiasAdd/ReadVariableOpReadVariableOp:sequential_1812_dense_9005_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
"sequential_1812/dense_9005/BiasAddBiasAdd+sequential_1812/dense_9005/MatMul:product:09sequential_1812/dense_9005/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_1812/dense_9005/ReluRelu+sequential_1812/dense_9005/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
%sequential_1812/dropout_3591/IdentityIdentity-sequential_1812/dense_9005/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
0sequential_1812/dense_9006/MatMul/ReadVariableOpReadVariableOp9sequential_1812_dense_9006_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
!sequential_1812/dense_9006/MatMulMatMul.sequential_1812/dropout_3591/Identity:output:08sequential_1812/dense_9006/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1sequential_1812/dense_9006/BiasAdd/ReadVariableOpReadVariableOp:sequential_1812_dense_9006_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
"sequential_1812/dense_9006/BiasAddBiasAdd+sequential_1812/dense_9006/MatMul:product:09sequential_1812/dense_9006/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_1812/dense_9006/ReluRelu+sequential_1812/dense_9006/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity-sequential_1812/dense_9006/Relu:activations:02^sequential_1812/dense_9004/BiasAdd/ReadVariableOp1^sequential_1812/dense_9004/MatMul/ReadVariableOp2^sequential_1812/dense_9005/BiasAdd/ReadVariableOp1^sequential_1812/dense_9005/MatMul/ReadVariableOp2^sequential_1812/dense_9006/BiasAdd/ReadVariableOp1^sequential_1812/dense_9006/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2f
1sequential_1812/dense_9006/BiasAdd/ReadVariableOp1sequential_1812/dense_9006/BiasAdd/ReadVariableOp2d
0sequential_1812/dense_9004/MatMul/ReadVariableOp0sequential_1812/dense_9004/MatMul/ReadVariableOp2f
1sequential_1812/dense_9005/BiasAdd/ReadVariableOp1sequential_1812/dense_9005/BiasAdd/ReadVariableOp2d
0sequential_1812/dense_9006/MatMul/ReadVariableOp0sequential_1812/dense_9006/MatMul/ReadVariableOp2f
1sequential_1812/dense_9004/BiasAdd/ReadVariableOp1sequential_1812/dense_9004/BiasAdd/ReadVariableOp2d
0sequential_1812/dense_9005/MatMul/ReadVariableOp0sequential_1812/dense_9005/MatMul/ReadVariableOp: : : : :0 ,
*
_user_specified_namedense_9004_input: : 
?	
?
2__inference_sequential_1812_layer_call_fn_15308791

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-15308657*V
fQRO
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308656*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308780

inputs-
)dense_9004_matmul_readvariableop_resource.
*dense_9004_biasadd_readvariableop_resource-
)dense_9005_matmul_readvariableop_resource.
*dense_9005_biasadd_readvariableop_resource-
)dense_9006_matmul_readvariableop_resource.
*dense_9006_biasadd_readvariableop_resource
identity??!dense_9004/BiasAdd/ReadVariableOp? dense_9004/MatMul/ReadVariableOp?!dense_9005/BiasAdd/ReadVariableOp? dense_9005/MatMul/ReadVariableOp?!dense_9006/BiasAdd/ReadVariableOp? dense_9006/MatMul/ReadVariableOp?
 dense_9004/MatMul/ReadVariableOpReadVariableOp)dense_9004_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@
dense_9004/MatMulMatMulinputs(dense_9004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_9004/BiasAdd/ReadVariableOpReadVariableOp*dense_9004_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_9004/BiasAddBiasAdddense_9004/MatMul:product:0)dense_9004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_9004/ReluReludense_9004/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
 dense_9005/MatMul/ReadVariableOpReadVariableOp)dense_9005_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
dense_9005/MatMulMatMuldense_9004/Relu:activations:0(dense_9005/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_9005/BiasAdd/ReadVariableOpReadVariableOp*dense_9005_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_9005/BiasAddBiasAdddense_9005/MatMul:product:0)dense_9005/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_9005/ReluReludense_9005/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@r
dropout_3591/IdentityIdentitydense_9005/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
 dense_9006/MatMul/ReadVariableOpReadVariableOp)dense_9006_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
dense_9006/MatMulMatMuldropout_3591/Identity:output:0(dense_9006/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_9006/BiasAdd/ReadVariableOpReadVariableOp*dense_9006_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_9006/BiasAddBiasAdddense_9006/MatMul:product:0)dense_9006/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_9006/ReluReludense_9006/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_9006/Relu:activations:0"^dense_9004/BiasAdd/ReadVariableOp!^dense_9004/MatMul/ReadVariableOp"^dense_9005/BiasAdd/ReadVariableOp!^dense_9005/MatMul/ReadVariableOp"^dense_9006/BiasAdd/ReadVariableOp!^dense_9006/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_9005/MatMul/ReadVariableOp dense_9005/MatMul/ReadVariableOp2F
!dense_9006/BiasAdd/ReadVariableOp!dense_9006/BiasAdd/ReadVariableOp2F
!dense_9005/BiasAdd/ReadVariableOp!dense_9005/BiasAdd/ReadVariableOp2F
!dense_9004/BiasAdd/ReadVariableOp!dense_9004/BiasAdd/ReadVariableOp2D
 dense_9004/MatMul/ReadVariableOp dense_9004/MatMul/ReadVariableOp2D
 dense_9006/MatMul/ReadVariableOp dense_9006/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308505

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308684

inputs-
)dense_9004_statefulpartitionedcall_args_1-
)dense_9004_statefulpartitionedcall_args_2-
)dense_9005_statefulpartitionedcall_args_1-
)dense_9005_statefulpartitionedcall_args_2-
)dense_9006_statefulpartitionedcall_args_1-
)dense_9006_statefulpartitionedcall_args_2
identity??"dense_9004/StatefulPartitionedCall?"dense_9005/StatefulPartitionedCall?"dense_9006/StatefulPartitionedCall?
"dense_9004/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_9004_statefulpartitionedcall_args_1)dense_9004_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308511*Q
fLRJ
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308505*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9005/StatefulPartitionedCallStatefulPartitionedCall+dense_9004/StatefulPartitionedCall:output:0)dense_9005_statefulpartitionedcall_args_1)dense_9005_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308539*Q
fLRJ
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308533*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
dropout_3591/PartitionedCallPartitionedCall+dense_9005/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-15308589*S
fNRL
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308577*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
"dense_9006/StatefulPartitionedCallStatefulPartitionedCall%dropout_3591/PartitionedCall:output:0)dense_9006_statefulpartitionedcall_args_1)dense_9006_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308611*Q
fLRJ
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308605*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:??????????
IdentityIdentity+dense_9006/StatefulPartitionedCall:output:0#^dense_9004/StatefulPartitionedCall#^dense_9005/StatefulPartitionedCall#^dense_9006/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2H
"dense_9004/StatefulPartitionedCall"dense_9004/StatefulPartitionedCall2H
"dense_9005/StatefulPartitionedCall"dense_9005/StatefulPartitionedCall2H
"dense_9006/StatefulPartitionedCall"dense_9006/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
h
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308863

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:?????????@[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*&
_input_shapes
:?????????@:& "
 
_user_specified_nameinputs
?
?
-__inference_dense_9004_layer_call_fn_15308820

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-15308511*Q
fLRJ
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308505*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:?????????@?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?	
?
&__inference_signature_wrapper_15308711
dense_9004_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_9004_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-15308702*,
f'R%
#__inference__wrapped_model_15308488*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
	2*'
_output_shapes
:??????????
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9004_input: : 
?f
?
$__inference__traced_restore_15309085
file_prefix&
"assignvariableop_dense_9004_kernel&
"assignvariableop_1_dense_9004_bias(
$assignvariableop_2_dense_9005_kernel&
"assignvariableop_3_dense_9005_bias(
$assignvariableop_4_dense_9006_kernel&
"assignvariableop_5_dense_9006_bias!
assignvariableop_6_nadam_iter#
assignvariableop_7_nadam_beta_1#
assignvariableop_8_nadam_beta_2"
assignvariableop_9_nadam_decay+
'assignvariableop_10_nadam_learning_rate,
(assignvariableop_11_nadam_momentum_cache
assignvariableop_12_total
assignvariableop_13_count1
-assignvariableop_14_nadam_dense_9004_kernel_m/
+assignvariableop_15_nadam_dense_9004_bias_m1
-assignvariableop_16_nadam_dense_9005_kernel_m/
+assignvariableop_17_nadam_dense_9005_bias_m1
-assignvariableop_18_nadam_dense_9006_kernel_m/
+assignvariableop_19_nadam_dense_9006_bias_m1
-assignvariableop_20_nadam_dense_9004_kernel_v/
+assignvariableop_21_nadam_dense_9004_bias_v1
-assignvariableop_22_nadam_dense_9005_kernel_v/
+assignvariableop_23_nadam_dense_9005_bias_v1
-assignvariableop_24_nadam_dense_9006_kernel_v/
+assignvariableop_25_nadam_dense_9006_bias_v
identity_27??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
RestoreV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
dtypes
2	*|
_output_shapesj
h::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:~
AssignVariableOpAssignVariableOp"assignvariableop_dense_9004_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_9004_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_9005_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_9005_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_9006_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_9006_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:}
AssignVariableOp_6AssignVariableOpassignvariableop_6_nadam_iterIdentity_6:output:0*
dtype0	*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_nadam_beta_1Identity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_nadam_beta_2Identity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:~
AssignVariableOp_9AssignVariableOpassignvariableop_9_nadam_decayIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_nadam_learning_rateIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp(assignvariableop_11_nadam_momentum_cacheIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:{
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:{
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp-assignvariableop_14_nadam_dense_9004_kernel_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_nadam_dense_9004_bias_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp-assignvariableop_16_nadam_dense_9005_kernel_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_nadam_dense_9005_bias_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp-assignvariableop_18_nadam_dense_9006_kernel_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_nadam_dense_9006_bias_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp-assignvariableop_20_nadam_dense_9004_kernel_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_nadam_dense_9004_bias_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_nadam_dense_9005_kernel_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_nadam_dense_9005_bias_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_nadam_dense_9006_kernel_vIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_nadam_dense_9006_bias_vIdentity_25:output:0*
dtype0*
_output_shapes
 ?
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_26Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_27IdentityIdentity_26:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_27Identity_27:output:0*}
_input_shapesl
j: ::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252$
AssignVariableOpAssignVariableOp: : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : :
 
?8
?
!__inference__traced_save_15308994
file_prefix0
,savev2_dense_9004_kernel_read_readvariableop.
*savev2_dense_9004_bias_read_readvariableop0
,savev2_dense_9005_kernel_read_readvariableop.
*savev2_dense_9005_bias_read_readvariableop0
,savev2_dense_9006_kernel_read_readvariableop.
*savev2_dense_9006_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_nadam_dense_9004_kernel_m_read_readvariableop6
2savev2_nadam_dense_9004_bias_m_read_readvariableop8
4savev2_nadam_dense_9005_kernel_m_read_readvariableop6
2savev2_nadam_dense_9005_bias_m_read_readvariableop8
4savev2_nadam_dense_9006_kernel_m_read_readvariableop6
2savev2_nadam_dense_9006_bias_m_read_readvariableop8
4savev2_nadam_dense_9004_kernel_v_read_readvariableop6
2savev2_nadam_dense_9004_bias_v_read_readvariableop8
4savev2_nadam_dense_9005_kernel_v_read_readvariableop6
2savev2_nadam_dense_9005_bias_v_read_readvariableop8
4savev2_nadam_dense_9006_kernel_v_read_readvariableop6
2savev2_nadam_dense_9006_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_ab9e9dbf77d2496f884b8e27128317c8/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
SaveV2/shape_and_slicesConst"/device:CPU:0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_9004_kernel_read_readvariableop*savev2_dense_9004_bias_read_readvariableop,savev2_dense_9005_kernel_read_readvariableop*savev2_dense_9005_bias_read_readvariableop,savev2_dense_9006_kernel_read_readvariableop*savev2_dense_9006_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_nadam_dense_9004_kernel_m_read_readvariableop2savev2_nadam_dense_9004_bias_m_read_readvariableop4savev2_nadam_dense_9005_kernel_m_read_readvariableop2savev2_nadam_dense_9005_bias_m_read_readvariableop4savev2_nadam_dense_9006_kernel_m_read_readvariableop2savev2_nadam_dense_9006_bias_m_read_readvariableop4savev2_nadam_dense_9004_kernel_v_read_readvariableop2savev2_nadam_dense_9004_bias_v_read_readvariableop4savev2_nadam_dense_9005_kernel_v_read_readvariableop2savev2_nadam_dense_9005_bias_v_read_readvariableop4savev2_nadam_dense_9006_kernel_v_read_readvariableop2savev2_nadam_dense_9006_bias_v_read_readvariableop"/device:CPU:0*(
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: ?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@:: : : : : : : : :@:@:@@:@:@::@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : : :
 "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
M
dense_9004_input9
"serving_default_dense_9004_input:0?????????>

dense_90060
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:??
?!
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
\__call__
]_default_save_signature
*^&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1812", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1812", "layers": [{"class_name": "Dense", "config": {"name": "dense_9004", "trainable": true, "batch_input_shape": [null, 20], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9005", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3591", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_9006", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1812", "layers": [{"class_name": "Dense", "config": {"name": "dense_9004", "trainable": true, "batch_input_shape": [null, 20], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9005", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3591", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_9006", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "dense_9004_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 20], "config": {"batch_input_shape": [null, 20], "dtype": "float32", "sparse": false, "name": "dense_9004_input"}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9004", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 20], "config": {"name": "dense_9004", "trainable": true, "batch_input_shape": [null, 20], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9005", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9005", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_3591", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_3591", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_9006", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9006", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
&iter

'beta_1

(beta_2
	)decay
*learning_rate
+momentum_cachemPmQmRmS mT!mUvVvWvXvY vZ!v["
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
?
regularization_losses

,layers
	variables
-metrics
.layer_regularization_losses
	trainable_variables
/non_trainable_variables
\__call__
]_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,
iserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

0layers
	variables
1metrics
2layer_regularization_losses
trainable_variables
3non_trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
#:!@2dense_9004/kernel
:@2dense_9004/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

4layers
	variables
5metrics
6layer_regularization_losses
trainable_variables
7non_trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
#:!@@2dense_9005/kernel
:@2dense_9005/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses

8layers
	variables
9metrics
:layer_regularization_losses
trainable_variables
;non_trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

<layers
	variables
=metrics
>layer_regularization_losses
trainable_variables
?non_trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
#:!@2dense_9006/kernel
:2dense_9006/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
"regularization_losses

@layers
#	variables
Ametrics
Blayer_regularization_losses
$trainable_variables
Cnon_trainable_variables
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: 2Nadam/momentum_cache
<
0
1
2
3"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Etotal
	Fcount
G
_fn_kwargs
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
j__call__
*k&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Hregularization_losses

Llayers
I	variables
Mmetrics
Nlayer_regularization_losses
Jtrainable_variables
Onon_trainable_variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
):'@2Nadam/dense_9004/kernel/m
#:!@2Nadam/dense_9004/bias/m
):'@@2Nadam/dense_9005/kernel/m
#:!@2Nadam/dense_9005/bias/m
):'@2Nadam/dense_9006/kernel/m
#:!2Nadam/dense_9006/bias/m
):'@2Nadam/dense_9004/kernel/v
#:!@2Nadam/dense_9004/bias/v
):'@@2Nadam/dense_9005/kernel/v
#:!@2Nadam/dense_9005/bias/v
):'@2Nadam/dense_9006/kernel/v
#:!2Nadam/dense_9006/bias/v
?2?
2__inference_sequential_1812_layer_call_fn_15308802
2__inference_sequential_1812_layer_call_fn_15308791
2__inference_sequential_1812_layer_call_fn_15308666
2__inference_sequential_1812_layer_call_fn_15308694?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_15308488?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
dense_9004_input?????????
?2?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308639
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308623
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308780
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308754?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
-__inference_dense_9004_layer_call_fn_15308820?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dense_9005_layer_call_fn_15308838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308831?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_dropout_3591_layer_call_fn_15308868
/__inference_dropout_3591_layer_call_fn_15308873?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308858
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308863?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dense_9006_layer_call_fn_15308891?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308884?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
>B<
&__inference_signature_wrapper_15308711dense_9004_input
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308623r !A?>
7?4
*?'
dense_9004_input?????????
p

 
? "%?"
?
0?????????
? ?
2__inference_sequential_1812_layer_call_fn_15308802[ !7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
#__inference__wrapped_model_15308488| !9?6
/?,
*?'
dense_9004_input?????????
? "7?4
2

dense_9006$?!

dense_9006??????????
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308754h !7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_dense_9005_layer_call_fn_15308838O/?,
%?"
 ?
inputs?????????@
? "??????????@?
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308858\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
J__inference_dropout_3591_layer_call_and_return_conditional_losses_15308863\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308639r !A?>
7?4
*?'
dense_9004_input?????????
p 

 
? "%?"
?
0?????????
? ?
&__inference_signature_wrapper_15308711? !M?J
? 
C?@
>
dense_9004_input*?'
dense_9004_input?????????"7?4
2

dense_9006$?!

dense_9006??????????
-__inference_dense_9006_layer_call_fn_15308891O !/?,
%?"
 ?
inputs?????????@
? "???????????
-__inference_dense_9004_layer_call_fn_15308820O/?,
%?"
 ?
inputs?????????
? "??????????@?
2__inference_sequential_1812_layer_call_fn_15308666e !A?>
7?4
*?'
dense_9004_input?????????
p

 
? "???????????
M__inference_sequential_1812_layer_call_and_return_conditional_losses_15308780h !7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_dense_9006_layer_call_and_return_conditional_losses_15308884\ !/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
2__inference_sequential_1812_layer_call_fn_15308791[ !7?4
-?*
 ?
inputs?????????
p

 
? "???????????
/__inference_dropout_3591_layer_call_fn_15308873O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
/__inference_dropout_3591_layer_call_fn_15308868O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
H__inference_dense_9004_layer_call_and_return_conditional_losses_15308813\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? ?
H__inference_dense_9005_layer_call_and_return_conditional_losses_15308831\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
2__inference_sequential_1812_layer_call_fn_15308694e !A?>
7?4
*?'
dense_9004_input?????????
p 

 
? "??????????