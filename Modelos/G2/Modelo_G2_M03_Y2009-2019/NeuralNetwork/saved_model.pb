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
dense_6570/kernelVarHandleOp*
shape
:@*"
shared_namedense_6570/kernel*
dtype0*
_output_shapes
: 
w
%dense_6570/kernel/Read/ReadVariableOpReadVariableOpdense_6570/kernel*
dtype0*
_output_shapes

:@
v
dense_6570/biasVarHandleOp*
shape:@* 
shared_namedense_6570/bias*
dtype0*
_output_shapes
: 
o
#dense_6570/bias/Read/ReadVariableOpReadVariableOpdense_6570/bias*
dtype0*
_output_shapes
:@
~
dense_6571/kernelVarHandleOp*
shape
:@@*"
shared_namedense_6571/kernel*
dtype0*
_output_shapes
: 
w
%dense_6571/kernel/Read/ReadVariableOpReadVariableOpdense_6571/kernel*
dtype0*
_output_shapes

:@@
v
dense_6571/biasVarHandleOp*
shape:@* 
shared_namedense_6571/bias*
dtype0*
_output_shapes
: 
o
#dense_6571/bias/Read/ReadVariableOpReadVariableOpdense_6571/bias*
dtype0*
_output_shapes
:@
~
dense_6572/kernelVarHandleOp*
shape
:@*"
shared_namedense_6572/kernel*
dtype0*
_output_shapes
: 
w
%dense_6572/kernel/Read/ReadVariableOpReadVariableOpdense_6572/kernel*
dtype0*
_output_shapes

:@
v
dense_6572/biasVarHandleOp*
shape:* 
shared_namedense_6572/bias*
dtype0*
_output_shapes
: 
o
#dense_6572/bias/Read/ReadVariableOpReadVariableOpdense_6572/bias*
dtype0*
_output_shapes
:
n
Adadelta/iterVarHandleOp*
shape: *
shared_nameAdadelta/iter*
dtype0	*
_output_shapes
: 
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
dtype0	*
_output_shapes
: 
p
Adadelta/decayVarHandleOp*
shape: *
shared_nameAdadelta/decay*
dtype0*
_output_shapes
: 
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
dtype0*
_output_shapes
: 
?
Adadelta/learning_rateVarHandleOp*
shape: *'
shared_nameAdadelta/learning_rate*
dtype0*
_output_shapes
: 
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
dtype0*
_output_shapes
: 
l
Adadelta/rhoVarHandleOp*
shape: *
shared_nameAdadelta/rho*
dtype0*
_output_shapes
: 
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
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
%Adadelta/dense_6570/kernel/accum_gradVarHandleOp*
shape
:@*6
shared_name'%Adadelta/dense_6570/kernel/accum_grad*
dtype0*
_output_shapes
: 
?
9Adadelta/dense_6570/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_6570/kernel/accum_grad*
dtype0*
_output_shapes

:@
?
#Adadelta/dense_6570/bias/accum_gradVarHandleOp*
shape:@*4
shared_name%#Adadelta/dense_6570/bias/accum_grad*
dtype0*
_output_shapes
: 
?
7Adadelta/dense_6570/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_6570/bias/accum_grad*
dtype0*
_output_shapes
:@
?
%Adadelta/dense_6571/kernel/accum_gradVarHandleOp*
shape
:@@*6
shared_name'%Adadelta/dense_6571/kernel/accum_grad*
dtype0*
_output_shapes
: 
?
9Adadelta/dense_6571/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_6571/kernel/accum_grad*
dtype0*
_output_shapes

:@@
?
#Adadelta/dense_6571/bias/accum_gradVarHandleOp*
shape:@*4
shared_name%#Adadelta/dense_6571/bias/accum_grad*
dtype0*
_output_shapes
: 
?
7Adadelta/dense_6571/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_6571/bias/accum_grad*
dtype0*
_output_shapes
:@
?
%Adadelta/dense_6572/kernel/accum_gradVarHandleOp*
shape
:@*6
shared_name'%Adadelta/dense_6572/kernel/accum_grad*
dtype0*
_output_shapes
: 
?
9Adadelta/dense_6572/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_6572/kernel/accum_grad*
dtype0*
_output_shapes

:@
?
#Adadelta/dense_6572/bias/accum_gradVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_6572/bias/accum_grad*
dtype0*
_output_shapes
: 
?
7Adadelta/dense_6572/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_6572/bias/accum_grad*
dtype0*
_output_shapes
:
?
$Adadelta/dense_6570/kernel/accum_varVarHandleOp*
shape
:@*5
shared_name&$Adadelta/dense_6570/kernel/accum_var*
dtype0*
_output_shapes
: 
?
8Adadelta/dense_6570/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_6570/kernel/accum_var*
dtype0*
_output_shapes

:@
?
"Adadelta/dense_6570/bias/accum_varVarHandleOp*
shape:@*3
shared_name$"Adadelta/dense_6570/bias/accum_var*
dtype0*
_output_shapes
: 
?
6Adadelta/dense_6570/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_6570/bias/accum_var*
dtype0*
_output_shapes
:@
?
$Adadelta/dense_6571/kernel/accum_varVarHandleOp*
shape
:@@*5
shared_name&$Adadelta/dense_6571/kernel/accum_var*
dtype0*
_output_shapes
: 
?
8Adadelta/dense_6571/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_6571/kernel/accum_var*
dtype0*
_output_shapes

:@@
?
"Adadelta/dense_6571/bias/accum_varVarHandleOp*
shape:@*3
shared_name$"Adadelta/dense_6571/bias/accum_var*
dtype0*
_output_shapes
: 
?
6Adadelta/dense_6571/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_6571/bias/accum_var*
dtype0*
_output_shapes
:@
?
$Adadelta/dense_6572/kernel/accum_varVarHandleOp*
shape
:@*5
shared_name&$Adadelta/dense_6572/kernel/accum_var*
dtype0*
_output_shapes
: 
?
8Adadelta/dense_6572/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_6572/kernel/accum_var*
dtype0*
_output_shapes

:@
?
"Adadelta/dense_6572/bias/accum_varVarHandleOp*
shape:*3
shared_name$"Adadelta/dense_6572/bias/accum_var*
dtype0*
_output_shapes
: 
?
6Adadelta/dense_6572/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_6572/bias/accum_var*
dtype0*
_output_shapes
:

NoOpNoOp
?+
ConstConst"/device:CPU:0*?*
value?*B?* B?*
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
?
&iter
	'decay
(learning_rate
)rho
accum_gradN
accum_gradO
accum_gradP
accum_gradQ 
accum_gradR!
accum_gradS	accum_varT	accum_varU	accum_varV	accum_varW 	accum_varX!	accum_varY
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

*layers
	variables
+metrics
,layer_regularization_losses
	trainable_variables
-non_trainable_variables
 
 
 
 
?
regularization_losses

.layers
	variables
/metrics
0layer_regularization_losses
trainable_variables
1non_trainable_variables
][
VARIABLE_VALUEdense_6570/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_6570/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

2layers
	variables
3metrics
4layer_regularization_losses
trainable_variables
5non_trainable_variables
][
VARIABLE_VALUEdense_6571/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_6571/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses

6layers
	variables
7metrics
8layer_regularization_losses
trainable_variables
9non_trainable_variables
 
 
 
?
regularization_losses

:layers
	variables
;metrics
<layer_regularization_losses
trainable_variables
=non_trainable_variables
][
VARIABLE_VALUEdense_6572/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_6572/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
?
"regularization_losses

>layers
#	variables
?metrics
@layer_regularization_losses
$trainable_variables
Anon_trainable_variables
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

B0
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
	Ctotal
	Dcount
E
_fn_kwargs
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

C0
D1
 
?
Fregularization_losses

Jlayers
G	variables
Kmetrics
Llayer_regularization_losses
Htrainable_variables
Mnon_trainable_variables
 
 
 

C0
D1
??
VARIABLE_VALUE%Adadelta/dense_6570/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/dense_6570/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adadelta/dense_6571/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/dense_6571/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE%Adadelta/dense_6572/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/dense_6572/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adadelta/dense_6570/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_6570/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adadelta/dense_6571/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_6571/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$Adadelta/dense_6572/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_6572/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
?
 serving_default_dense_6570_inputPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_6570_inputdense_6570/kerneldense_6570/biasdense_6571/kerneldense_6571/biasdense_6572/kerneldense_6572/bias*/
_gradient_op_typePartitionedCall-11170958*/
f*R(
&__inference_signature_wrapper_11170726*
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
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_6570/kernel/Read/ReadVariableOp#dense_6570/bias/Read/ReadVariableOp%dense_6571/kernel/Read/ReadVariableOp#dense_6571/bias/Read/ReadVariableOp%dense_6572/kernel/Read/ReadVariableOp#dense_6572/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp9Adadelta/dense_6570/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_6570/bias/accum_grad/Read/ReadVariableOp9Adadelta/dense_6571/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_6571/bias/accum_grad/Read/ReadVariableOp9Adadelta/dense_6572/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_6572/bias/accum_grad/Read/ReadVariableOp8Adadelta/dense_6570/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_6570/bias/accum_var/Read/ReadVariableOp8Adadelta/dense_6571/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_6571/bias/accum_var/Read/ReadVariableOp8Adadelta/dense_6572/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_6572/bias/accum_var/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-11171004**
f%R#
!__inference__traced_save_11171003*
Tout
2**
config_proto

CPU

GPU 2J 8*%
Tin
2	*
_output_shapes
: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6570/kerneldense_6570/biasdense_6571/kerneldense_6571/biasdense_6572/kerneldense_6572/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcount%Adadelta/dense_6570/kernel/accum_grad#Adadelta/dense_6570/bias/accum_grad%Adadelta/dense_6571/kernel/accum_grad#Adadelta/dense_6571/bias/accum_grad%Adadelta/dense_6572/kernel/accum_grad#Adadelta/dense_6572/bias/accum_grad$Adadelta/dense_6570/kernel/accum_var"Adadelta/dense_6570/bias/accum_var$Adadelta/dense_6571/kernel/accum_var"Adadelta/dense_6571/bias/accum_var$Adadelta/dense_6572/kernel/accum_var"Adadelta/dense_6572/bias/accum_var*/
_gradient_op_typePartitionedCall-11171089*-
f(R&
$__inference__traced_restore_11171088*
Tout
2**
config_proto

CPU

GPU 2J 8*$
Tin
2*
_output_shapes
: ˚
?
h
/__inference_dropout_2620_layer_call_fn_11170883

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*/
_gradient_op_typePartitionedCall-11170597*S
fNRL
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170586*
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
?	
?
2__inference_sequential_1322_layer_call_fn_11170817

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-11170701*V
fQRO
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170700*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
i
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170586

inputs
identity?Q
dropout/rateConst*
valueB
 *???=*
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
?9
?
!__inference__traced_save_11171003
file_prefix0
,savev2_dense_6570_kernel_read_readvariableop.
*savev2_dense_6570_bias_read_readvariableop0
,savev2_dense_6571_kernel_read_readvariableop.
*savev2_dense_6571_bias_read_readvariableop0
,savev2_dense_6572_kernel_read_readvariableop.
*savev2_dense_6572_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopD
@savev2_adadelta_dense_6570_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_6570_bias_accum_grad_read_readvariableopD
@savev2_adadelta_dense_6571_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_6571_bias_accum_grad_read_readvariableopD
@savev2_adadelta_dense_6572_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_6572_bias_accum_grad_read_readvariableopC
?savev2_adadelta_dense_6570_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_6570_bias_accum_var_read_readvariableopC
?savev2_adadelta_dense_6571_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_6571_bias_accum_var_read_readvariableopC
?savev2_adadelta_dense_6572_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_6572_bias_accum_var_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_5a0a9c3a3e604982818206f84e7a0f03/part*
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
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
SaveV2/shape_and_slicesConst"/device:CPU:0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_6570_kernel_read_readvariableop*savev2_dense_6570_bias_read_readvariableop,savev2_dense_6571_kernel_read_readvariableop*savev2_dense_6571_bias_read_readvariableop,savev2_dense_6572_kernel_read_readvariableop*savev2_dense_6572_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop@savev2_adadelta_dense_6570_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_6570_bias_accum_grad_read_readvariableop@savev2_adadelta_dense_6571_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_6571_bias_accum_grad_read_readvariableop@savev2_adadelta_dense_6572_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_6572_bias_accum_grad_read_readvariableop?savev2_adadelta_dense_6570_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_6570_bias_accum_var_read_readvariableop?savev2_adadelta_dense_6571_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_6571_bias_accum_var_read_readvariableop?savev2_adadelta_dense_6572_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_6572_bias_accum_var_read_readvariableop"/device:CPU:0*&
dtypes
2	*
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
?: :@:@:@@:@:@:: : : : : : :@:@:@@:@:@::@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : :
 : : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : 
?b
?
$__inference__traced_restore_11171088
file_prefix&
"assignvariableop_dense_6570_kernel&
"assignvariableop_1_dense_6570_bias(
$assignvariableop_2_dense_6571_kernel&
"assignvariableop_3_dense_6571_bias(
$assignvariableop_4_dense_6572_kernel&
"assignvariableop_5_dense_6572_bias$
 assignvariableop_6_adadelta_iter%
!assignvariableop_7_adadelta_decay-
)assignvariableop_8_adadelta_learning_rate#
assignvariableop_9_adadelta_rho
assignvariableop_10_total
assignvariableop_11_count=
9assignvariableop_12_adadelta_dense_6570_kernel_accum_grad;
7assignvariableop_13_adadelta_dense_6570_bias_accum_grad=
9assignvariableop_14_adadelta_dense_6571_kernel_accum_grad;
7assignvariableop_15_adadelta_dense_6571_bias_accum_grad=
9assignvariableop_16_adadelta_dense_6572_kernel_accum_grad;
7assignvariableop_17_adadelta_dense_6572_bias_accum_grad<
8assignvariableop_18_adadelta_dense_6570_kernel_accum_var:
6assignvariableop_19_adadelta_dense_6570_bias_accum_var<
8assignvariableop_20_adadelta_dense_6571_kernel_accum_var:
6assignvariableop_21_adadelta_dense_6571_bias_accum_var<
8assignvariableop_22_adadelta_dense_6572_kernel_accum_var:
6assignvariableop_23_adadelta_dense_6572_bias_accum_var
identity_25??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:?
RestoreV2/shape_and_slicesConst"/device:CPU:0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*&
dtypes
2	*t
_output_shapesb
`::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:~
AssignVariableOpAssignVariableOp"assignvariableop_dense_6570_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_6570_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_6571_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_6571_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_6572_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_6572_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp assignvariableop_6_adadelta_iterIdentity_6:output:0*
dtype0	*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_adadelta_decayIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp)assignvariableop_8_adadelta_learning_rateIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adadelta_rhoIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:{
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:{
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_adadelta_dense_6570_kernel_accum_gradIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp7assignvariableop_13_adadelta_dense_6570_bias_accum_gradIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp9assignvariableop_14_adadelta_dense_6571_kernel_accum_gradIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adadelta_dense_6571_bias_accum_gradIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp9assignvariableop_16_adadelta_dense_6572_kernel_accum_gradIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adadelta_dense_6572_bias_accum_gradIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adadelta_dense_6570_kernel_accum_varIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adadelta_dense_6570_bias_accum_varIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adadelta_dense_6571_kernel_accum_varIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adadelta_dense_6571_bias_accum_varIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adadelta_dense_6572_kernel_accum_varIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adadelta_dense_6572_bias_accum_varIdentity_23:output:0*
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
 ?
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ?
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2: : : : : :
 : : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : 
?	
?
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170549

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
?	
?
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170899

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
:?????????@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
K
/__inference_dropout_2620_layer_call_fn_11170888

inputs
identity?
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-11170605*S
fNRL
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170593*
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
?	
?
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170621

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
?
?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170700

inputs-
)dense_6570_statefulpartitionedcall_args_1-
)dense_6570_statefulpartitionedcall_args_2-
)dense_6571_statefulpartitionedcall_args_1-
)dense_6571_statefulpartitionedcall_args_2-
)dense_6572_statefulpartitionedcall_args_1-
)dense_6572_statefulpartitionedcall_args_2
identity??"dense_6570/StatefulPartitionedCall?"dense_6571/StatefulPartitionedCall?"dense_6572/StatefulPartitionedCall?
"dense_6570/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_6570_statefulpartitionedcall_args_1)dense_6570_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170527*Q
fLRJ
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170521*
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
"dense_6571/StatefulPartitionedCallStatefulPartitionedCall+dense_6570/StatefulPartitionedCall:output:0)dense_6571_statefulpartitionedcall_args_1)dense_6571_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170555*Q
fLRJ
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170549*
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
dropout_2620/PartitionedCallPartitionedCall+dense_6571/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-11170605*S
fNRL
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170593*
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
"dense_6572/StatefulPartitionedCallStatefulPartitionedCall%dropout_2620/PartitionedCall:output:0)dense_6572_statefulpartitionedcall_args_1)dense_6572_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170627*Q
fLRJ
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170621*
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
IdentityIdentity+dense_6572/StatefulPartitionedCall:output:0#^dense_6570/StatefulPartitionedCall#^dense_6571/StatefulPartitionedCall#^dense_6572/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2H
"dense_6570/StatefulPartitionedCall"dense_6570/StatefulPartitionedCall2H
"dense_6571/StatefulPartitionedCall"dense_6571/StatefulPartitionedCall2H
"dense_6572/StatefulPartitionedCall"dense_6572/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170795

inputs-
)dense_6570_matmul_readvariableop_resource.
*dense_6570_biasadd_readvariableop_resource-
)dense_6571_matmul_readvariableop_resource.
*dense_6571_biasadd_readvariableop_resource-
)dense_6572_matmul_readvariableop_resource.
*dense_6572_biasadd_readvariableop_resource
identity??!dense_6570/BiasAdd/ReadVariableOp? dense_6570/MatMul/ReadVariableOp?!dense_6571/BiasAdd/ReadVariableOp? dense_6571/MatMul/ReadVariableOp?!dense_6572/BiasAdd/ReadVariableOp? dense_6572/MatMul/ReadVariableOp?
 dense_6570/MatMul/ReadVariableOpReadVariableOp)dense_6570_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@
dense_6570/MatMulMatMulinputs(dense_6570/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_6570/BiasAdd/ReadVariableOpReadVariableOp*dense_6570_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_6570/BiasAddBiasAdddense_6570/MatMul:product:0)dense_6570/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_6570/ReluReludense_6570/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
 dense_6571/MatMul/ReadVariableOpReadVariableOp)dense_6571_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
dense_6571/MatMulMatMuldense_6570/Relu:activations:0(dense_6571/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_6571/BiasAdd/ReadVariableOpReadVariableOp*dense_6571_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_6571/BiasAddBiasAdddense_6571/MatMul:product:0)dense_6571/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_6571/ReluReludense_6571/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@r
dropout_2620/IdentityIdentitydense_6571/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
 dense_6572/MatMul/ReadVariableOpReadVariableOp)dense_6572_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
dense_6572/MatMulMatMuldropout_2620/Identity:output:0(dense_6572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_6572/BiasAdd/ReadVariableOpReadVariableOp*dense_6572_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_6572/BiasAddBiasAdddense_6572/MatMul:product:0)dense_6572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_6572/ReluReludense_6572/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_6572/Relu:activations:0"^dense_6570/BiasAdd/ReadVariableOp!^dense_6570/MatMul/ReadVariableOp"^dense_6571/BiasAdd/ReadVariableOp!^dense_6571/MatMul/ReadVariableOp"^dense_6572/BiasAdd/ReadVariableOp!^dense_6572/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_6571/MatMul/ReadVariableOp dense_6571/MatMul/ReadVariableOp2F
!dense_6572/BiasAdd/ReadVariableOp!dense_6572/BiasAdd/ReadVariableOp2D
 dense_6570/MatMul/ReadVariableOp dense_6570/MatMul/ReadVariableOp2F
!dense_6571/BiasAdd/ReadVariableOp!dense_6571/BiasAdd/ReadVariableOp2D
 dense_6572/MatMul/ReadVariableOp dense_6572/MatMul/ReadVariableOp2F
!dense_6570/BiasAdd/ReadVariableOp!dense_6570/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
2__inference_sequential_1322_layer_call_fn_11170682
dense_6570_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_6570_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-11170673*V
fQRO
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170672*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_6570_input: : 
?	
?
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170846

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
:?????????@::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170639
dense_6570_input-
)dense_6570_statefulpartitionedcall_args_1-
)dense_6570_statefulpartitionedcall_args_2-
)dense_6571_statefulpartitionedcall_args_1-
)dense_6571_statefulpartitionedcall_args_2-
)dense_6572_statefulpartitionedcall_args_1-
)dense_6572_statefulpartitionedcall_args_2
identity??"dense_6570/StatefulPartitionedCall?"dense_6571/StatefulPartitionedCall?"dense_6572/StatefulPartitionedCall?$dropout_2620/StatefulPartitionedCall?
"dense_6570/StatefulPartitionedCallStatefulPartitionedCalldense_6570_input)dense_6570_statefulpartitionedcall_args_1)dense_6570_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170527*Q
fLRJ
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170521*
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
"dense_6571/StatefulPartitionedCallStatefulPartitionedCall+dense_6570/StatefulPartitionedCall:output:0)dense_6571_statefulpartitionedcall_args_1)dense_6571_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170555*Q
fLRJ
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170549*
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
$dropout_2620/StatefulPartitionedCallStatefulPartitionedCall+dense_6571/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-11170597*S
fNRL
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170586*
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
"dense_6572/StatefulPartitionedCallStatefulPartitionedCall-dropout_2620/StatefulPartitionedCall:output:0)dense_6572_statefulpartitionedcall_args_1)dense_6572_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170627*Q
fLRJ
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170621*
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
IdentityIdentity+dense_6572/StatefulPartitionedCall:output:0#^dense_6570/StatefulPartitionedCall#^dense_6571/StatefulPartitionedCall#^dense_6572/StatefulPartitionedCall%^dropout_2620/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2L
$dropout_2620/StatefulPartitionedCall$dropout_2620/StatefulPartitionedCall2H
"dense_6570/StatefulPartitionedCall"dense_6570/StatefulPartitionedCall2H
"dense_6571/StatefulPartitionedCall"dense_6571/StatefulPartitionedCall2H
"dense_6572/StatefulPartitionedCall"dense_6572/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_6570_input: : 
?
?
-__inference_dense_6572_layer_call_fn_11170906

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170627*Q
fLRJ
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170621*
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
?
h
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170593

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
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170672

inputs-
)dense_6570_statefulpartitionedcall_args_1-
)dense_6570_statefulpartitionedcall_args_2-
)dense_6571_statefulpartitionedcall_args_1-
)dense_6571_statefulpartitionedcall_args_2-
)dense_6572_statefulpartitionedcall_args_1-
)dense_6572_statefulpartitionedcall_args_2
identity??"dense_6570/StatefulPartitionedCall?"dense_6571/StatefulPartitionedCall?"dense_6572/StatefulPartitionedCall?$dropout_2620/StatefulPartitionedCall?
"dense_6570/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_6570_statefulpartitionedcall_args_1)dense_6570_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170527*Q
fLRJ
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170521*
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
"dense_6571/StatefulPartitionedCallStatefulPartitionedCall+dense_6570/StatefulPartitionedCall:output:0)dense_6571_statefulpartitionedcall_args_1)dense_6571_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170555*Q
fLRJ
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170549*
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
$dropout_2620/StatefulPartitionedCallStatefulPartitionedCall+dense_6571/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-11170597*S
fNRL
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170586*
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
"dense_6572/StatefulPartitionedCallStatefulPartitionedCall-dropout_2620/StatefulPartitionedCall:output:0)dense_6572_statefulpartitionedcall_args_1)dense_6572_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170627*Q
fLRJ
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170621*
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
IdentityIdentity+dense_6572/StatefulPartitionedCall:output:0#^dense_6570/StatefulPartitionedCall#^dense_6571/StatefulPartitionedCall#^dense_6572/StatefulPartitionedCall%^dropout_2620/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2L
$dropout_2620/StatefulPartitionedCall$dropout_2620/StatefulPartitionedCall2H
"dense_6570/StatefulPartitionedCall"dense_6570/StatefulPartitionedCall2H
"dense_6571/StatefulPartitionedCall"dense_6571/StatefulPartitionedCall2H
"dense_6572/StatefulPartitionedCall"dense_6572/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
2__inference_sequential_1322_layer_call_fn_11170710
dense_6570_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_6570_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-11170701*V
fQRO
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170700*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_6570_input: : 
?	
?
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170521

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
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
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170828

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
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
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?	
?
2__inference_sequential_1322_layer_call_fn_11170806

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-11170673*V
fQRO
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170672*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
-__inference_dense_6570_layer_call_fn_11170835

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170527*Q
fLRJ
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170521*
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
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?'
?
#__inference__wrapped_model_11170504
dense_6570_input=
9sequential_1322_dense_6570_matmul_readvariableop_resource>
:sequential_1322_dense_6570_biasadd_readvariableop_resource=
9sequential_1322_dense_6571_matmul_readvariableop_resource>
:sequential_1322_dense_6571_biasadd_readvariableop_resource=
9sequential_1322_dense_6572_matmul_readvariableop_resource>
:sequential_1322_dense_6572_biasadd_readvariableop_resource
identity??1sequential_1322/dense_6570/BiasAdd/ReadVariableOp?0sequential_1322/dense_6570/MatMul/ReadVariableOp?1sequential_1322/dense_6571/BiasAdd/ReadVariableOp?0sequential_1322/dense_6571/MatMul/ReadVariableOp?1sequential_1322/dense_6572/BiasAdd/ReadVariableOp?0sequential_1322/dense_6572/MatMul/ReadVariableOp?
0sequential_1322/dense_6570/MatMul/ReadVariableOpReadVariableOp9sequential_1322_dense_6570_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
!sequential_1322/dense_6570/MatMulMatMuldense_6570_input8sequential_1322/dense_6570/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1sequential_1322/dense_6570/BiasAdd/ReadVariableOpReadVariableOp:sequential_1322_dense_6570_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
"sequential_1322/dense_6570/BiasAddBiasAdd+sequential_1322/dense_6570/MatMul:product:09sequential_1322/dense_6570/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_1322/dense_6570/ReluRelu+sequential_1322/dense_6570/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
0sequential_1322/dense_6571/MatMul/ReadVariableOpReadVariableOp9sequential_1322_dense_6571_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
!sequential_1322/dense_6571/MatMulMatMul-sequential_1322/dense_6570/Relu:activations:08sequential_1322/dense_6571/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
1sequential_1322/dense_6571/BiasAdd/ReadVariableOpReadVariableOp:sequential_1322_dense_6571_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
"sequential_1322/dense_6571/BiasAddBiasAdd+sequential_1322/dense_6571/MatMul:product:09sequential_1322/dense_6571/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_1322/dense_6571/ReluRelu+sequential_1322/dense_6571/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
%sequential_1322/dropout_2620/IdentityIdentity-sequential_1322/dense_6571/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
0sequential_1322/dense_6572/MatMul/ReadVariableOpReadVariableOp9sequential_1322_dense_6572_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
!sequential_1322/dense_6572/MatMulMatMul.sequential_1322/dropout_2620/Identity:output:08sequential_1322/dense_6572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1sequential_1322/dense_6572/BiasAdd/ReadVariableOpReadVariableOp:sequential_1322_dense_6572_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
"sequential_1322/dense_6572/BiasAddBiasAdd+sequential_1322/dense_6572/MatMul:product:09sequential_1322/dense_6572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_1322/dense_6572/ReluRelu+sequential_1322/dense_6572/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity-sequential_1322/dense_6572/Relu:activations:02^sequential_1322/dense_6570/BiasAdd/ReadVariableOp1^sequential_1322/dense_6570/MatMul/ReadVariableOp2^sequential_1322/dense_6571/BiasAdd/ReadVariableOp1^sequential_1322/dense_6571/MatMul/ReadVariableOp2^sequential_1322/dense_6572/BiasAdd/ReadVariableOp1^sequential_1322/dense_6572/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2f
1sequential_1322/dense_6570/BiasAdd/ReadVariableOp1sequential_1322/dense_6570/BiasAdd/ReadVariableOp2d
0sequential_1322/dense_6571/MatMul/ReadVariableOp0sequential_1322/dense_6571/MatMul/ReadVariableOp2d
0sequential_1322/dense_6570/MatMul/ReadVariableOp0sequential_1322/dense_6570/MatMul/ReadVariableOp2d
0sequential_1322/dense_6572/MatMul/ReadVariableOp0sequential_1322/dense_6572/MatMul/ReadVariableOp2f
1sequential_1322/dense_6572/BiasAdd/ReadVariableOp1sequential_1322/dense_6572/BiasAdd/ReadVariableOp2f
1sequential_1322/dense_6571/BiasAdd/ReadVariableOp1sequential_1322/dense_6571/BiasAdd/ReadVariableOp: : : : :0 ,
*
_user_specified_namedense_6570_input: : 
?
?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170655
dense_6570_input-
)dense_6570_statefulpartitionedcall_args_1-
)dense_6570_statefulpartitionedcall_args_2-
)dense_6571_statefulpartitionedcall_args_1-
)dense_6571_statefulpartitionedcall_args_2-
)dense_6572_statefulpartitionedcall_args_1-
)dense_6572_statefulpartitionedcall_args_2
identity??"dense_6570/StatefulPartitionedCall?"dense_6571/StatefulPartitionedCall?"dense_6572/StatefulPartitionedCall?
"dense_6570/StatefulPartitionedCallStatefulPartitionedCalldense_6570_input)dense_6570_statefulpartitionedcall_args_1)dense_6570_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170527*Q
fLRJ
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170521*
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
"dense_6571/StatefulPartitionedCallStatefulPartitionedCall+dense_6570/StatefulPartitionedCall:output:0)dense_6571_statefulpartitionedcall_args_1)dense_6571_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170555*Q
fLRJ
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170549*
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
dropout_2620/PartitionedCallPartitionedCall+dense_6571/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-11170605*S
fNRL
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170593*
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
"dense_6572/StatefulPartitionedCallStatefulPartitionedCall%dropout_2620/PartitionedCall:output:0)dense_6572_statefulpartitionedcall_args_1)dense_6572_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170627*Q
fLRJ
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170621*
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
IdentityIdentity+dense_6572/StatefulPartitionedCall:output:0#^dense_6570/StatefulPartitionedCall#^dense_6571/StatefulPartitionedCall#^dense_6572/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2H
"dense_6570/StatefulPartitionedCall"dense_6570/StatefulPartitionedCall2H
"dense_6571/StatefulPartitionedCall"dense_6571/StatefulPartitionedCall2H
"dense_6572/StatefulPartitionedCall"dense_6572/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_6570_input: : 
?
i
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170873

inputs
identity?Q
dropout/rateConst*
valueB
 *???=*
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
&__inference_signature_wrapper_11170726
dense_6570_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_6570_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-11170717*,
f'R%
#__inference__wrapped_model_11170504*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_6570_input: : 
?
?
-__inference_dense_6571_layer_call_fn_11170853

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-11170555*Q
fLRJ
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170549*
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
?/
?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170769

inputs-
)dense_6570_matmul_readvariableop_resource.
*dense_6570_biasadd_readvariableop_resource-
)dense_6571_matmul_readvariableop_resource.
*dense_6571_biasadd_readvariableop_resource-
)dense_6572_matmul_readvariableop_resource.
*dense_6572_biasadd_readvariableop_resource
identity??!dense_6570/BiasAdd/ReadVariableOp? dense_6570/MatMul/ReadVariableOp?!dense_6571/BiasAdd/ReadVariableOp? dense_6571/MatMul/ReadVariableOp?!dense_6572/BiasAdd/ReadVariableOp? dense_6572/MatMul/ReadVariableOp?
 dense_6570/MatMul/ReadVariableOpReadVariableOp)dense_6570_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@
dense_6570/MatMulMatMulinputs(dense_6570/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_6570/BiasAdd/ReadVariableOpReadVariableOp*dense_6570_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_6570/BiasAddBiasAdddense_6570/MatMul:product:0)dense_6570/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_6570/ReluReludense_6570/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
 dense_6571/MatMul/ReadVariableOpReadVariableOp)dense_6571_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
dense_6571/MatMulMatMuldense_6570/Relu:activations:0(dense_6571/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
!dense_6571/BiasAdd/ReadVariableOpReadVariableOp*dense_6571_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_6571/BiasAddBiasAdddense_6571/MatMul:product:0)dense_6571/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@f
dense_6571/ReluReludense_6571/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@^
dropout_2620/dropout/rateConst*
valueB
 *???=*
dtype0*
_output_shapes
: g
dropout_2620/dropout/ShapeShapedense_6571/Relu:activations:0*
T0*
_output_shapes
:l
'dropout_2620/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_2620/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
1dropout_2620/dropout/random_uniform/RandomUniformRandomUniform#dropout_2620/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:?????????@?
'dropout_2620/dropout/random_uniform/subSub0dropout_2620/dropout/random_uniform/max:output:00dropout_2620/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
'dropout_2620/dropout/random_uniform/mulMul:dropout_2620/dropout/random_uniform/RandomUniform:output:0+dropout_2620/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@?
#dropout_2620/dropout/random_uniformAdd+dropout_2620/dropout/random_uniform/mul:z:00dropout_2620/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@_
dropout_2620/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_2620/dropout/subSub#dropout_2620/dropout/sub/x:output:0"dropout_2620/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_2620/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_2620/dropout/truedivRealDiv'dropout_2620/dropout/truediv/x:output:0dropout_2620/dropout/sub:z:0*
T0*
_output_shapes
: ?
!dropout_2620/dropout/GreaterEqualGreaterEqual'dropout_2620/dropout/random_uniform:z:0"dropout_2620/dropout/rate:output:0*
T0*'
_output_shapes
:?????????@?
dropout_2620/dropout/mulMuldense_6571/Relu:activations:0 dropout_2620/dropout/truediv:z:0*
T0*'
_output_shapes
:?????????@?
dropout_2620/dropout/CastCast%dropout_2620/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????@?
dropout_2620/dropout/mul_1Muldropout_2620/dropout/mul:z:0dropout_2620/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
 dense_6572/MatMul/ReadVariableOpReadVariableOp)dense_6572_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
dense_6572/MatMulMatMuldropout_2620/dropout/mul_1:z:0(dense_6572/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
!dense_6572/BiasAdd/ReadVariableOpReadVariableOp*dense_6572_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_6572/BiasAddBiasAdddense_6572/MatMul:product:0)dense_6572/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????f
dense_6572/ReluReludense_6572/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_6572/Relu:activations:0"^dense_6570/BiasAdd/ReadVariableOp!^dense_6570/MatMul/ReadVariableOp"^dense_6571/BiasAdd/ReadVariableOp!^dense_6571/MatMul/ReadVariableOp"^dense_6572/BiasAdd/ReadVariableOp!^dense_6572/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_6571/MatMul/ReadVariableOp dense_6571/MatMul/ReadVariableOp2F
!dense_6572/BiasAdd/ReadVariableOp!dense_6572/BiasAdd/ReadVariableOp2D
 dense_6570/MatMul/ReadVariableOp dense_6570/MatMul/ReadVariableOp2F
!dense_6571/BiasAdd/ReadVariableOp!dense_6571/BiasAdd/ReadVariableOp2D
 dense_6572/MatMul/ReadVariableOp dense_6572/MatMul/ReadVariableOp2F
!dense_6570/BiasAdd/ReadVariableOp!dense_6570/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?
h
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170878

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
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
M
dense_6570_input9
"serving_default_dense_6570_input:0?????????>

dense_65720
StatefulPartitionedCall:0?????????tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ɣ
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
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_1322", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1322", "layers": [{"class_name": "Dense", "config": {"name": "dense_6570", "trainable": true, "batch_input_shape": [null, 2], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_6571", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2620", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6572", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1322", "layers": [{"class_name": "Dense", "config": {"name": "dense_6570", "trainable": true, "batch_input_shape": [null, 2], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_6571", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2620", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6572", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "dense_6570_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 2], "config": {"batch_input_shape": [null, 2], "dtype": "float32", "sparse": false, "name": "dense_6570_input"}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6570", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 2], "config": {"name": "dense_6570", "trainable": true, "batch_input_shape": [null, 2], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6571", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_6571", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_2620", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2620", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_6572", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_6572", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
&iter
	'decay
(learning_rate
)rho
accum_gradN
accum_gradO
accum_gradP
accum_gradQ 
accum_gradR!
accum_gradS	accum_varT	accum_varU	accum_varV	accum_varW 	accum_varX!	accum_varY"
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

*layers
	variables
+metrics
,layer_regularization_losses
	trainable_variables
-non_trainable_variables
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

.layers
	variables
/metrics
0layer_regularization_losses
trainable_variables
1non_trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
#:!@2dense_6570/kernel
:@2dense_6570/bias
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

2layers
	variables
3metrics
4layer_regularization_losses
trainable_variables
5non_trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
#:!@@2dense_6571/kernel
:@2dense_6571/bias
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

6layers
	variables
7metrics
8layer_regularization_losses
trainable_variables
9non_trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses

:layers
	variables
;metrics
<layer_regularization_losses
trainable_variables
=non_trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
#:!@2dense_6572/kernel
:2dense_6572/bias
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

>layers
#	variables
?metrics
@layer_regularization_losses
$trainable_variables
Anon_trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
<
0
1
2
3"
trackable_list_wrapper
'
B0"
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
	Ctotal
	Dcount
E
_fn_kwargs
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h__call__
*i&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fregularization_losses

Jlayers
G	variables
Kmetrics
Llayer_regularization_losses
Htrainable_variables
Mnon_trainable_variables
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
5:3@2%Adadelta/dense_6570/kernel/accum_grad
/:-@2#Adadelta/dense_6570/bias/accum_grad
5:3@@2%Adadelta/dense_6571/kernel/accum_grad
/:-@2#Adadelta/dense_6571/bias/accum_grad
5:3@2%Adadelta/dense_6572/kernel/accum_grad
/:-2#Adadelta/dense_6572/bias/accum_grad
4:2@2$Adadelta/dense_6570/kernel/accum_var
.:,@2"Adadelta/dense_6570/bias/accum_var
4:2@@2$Adadelta/dense_6571/kernel/accum_var
.:,@2"Adadelta/dense_6571/bias/accum_var
4:2@2$Adadelta/dense_6572/kernel/accum_var
.:,2"Adadelta/dense_6572/bias/accum_var
?2?
2__inference_sequential_1322_layer_call_fn_11170682
2__inference_sequential_1322_layer_call_fn_11170806
2__inference_sequential_1322_layer_call_fn_11170817
2__inference_sequential_1322_layer_call_fn_11170710?
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
#__inference__wrapped_model_11170504?
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
dense_6570_input?????????
?2?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170769
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170795
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170655
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170639?
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
-__inference_dense_6570_layer_call_fn_11170835?
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
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170828?
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
-__inference_dense_6571_layer_call_fn_11170853?
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
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170846?
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
/__inference_dropout_2620_layer_call_fn_11170888
/__inference_dropout_2620_layer_call_fn_11170883?
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
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170873
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170878?
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
-__inference_dense_6572_layer_call_fn_11170906?
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
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170899?
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
&__inference_signature_wrapper_11170726dense_6570_input
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
-__inference_dense_6572_layer_call_fn_11170906O !/?,
%?"
 ?
inputs?????????@
? "???????????
2__inference_sequential_1322_layer_call_fn_11170682e !A?>
7?4
*?'
dense_6570_input?????????
p

 
? "???????????
H__inference_dense_6570_layer_call_and_return_conditional_losses_11170828\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? ?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170795h !7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_dense_6571_layer_call_and_return_conditional_losses_11170846\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170873\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
J__inference_dropout_2620_layer_call_and_return_conditional_losses_11170878\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? ?
2__inference_sequential_1322_layer_call_fn_11170806[ !7?4
-?*
 ?
inputs?????????
p

 
? "???????????
-__inference_dense_6571_layer_call_fn_11170853O/?,
%?"
 ?
inputs?????????@
? "??????????@?
#__inference__wrapped_model_11170504| !9?6
/?,
*?'
dense_6570_input?????????
? "7?4
2

dense_6572$?!

dense_6572??????????
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170639r !A?>
7?4
*?'
dense_6570_input?????????
p

 
? "%?"
?
0?????????
? ?
2__inference_sequential_1322_layer_call_fn_11170817[ !7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
2__inference_sequential_1322_layer_call_fn_11170710e !A?>
7?4
*?'
dense_6570_input?????????
p 

 
? "???????????
&__inference_signature_wrapper_11170726? !M?J
? 
C?@
>
dense_6570_input*?'
dense_6570_input?????????"7?4
2

dense_6572$?!

dense_6572??????????
/__inference_dropout_2620_layer_call_fn_11170883O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170655r !A?>
7?4
*?'
dense_6570_input?????????
p 

 
? "%?"
?
0?????????
? ?
-__inference_dense_6570_layer_call_fn_11170835O/?,
%?"
 ?
inputs?????????
? "??????????@?
M__inference_sequential_1322_layer_call_and_return_conditional_losses_11170769h !7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_dense_6572_layer_call_and_return_conditional_losses_11170899\ !/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
/__inference_dropout_2620_layer_call_fn_11170888O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@