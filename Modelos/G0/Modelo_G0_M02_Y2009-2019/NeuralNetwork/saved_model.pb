ٰ
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
|
dense_483/kernelVarHandleOp*
shape
:@*!
shared_namedense_483/kernel*
dtype0*
_output_shapes
: 
u
$dense_483/kernel/Read/ReadVariableOpReadVariableOpdense_483/kernel*
dtype0*
_output_shapes

:@
t
dense_483/biasVarHandleOp*
shape:@*
shared_namedense_483/bias*
dtype0*
_output_shapes
: 
m
"dense_483/bias/Read/ReadVariableOpReadVariableOpdense_483/bias*
dtype0*
_output_shapes
:@
|
dense_484/kernelVarHandleOp*
shape
:@@*!
shared_namedense_484/kernel*
dtype0*
_output_shapes
: 
u
$dense_484/kernel/Read/ReadVariableOpReadVariableOpdense_484/kernel*
dtype0*
_output_shapes

:@@
t
dense_484/biasVarHandleOp*
shape:@*
shared_namedense_484/bias*
dtype0*
_output_shapes
: 
m
"dense_484/bias/Read/ReadVariableOpReadVariableOpdense_484/bias*
dtype0*
_output_shapes
:@
|
dense_485/kernelVarHandleOp*
shape
:@*!
shared_namedense_485/kernel*
dtype0*
_output_shapes
: 
u
$dense_485/kernel/Read/ReadVariableOpReadVariableOpdense_485/kernel*
dtype0*
_output_shapes

:@
t
dense_485/biasVarHandleOp*
shape:*
shared_namedense_485/bias*
dtype0*
_output_shapes
: 
m
"dense_485/bias/Read/ReadVariableOpReadVariableOpdense_485/bias*
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
Nadam/dense_483/kernel/mVarHandleOp*
shape
:@*)
shared_nameNadam/dense_483/kernel/m*
dtype0*
_output_shapes
: 
?
,Nadam/dense_483/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_483/kernel/m*
dtype0*
_output_shapes

:@
?
Nadam/dense_483/bias/mVarHandleOp*
shape:@*'
shared_nameNadam/dense_483/bias/m*
dtype0*
_output_shapes
: 
}
*Nadam/dense_483/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_483/bias/m*
dtype0*
_output_shapes
:@
?
Nadam/dense_484/kernel/mVarHandleOp*
shape
:@@*)
shared_nameNadam/dense_484/kernel/m*
dtype0*
_output_shapes
: 
?
,Nadam/dense_484/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_484/kernel/m*
dtype0*
_output_shapes

:@@
?
Nadam/dense_484/bias/mVarHandleOp*
shape:@*'
shared_nameNadam/dense_484/bias/m*
dtype0*
_output_shapes
: 
}
*Nadam/dense_484/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_484/bias/m*
dtype0*
_output_shapes
:@
?
Nadam/dense_485/kernel/mVarHandleOp*
shape
:@*)
shared_nameNadam/dense_485/kernel/m*
dtype0*
_output_shapes
: 
?
,Nadam/dense_485/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_485/kernel/m*
dtype0*
_output_shapes

:@
?
Nadam/dense_485/bias/mVarHandleOp*
shape:*'
shared_nameNadam/dense_485/bias/m*
dtype0*
_output_shapes
: 
}
*Nadam/dense_485/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_485/bias/m*
dtype0*
_output_shapes
:
?
Nadam/dense_483/kernel/vVarHandleOp*
shape
:@*)
shared_nameNadam/dense_483/kernel/v*
dtype0*
_output_shapes
: 
?
,Nadam/dense_483/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_483/kernel/v*
dtype0*
_output_shapes

:@
?
Nadam/dense_483/bias/vVarHandleOp*
shape:@*'
shared_nameNadam/dense_483/bias/v*
dtype0*
_output_shapes
: 
}
*Nadam/dense_483/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_483/bias/v*
dtype0*
_output_shapes
:@
?
Nadam/dense_484/kernel/vVarHandleOp*
shape
:@@*)
shared_nameNadam/dense_484/kernel/v*
dtype0*
_output_shapes
: 
?
,Nadam/dense_484/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_484/kernel/v*
dtype0*
_output_shapes

:@@
?
Nadam/dense_484/bias/vVarHandleOp*
shape:@*'
shared_nameNadam/dense_484/bias/v*
dtype0*
_output_shapes
: 
}
*Nadam/dense_484/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_484/bias/v*
dtype0*
_output_shapes
:@
?
Nadam/dense_485/kernel/vVarHandleOp*
shape
:@*)
shared_nameNadam/dense_485/kernel/v*
dtype0*
_output_shapes
: 
?
,Nadam/dense_485/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_485/kernel/v*
dtype0*
_output_shapes

:@
?
Nadam/dense_485/bias/vVarHandleOp*
shape:*'
shared_nameNadam/dense_485/bias/v*
dtype0*
_output_shapes
: 
}
*Nadam/dense_485/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_485/bias/v*
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
\Z
VARIABLE_VALUEdense_483/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_483/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_484/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_484/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
\Z
VARIABLE_VALUEdense_485/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_485/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
?~
VARIABLE_VALUENadam/dense_483/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_483/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_484/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_484/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_485/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_485/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_483/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_483/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_484/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_484/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUENadam/dense_485/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUENadam/dense_485/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
?
serving_default_dense_483_inputPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_483_inputdense_483/kerneldense_483/biasdense_484/kerneldense_484/biasdense_485/kerneldense_485/bias*-
_gradient_op_typePartitionedCall-826795*-
f(R&
$__inference_signature_wrapper_826559*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dense_483/kernel/Read/ReadVariableOp"dense_483/bias/Read/ReadVariableOp$dense_484/kernel/Read/ReadVariableOp"dense_484/bias/Read/ReadVariableOp$dense_485/kernel/Read/ReadVariableOp"dense_485/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Nadam/dense_483/kernel/m/Read/ReadVariableOp*Nadam/dense_483/bias/m/Read/ReadVariableOp,Nadam/dense_484/kernel/m/Read/ReadVariableOp*Nadam/dense_484/bias/m/Read/ReadVariableOp,Nadam/dense_485/kernel/m/Read/ReadVariableOp*Nadam/dense_485/bias/m/Read/ReadVariableOp,Nadam/dense_483/kernel/v/Read/ReadVariableOp*Nadam/dense_483/bias/v/Read/ReadVariableOp,Nadam/dense_484/kernel/v/Read/ReadVariableOp*Nadam/dense_484/bias/v/Read/ReadVariableOp,Nadam/dense_485/kernel/v/Read/ReadVariableOp*Nadam/dense_485/bias/v/Read/ReadVariableOpConst*-
_gradient_op_typePartitionedCall-826843*(
f#R!
__inference__traced_save_826842*
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
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_483/kerneldense_483/biasdense_484/kerneldense_484/biasdense_485/kerneldense_485/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/dense_483/kernel/mNadam/dense_483/bias/mNadam/dense_484/kernel/mNadam/dense_484/bias/mNadam/dense_485/kernel/mNadam/dense_485/bias/mNadam/dense_483/kernel/vNadam/dense_483/bias/vNadam/dense_484/kernel/vNadam/dense_484/bias/vNadam/dense_485/kernel/vNadam/dense_485/bias/v*-
_gradient_op_typePartitionedCall-826934*+
f&R$
"__inference__traced_restore_826933*
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
: ??
?
?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826628

inputs,
(dense_483_matmul_readvariableop_resource-
)dense_483_biasadd_readvariableop_resource,
(dense_484_matmul_readvariableop_resource-
)dense_484_biasadd_readvariableop_resource,
(dense_485_matmul_readvariableop_resource-
)dense_485_biasadd_readvariableop_resource
identity?? dense_483/BiasAdd/ReadVariableOp?dense_483/MatMul/ReadVariableOp? dense_484/BiasAdd/ReadVariableOp?dense_484/MatMul/ReadVariableOp? dense_485/BiasAdd/ReadVariableOp?dense_485/MatMul/ReadVariableOp?
dense_483/MatMul/ReadVariableOpReadVariableOp(dense_483_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@}
dense_483/MatMulMatMulinputs'dense_483/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
 dense_483/BiasAdd/ReadVariableOpReadVariableOp)dense_483_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_483/BiasAddBiasAdddense_483/MatMul:product:0(dense_483/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_483/ReluReludense_483/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_484/MatMul/ReadVariableOpReadVariableOp(dense_484_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
dense_484/MatMulMatMuldense_483/Relu:activations:0'dense_484/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
 dense_484/BiasAdd/ReadVariableOpReadVariableOp)dense_484_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_484/BiasAddBiasAdddense_484/MatMul:product:0(dense_484/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_484/ReluReludense_484/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@p
dropout_192/IdentityIdentitydense_484/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
dense_485/MatMul/ReadVariableOpReadVariableOp(dense_485_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
dense_485/MatMulMatMuldropout_192/Identity:output:0'dense_485/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_485/BiasAdd/ReadVariableOpReadVariableOp)dense_485_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_485/BiasAddBiasAdddense_485/MatMul:product:0(dense_485/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_485/ReluReludense_485/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_485/Relu:activations:0!^dense_483/BiasAdd/ReadVariableOp ^dense_483/MatMul/ReadVariableOp!^dense_484/BiasAdd/ReadVariableOp ^dense_484/MatMul/ReadVariableOp!^dense_485/BiasAdd/ReadVariableOp ^dense_485/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_483/BiasAdd/ReadVariableOp dense_483/BiasAdd/ReadVariableOp2B
dense_484/MatMul/ReadVariableOpdense_484/MatMul/ReadVariableOp2B
dense_483/MatMul/ReadVariableOpdense_483/MatMul/ReadVariableOp2B
dense_485/MatMul/ReadVariableOpdense_485/MatMul/ReadVariableOp2D
 dense_485/BiasAdd/ReadVariableOp dense_485/BiasAdd/ReadVariableOp2D
 dense_484/BiasAdd/ReadVariableOp dense_484/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
E__inference_dense_483_layer_call_and_return_conditional_losses_826661

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
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
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
?
*__inference_dense_485_layer_call_fn_826739

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826459*N
fIRG
E__inference_dense_485_layer_call_and_return_conditional_losses_826453*
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
E__inference_dense_485_layer_call_and_return_conditional_losses_826732

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
?
?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826504

inputs,
(dense_483_statefulpartitionedcall_args_1,
(dense_483_statefulpartitionedcall_args_2,
(dense_484_statefulpartitionedcall_args_1,
(dense_484_statefulpartitionedcall_args_2,
(dense_485_statefulpartitionedcall_args_1,
(dense_485_statefulpartitionedcall_args_2
identity??!dense_483/StatefulPartitionedCall?!dense_484/StatefulPartitionedCall?!dense_485/StatefulPartitionedCall?#dropout_192/StatefulPartitionedCall?
!dense_483/StatefulPartitionedCallStatefulPartitionedCallinputs(dense_483_statefulpartitionedcall_args_1(dense_483_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826359*N
fIRG
E__inference_dense_483_layer_call_and_return_conditional_losses_826353*
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
!dense_484/StatefulPartitionedCallStatefulPartitionedCall*dense_483/StatefulPartitionedCall:output:0(dense_484_statefulpartitionedcall_args_1(dense_484_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826387*N
fIRG
E__inference_dense_484_layer_call_and_return_conditional_losses_826381*
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
#dropout_192/StatefulPartitionedCallStatefulPartitionedCall*dense_484/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-826429*P
fKRI
G__inference_dropout_192_layer_call_and_return_conditional_losses_826418*
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
!dense_485/StatefulPartitionedCallStatefulPartitionedCall,dropout_192/StatefulPartitionedCall:output:0(dense_485_statefulpartitionedcall_args_1(dense_485_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826459*N
fIRG
E__inference_dense_485_layer_call_and_return_conditional_losses_826453*
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
IdentityIdentity*dense_485/StatefulPartitionedCall:output:0"^dense_483/StatefulPartitionedCall"^dense_484/StatefulPartitionedCall"^dense_485/StatefulPartitionedCall$^dropout_192/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_483/StatefulPartitionedCall!dense_483/StatefulPartitionedCall2F
!dense_484/StatefulPartitionedCall!dense_484/StatefulPartitionedCall2F
!dense_485/StatefulPartitionedCall!dense_485/StatefulPartitionedCall2J
#dropout_192/StatefulPartitionedCall#dropout_192/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?	
?
.__inference_sequential_97_layer_call_fn_826514
dense_483_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_483_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-826505*R
fMRK
I__inference_sequential_97_layer_call_and_return_conditional_losses_826504*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :/ +
)
_user_specified_namedense_483_input: : 
?	
?
.__inference_sequential_97_layer_call_fn_826542
dense_483_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_483_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-826533*R
fMRK
I__inference_sequential_97_layer_call_and_return_conditional_losses_826532*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :/ +
)
_user_specified_namedense_483_input: : 
?.
?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826602

inputs,
(dense_483_matmul_readvariableop_resource-
)dense_483_biasadd_readvariableop_resource,
(dense_484_matmul_readvariableop_resource-
)dense_484_biasadd_readvariableop_resource,
(dense_485_matmul_readvariableop_resource-
)dense_485_biasadd_readvariableop_resource
identity?? dense_483/BiasAdd/ReadVariableOp?dense_483/MatMul/ReadVariableOp? dense_484/BiasAdd/ReadVariableOp?dense_484/MatMul/ReadVariableOp? dense_485/BiasAdd/ReadVariableOp?dense_485/MatMul/ReadVariableOp?
dense_483/MatMul/ReadVariableOpReadVariableOp(dense_483_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@}
dense_483/MatMulMatMulinputs'dense_483/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
 dense_483/BiasAdd/ReadVariableOpReadVariableOp)dense_483_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_483/BiasAddBiasAdddense_483/MatMul:product:0(dense_483/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_483/ReluReludense_483/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
dense_484/MatMul/ReadVariableOpReadVariableOp(dense_484_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
dense_484/MatMulMatMuldense_483/Relu:activations:0'dense_484/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
 dense_484/BiasAdd/ReadVariableOpReadVariableOp)dense_484_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
dense_484/BiasAddBiasAdddense_484/MatMul:product:0(dense_484/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@d
dense_484/ReluReludense_484/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@]
dropout_192/dropout/rateConst*
valueB
 *  ?>*
dtype0*
_output_shapes
: e
dropout_192/dropout/ShapeShapedense_484/Relu:activations:0*
T0*
_output_shapes
:k
&dropout_192/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: k
&dropout_192/dropout/random_uniform/maxConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
0dropout_192/dropout/random_uniform/RandomUniformRandomUniform"dropout_192/dropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:?????????@?
&dropout_192/dropout/random_uniform/subSub/dropout_192/dropout/random_uniform/max:output:0/dropout_192/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ?
&dropout_192/dropout/random_uniform/mulMul9dropout_192/dropout/random_uniform/RandomUniform:output:0*dropout_192/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:?????????@?
"dropout_192/dropout/random_uniformAdd*dropout_192/dropout/random_uniform/mul:z:0/dropout_192/dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:?????????@^
dropout_192/dropout/sub/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_192/dropout/subSub"dropout_192/dropout/sub/x:output:0!dropout_192/dropout/rate:output:0*
T0*
_output_shapes
: b
dropout_192/dropout/truediv/xConst*
valueB
 *  ??*
dtype0*
_output_shapes
: ?
dropout_192/dropout/truedivRealDiv&dropout_192/dropout/truediv/x:output:0dropout_192/dropout/sub:z:0*
T0*
_output_shapes
: ?
 dropout_192/dropout/GreaterEqualGreaterEqual&dropout_192/dropout/random_uniform:z:0!dropout_192/dropout/rate:output:0*
T0*'
_output_shapes
:?????????@?
dropout_192/dropout/mulMuldense_484/Relu:activations:0dropout_192/dropout/truediv:z:0*
T0*'
_output_shapes
:?????????@?
dropout_192/dropout/CastCast$dropout_192/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:?????????@?
dropout_192/dropout/mul_1Muldropout_192/dropout/mul:z:0dropout_192/dropout/Cast:y:0*
T0*'
_output_shapes
:?????????@?
dense_485/MatMul/ReadVariableOpReadVariableOp(dense_485_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
dense_485/MatMulMatMuldropout_192/dropout/mul_1:z:0'dense_485/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
 dense_485/BiasAdd/ReadVariableOpReadVariableOp)dense_485_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
dense_485/BiasAddBiasAdddense_485/MatMul:product:0(dense_485/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????d
dense_485/ReluReludense_485/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentitydense_485/Relu:activations:0!^dense_483/BiasAdd/ReadVariableOp ^dense_483/MatMul/ReadVariableOp!^dense_484/BiasAdd/ReadVariableOp ^dense_484/MatMul/ReadVariableOp!^dense_485/BiasAdd/ReadVariableOp ^dense_485/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2D
 dense_483/BiasAdd/ReadVariableOp dense_483/BiasAdd/ReadVariableOp2B
dense_484/MatMul/ReadVariableOpdense_484/MatMul/ReadVariableOp2B
dense_483/MatMul/ReadVariableOpdense_483/MatMul/ReadVariableOp2B
dense_485/MatMul/ReadVariableOpdense_485/MatMul/ReadVariableOp2D
 dense_485/BiasAdd/ReadVariableOp dense_485/BiasAdd/ReadVariableOp2D
 dense_484/BiasAdd/ReadVariableOp dense_484/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
?
H
,__inference_dropout_192_layer_call_fn_826721

inputs
identity?
PartitionedCallPartitionedCallinputs*-
_gradient_op_typePartitionedCall-826437*P
fKRI
G__inference_dropout_192_layer_call_and_return_conditional_losses_826425*
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
?
?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826471
dense_483_input,
(dense_483_statefulpartitionedcall_args_1,
(dense_483_statefulpartitionedcall_args_2,
(dense_484_statefulpartitionedcall_args_1,
(dense_484_statefulpartitionedcall_args_2,
(dense_485_statefulpartitionedcall_args_1,
(dense_485_statefulpartitionedcall_args_2
identity??!dense_483/StatefulPartitionedCall?!dense_484/StatefulPartitionedCall?!dense_485/StatefulPartitionedCall?#dropout_192/StatefulPartitionedCall?
!dense_483/StatefulPartitionedCallStatefulPartitionedCalldense_483_input(dense_483_statefulpartitionedcall_args_1(dense_483_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826359*N
fIRG
E__inference_dense_483_layer_call_and_return_conditional_losses_826353*
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
!dense_484/StatefulPartitionedCallStatefulPartitionedCall*dense_483/StatefulPartitionedCall:output:0(dense_484_statefulpartitionedcall_args_1(dense_484_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826387*N
fIRG
E__inference_dense_484_layer_call_and_return_conditional_losses_826381*
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
#dropout_192/StatefulPartitionedCallStatefulPartitionedCall*dense_484/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-826429*P
fKRI
G__inference_dropout_192_layer_call_and_return_conditional_losses_826418*
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
!dense_485/StatefulPartitionedCallStatefulPartitionedCall,dropout_192/StatefulPartitionedCall:output:0(dense_485_statefulpartitionedcall_args_1(dense_485_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826459*N
fIRG
E__inference_dense_485_layer_call_and_return_conditional_losses_826453*
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
IdentityIdentity*dense_485/StatefulPartitionedCall:output:0"^dense_483/StatefulPartitionedCall"^dense_484/StatefulPartitionedCall"^dense_485/StatefulPartitionedCall$^dropout_192/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_483/StatefulPartitionedCall!dense_483/StatefulPartitionedCall2F
!dense_484/StatefulPartitionedCall!dense_484/StatefulPartitionedCall2F
!dense_485/StatefulPartitionedCall!dense_485/StatefulPartitionedCall2J
#dropout_192/StatefulPartitionedCall#dropout_192/StatefulPartitionedCall: : : : :/ +
)
_user_specified_namedense_483_input: : 
?
f
G__inference_dropout_192_layer_call_and_return_conditional_losses_826418

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
E__inference_dense_483_layer_call_and_return_conditional_losses_826353

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
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
:?????????::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
?
e
G__inference_dropout_192_layer_call_and_return_conditional_losses_826425

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
?
f
G__inference_dropout_192_layer_call_and_return_conditional_losses_826706

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
?
?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826487
dense_483_input,
(dense_483_statefulpartitionedcall_args_1,
(dense_483_statefulpartitionedcall_args_2,
(dense_484_statefulpartitionedcall_args_1,
(dense_484_statefulpartitionedcall_args_2,
(dense_485_statefulpartitionedcall_args_1,
(dense_485_statefulpartitionedcall_args_2
identity??!dense_483/StatefulPartitionedCall?!dense_484/StatefulPartitionedCall?!dense_485/StatefulPartitionedCall?
!dense_483/StatefulPartitionedCallStatefulPartitionedCalldense_483_input(dense_483_statefulpartitionedcall_args_1(dense_483_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826359*N
fIRG
E__inference_dense_483_layer_call_and_return_conditional_losses_826353*
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
!dense_484/StatefulPartitionedCallStatefulPartitionedCall*dense_483/StatefulPartitionedCall:output:0(dense_484_statefulpartitionedcall_args_1(dense_484_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826387*N
fIRG
E__inference_dense_484_layer_call_and_return_conditional_losses_826381*
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
dropout_192/PartitionedCallPartitionedCall*dense_484/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-826437*P
fKRI
G__inference_dropout_192_layer_call_and_return_conditional_losses_826425*
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
!dense_485/StatefulPartitionedCallStatefulPartitionedCall$dropout_192/PartitionedCall:output:0(dense_485_statefulpartitionedcall_args_1(dense_485_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826459*N
fIRG
E__inference_dense_485_layer_call_and_return_conditional_losses_826453*
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
IdentityIdentity*dense_485/StatefulPartitionedCall:output:0"^dense_483/StatefulPartitionedCall"^dense_484/StatefulPartitionedCall"^dense_485/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_483/StatefulPartitionedCall!dense_483/StatefulPartitionedCall2F
!dense_484/StatefulPartitionedCall!dense_484/StatefulPartitionedCall2F
!dense_485/StatefulPartitionedCall!dense_485/StatefulPartitionedCall: : : : :/ +
)
_user_specified_namedense_483_input: : 
?
e
,__inference_dropout_192_layer_call_fn_826716

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*-
_gradient_op_typePartitionedCall-826429*P
fKRI
G__inference_dropout_192_layer_call_and_return_conditional_losses_826418*
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
?
E__inference_dense_484_layer_call_and_return_conditional_losses_826679

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
E__inference_dense_485_layer_call_and_return_conditional_losses_826453

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
?f
?
"__inference__traced_restore_826933
file_prefix%
!assignvariableop_dense_483_kernel%
!assignvariableop_1_dense_483_bias'
#assignvariableop_2_dense_484_kernel%
!assignvariableop_3_dense_484_bias'
#assignvariableop_4_dense_485_kernel%
!assignvariableop_5_dense_485_bias!
assignvariableop_6_nadam_iter#
assignvariableop_7_nadam_beta_1#
assignvariableop_8_nadam_beta_2"
assignvariableop_9_nadam_decay+
'assignvariableop_10_nadam_learning_rate,
(assignvariableop_11_nadam_momentum_cache
assignvariableop_12_total
assignvariableop_13_count0
,assignvariableop_14_nadam_dense_483_kernel_m.
*assignvariableop_15_nadam_dense_483_bias_m0
,assignvariableop_16_nadam_dense_484_kernel_m.
*assignvariableop_17_nadam_dense_484_bias_m0
,assignvariableop_18_nadam_dense_485_kernel_m.
*assignvariableop_19_nadam_dense_485_bias_m0
,assignvariableop_20_nadam_dense_483_kernel_v.
*assignvariableop_21_nadam_dense_483_bias_v0
,assignvariableop_22_nadam_dense_484_kernel_v.
*assignvariableop_23_nadam_dense_484_bias_v0
,assignvariableop_24_nadam_dense_485_kernel_v.
*assignvariableop_25_nadam_dense_485_bias_v
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
:}
AssignVariableOpAssignVariableOp!assignvariableop_dense_483_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dense_483_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_484_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_484_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_485_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_485_biasIdentity_5:output:0*
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
AssignVariableOp_14AssignVariableOp,assignvariableop_14_nadam_dense_483_kernel_mIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_nadam_dense_483_bias_mIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_nadam_dense_484_kernel_mIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_nadam_dense_484_bias_mIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_nadam_dense_485_kernel_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_nadam_dense_485_bias_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_nadam_dense_483_kernel_vIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_nadam_dense_483_bias_vIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_nadam_dense_484_kernel_vIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_nadam_dense_484_bias_vIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_nadam_dense_485_kernel_vIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_nadam_dense_485_bias_vIdentity_25:output:0*
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
j: ::::::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252$
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
 : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : 
?	
?
.__inference_sequential_97_layer_call_fn_826639

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-826505*R
fMRK
I__inference_sequential_97_layer_call_and_return_conditional_losses_826504*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?%
?
!__inference__wrapped_model_826336
dense_483_input:
6sequential_97_dense_483_matmul_readvariableop_resource;
7sequential_97_dense_483_biasadd_readvariableop_resource:
6sequential_97_dense_484_matmul_readvariableop_resource;
7sequential_97_dense_484_biasadd_readvariableop_resource:
6sequential_97_dense_485_matmul_readvariableop_resource;
7sequential_97_dense_485_biasadd_readvariableop_resource
identity??.sequential_97/dense_483/BiasAdd/ReadVariableOp?-sequential_97/dense_483/MatMul/ReadVariableOp?.sequential_97/dense_484/BiasAdd/ReadVariableOp?-sequential_97/dense_484/MatMul/ReadVariableOp?.sequential_97/dense_485/BiasAdd/ReadVariableOp?-sequential_97/dense_485/MatMul/ReadVariableOp?
-sequential_97/dense_483/MatMul/ReadVariableOpReadVariableOp6sequential_97_dense_483_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
sequential_97/dense_483/MatMulMatMuldense_483_input5sequential_97/dense_483/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
.sequential_97/dense_483/BiasAdd/ReadVariableOpReadVariableOp7sequential_97_dense_483_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
sequential_97/dense_483/BiasAddBiasAdd(sequential_97/dense_483/MatMul:product:06sequential_97/dense_483/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_97/dense_483/ReluRelu(sequential_97/dense_483/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
-sequential_97/dense_484/MatMul/ReadVariableOpReadVariableOp6sequential_97_dense_484_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@@?
sequential_97/dense_484/MatMulMatMul*sequential_97/dense_483/Relu:activations:05sequential_97/dense_484/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
.sequential_97/dense_484/BiasAdd/ReadVariableOpReadVariableOp7sequential_97_dense_484_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@?
sequential_97/dense_484/BiasAddBiasAdd(sequential_97/dense_484/MatMul:product:06sequential_97/dense_484/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
sequential_97/dense_484/ReluRelu(sequential_97/dense_484/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@?
"sequential_97/dropout_192/IdentityIdentity*sequential_97/dense_484/Relu:activations:0*
T0*'
_output_shapes
:?????????@?
-sequential_97/dense_485/MatMul/ReadVariableOpReadVariableOp6sequential_97_dense_485_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@?
sequential_97/dense_485/MatMulMatMul+sequential_97/dropout_192/Identity:output:05sequential_97/dense_485/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.sequential_97/dense_485/BiasAdd/ReadVariableOpReadVariableOp7sequential_97_dense_485_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:?
sequential_97/dense_485/BiasAddBiasAdd(sequential_97/dense_485/MatMul:product:06sequential_97/dense_485/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_97/dense_485/ReluRelu(sequential_97/dense_485/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
IdentityIdentity*sequential_97/dense_485/Relu:activations:0/^sequential_97/dense_483/BiasAdd/ReadVariableOp.^sequential_97/dense_483/MatMul/ReadVariableOp/^sequential_97/dense_484/BiasAdd/ReadVariableOp.^sequential_97/dense_484/MatMul/ReadVariableOp/^sequential_97/dense_485/BiasAdd/ReadVariableOp.^sequential_97/dense_485/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2`
.sequential_97/dense_483/BiasAdd/ReadVariableOp.sequential_97/dense_483/BiasAdd/ReadVariableOp2^
-sequential_97/dense_483/MatMul/ReadVariableOp-sequential_97/dense_483/MatMul/ReadVariableOp2^
-sequential_97/dense_485/MatMul/ReadVariableOp-sequential_97/dense_485/MatMul/ReadVariableOp2`
.sequential_97/dense_485/BiasAdd/ReadVariableOp.sequential_97/dense_485/BiasAdd/ReadVariableOp2`
.sequential_97/dense_484/BiasAdd/ReadVariableOp.sequential_97/dense_484/BiasAdd/ReadVariableOp2^
-sequential_97/dense_484/MatMul/ReadVariableOp-sequential_97/dense_484/MatMul/ReadVariableOp: : : : :/ +
)
_user_specified_namedense_483_input: : 
?
?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826532

inputs,
(dense_483_statefulpartitionedcall_args_1,
(dense_483_statefulpartitionedcall_args_2,
(dense_484_statefulpartitionedcall_args_1,
(dense_484_statefulpartitionedcall_args_2,
(dense_485_statefulpartitionedcall_args_1,
(dense_485_statefulpartitionedcall_args_2
identity??!dense_483/StatefulPartitionedCall?!dense_484/StatefulPartitionedCall?!dense_485/StatefulPartitionedCall?
!dense_483/StatefulPartitionedCallStatefulPartitionedCallinputs(dense_483_statefulpartitionedcall_args_1(dense_483_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826359*N
fIRG
E__inference_dense_483_layer_call_and_return_conditional_losses_826353*
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
!dense_484/StatefulPartitionedCallStatefulPartitionedCall*dense_483/StatefulPartitionedCall:output:0(dense_484_statefulpartitionedcall_args_1(dense_484_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826387*N
fIRG
E__inference_dense_484_layer_call_and_return_conditional_losses_826381*
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
dropout_192/PartitionedCallPartitionedCall*dense_484/StatefulPartitionedCall:output:0*-
_gradient_op_typePartitionedCall-826437*P
fKRI
G__inference_dropout_192_layer_call_and_return_conditional_losses_826425*
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
!dense_485/StatefulPartitionedCallStatefulPartitionedCall$dropout_192/PartitionedCall:output:0(dense_485_statefulpartitionedcall_args_1(dense_485_statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826459*N
fIRG
E__inference_dense_485_layer_call_and_return_conditional_losses_826453*
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
IdentityIdentity*dense_485/StatefulPartitionedCall:output:0"^dense_483/StatefulPartitionedCall"^dense_484/StatefulPartitionedCall"^dense_485/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*>
_input_shapes-
+:?????????::::::2F
!dense_483/StatefulPartitionedCall!dense_483/StatefulPartitionedCall2F
!dense_484/StatefulPartitionedCall!dense_484/StatefulPartitionedCall2F
!dense_485/StatefulPartitionedCall!dense_485/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
*__inference_dense_484_layer_call_fn_826686

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826387*N
fIRG
E__inference_dense_484_layer_call_and_return_conditional_losses_826381*
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
?	
?
$__inference_signature_wrapper_826559
dense_483_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_483_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-826550**
f%R#
!__inference__wrapped_model_826336*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :/ +
)
_user_specified_namedense_483_input: : 
?	
?
.__inference_sequential_97_layer_call_fn_826650

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*-
_gradient_op_typePartitionedCall-826533*R
fMRK
I__inference_sequential_97_layer_call_and_return_conditional_losses_826532*
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
+:?????????::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
?
?
*__inference_dense_483_layer_call_fn_826668

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
_gradient_op_typePartitionedCall-826359*N
fIRG
E__inference_dense_483_layer_call_and_return_conditional_losses_826353*
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
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
?
e
G__inference_dropout_192_layer_call_and_return_conditional_losses_826711

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
?8
?
__inference__traced_save_826842
file_prefix/
+savev2_dense_483_kernel_read_readvariableop-
)savev2_dense_483_bias_read_readvariableop/
+savev2_dense_484_kernel_read_readvariableop-
)savev2_dense_484_bias_read_readvariableop/
+savev2_dense_485_kernel_read_readvariableop-
)savev2_dense_485_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_nadam_dense_483_kernel_m_read_readvariableop5
1savev2_nadam_dense_483_bias_m_read_readvariableop7
3savev2_nadam_dense_484_kernel_m_read_readvariableop5
1savev2_nadam_dense_484_bias_m_read_readvariableop7
3savev2_nadam_dense_485_kernel_m_read_readvariableop5
1savev2_nadam_dense_485_bias_m_read_readvariableop7
3savev2_nadam_dense_483_kernel_v_read_readvariableop5
1savev2_nadam_dense_483_bias_v_read_readvariableop7
3savev2_nadam_dense_484_kernel_v_read_readvariableop5
1savev2_nadam_dense_484_bias_v_read_readvariableop7
3savev2_nadam_dense_485_kernel_v_read_readvariableop5
1savev2_nadam_dense_485_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_8581b7e8098b4615926a89820b55bbe4/part*
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

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dense_483_kernel_read_readvariableop)savev2_dense_483_bias_read_readvariableop+savev2_dense_484_kernel_read_readvariableop)savev2_dense_484_bias_read_readvariableop+savev2_dense_485_kernel_read_readvariableop)savev2_dense_485_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_nadam_dense_483_kernel_m_read_readvariableop1savev2_nadam_dense_483_bias_m_read_readvariableop3savev2_nadam_dense_484_kernel_m_read_readvariableop1savev2_nadam_dense_484_bias_m_read_readvariableop3savev2_nadam_dense_485_kernel_m_read_readvariableop1savev2_nadam_dense_485_bias_m_read_readvariableop3savev2_nadam_dense_483_kernel_v_read_readvariableop1savev2_nadam_dense_483_bias_v_read_readvariableop3savev2_nadam_dense_484_kernel_v_read_readvariableop1savev2_nadam_dense_484_bias_v_read_readvariableop3savev2_nadam_dense_485_kernel_v_read_readvariableop1savev2_nadam_dense_485_bias_v_read_readvariableop"/device:CPU:0*(
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
?: :@:@:@@:@:@:: : : : : : : : :@:@:@@:@:@::@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : : :
 : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : 
?	
?
E__inference_dense_484_layer_call_and_return_conditional_losses_826381

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
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*?
serving_default?
K
dense_483_input8
!serving_default_dense_483_input:0?????????=
	dense_4850
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_97", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_97", "layers": [{"class_name": "Dense", "config": {"name": "dense_483", "trainable": true, "batch_input_shape": [null, 4], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_484", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_192", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_485", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_97", "layers": [{"class_name": "Dense", "config": {"name": "dense_483", "trainable": true, "batch_input_shape": [null, 4], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_484", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_192", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_485", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "InputLayer", "name": "dense_483_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 4], "config": {"batch_input_shape": [null, 4], "dtype": "float32", "sparse": false, "name": "dense_483_input"}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_483", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 4], "config": {"name": "dense_483", "trainable": true, "batch_input_shape": [null, 4], "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_484", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_484", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
?
regularization_losses
	variables
trainable_variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_192", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_192", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
?

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_485", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_485", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
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
": @2dense_483/kernel
:@2dense_483/bias
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
": @@2dense_484/kernel
:@2dense_484/bias
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
": @2dense_485/kernel
:2dense_485/bias
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
(:&@2Nadam/dense_483/kernel/m
": @2Nadam/dense_483/bias/m
(:&@@2Nadam/dense_484/kernel/m
": @2Nadam/dense_484/bias/m
(:&@2Nadam/dense_485/kernel/m
": 2Nadam/dense_485/bias/m
(:&@2Nadam/dense_483/kernel/v
": @2Nadam/dense_483/bias/v
(:&@@2Nadam/dense_484/kernel/v
": @2Nadam/dense_484/bias/v
(:&@2Nadam/dense_485/kernel/v
": 2Nadam/dense_485/bias/v
?2?
.__inference_sequential_97_layer_call_fn_826650
.__inference_sequential_97_layer_call_fn_826639
.__inference_sequential_97_layer_call_fn_826542
.__inference_sequential_97_layer_call_fn_826514?
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
!__inference__wrapped_model_826336?
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
annotations? *.?+
)?&
dense_483_input?????????
?2?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826471
I__inference_sequential_97_layer_call_and_return_conditional_losses_826487
I__inference_sequential_97_layer_call_and_return_conditional_losses_826602
I__inference_sequential_97_layer_call_and_return_conditional_losses_826628?
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
*__inference_dense_483_layer_call_fn_826668?
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
E__inference_dense_483_layer_call_and_return_conditional_losses_826661?
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
*__inference_dense_484_layer_call_fn_826686?
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
E__inference_dense_484_layer_call_and_return_conditional_losses_826679?
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
,__inference_dropout_192_layer_call_fn_826716
,__inference_dropout_192_layer_call_fn_826721?
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
G__inference_dropout_192_layer_call_and_return_conditional_losses_826711
G__inference_dropout_192_layer_call_and_return_conditional_losses_826706?
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
*__inference_dense_485_layer_call_fn_826739?
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
E__inference_dense_485_layer_call_and_return_conditional_losses_826732?
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
;B9
$__inference_signature_wrapper_826559dense_483_input
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
E__inference_dense_483_layer_call_and_return_conditional_losses_826661\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? ?
.__inference_sequential_97_layer_call_fn_826542d !@?=
6?3
)?&
dense_483_input?????????
p 

 
? "??????????}
*__inference_dense_484_layer_call_fn_826686O/?,
%?"
 ?
inputs?????????@
? "??????????@
,__inference_dropout_192_layer_call_fn_826716O3?0
)?&
 ?
inputs?????????@
p
? "??????????@?
G__inference_dropout_192_layer_call_and_return_conditional_losses_826706\3?0
)?&
 ?
inputs?????????@
p
? "%?"
?
0?????????@
? ?
G__inference_dropout_192_layer_call_and_return_conditional_losses_826711\3?0
)?&
 ?
inputs?????????@
p 
? "%?"
?
0?????????@
? 
,__inference_dropout_192_layer_call_fn_826721O3?0
)?&
 ?
inputs?????????@
p 
? "??????????@?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826628h !7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826471q !@?=
6?3
)?&
dense_483_input?????????
p

 
? "%?"
?
0?????????
? ?
$__inference_signature_wrapper_826559? !K?H
? 
A?>
<
dense_483_input)?&
dense_483_input?????????"5?2
0
	dense_485#? 
	dense_485?????????}
*__inference_dense_483_layer_call_fn_826668O/?,
%?"
 ?
inputs?????????
? "??????????@?
E__inference_dense_485_layer_call_and_return_conditional_losses_826732\ !/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
E__inference_dense_484_layer_call_and_return_conditional_losses_826679\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826487q !@?=
6?3
)?&
dense_483_input?????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_97_layer_call_and_return_conditional_losses_826602h !7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_97_layer_call_fn_826514d !@?=
6?3
)?&
dense_483_input?????????
p

 
? "??????????}
*__inference_dense_485_layer_call_fn_826739O !/?,
%?"
 ?
inputs?????????@
? "???????????
.__inference_sequential_97_layer_call_fn_826639[ !7?4
-?*
 ?
inputs?????????
p

 
? "???????????
!__inference__wrapped_model_826336y !8?5
.?+
)?&
dense_483_input?????????
? "5?2
0
	dense_485#? 
	dense_485??????????
.__inference_sequential_97_layer_call_fn_826650[ !7?4
-?*
 ?
inputs?????????
p 

 
? "??????????