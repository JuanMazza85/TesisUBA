¬Ñ
«ý
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
dtypetype
¾
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8¥¹

dense_10219/kernelVarHandleOp*
shape:	*#
shared_namedense_10219/kernel*
dtype0*
_output_shapes
: 
z
&dense_10219/kernel/Read/ReadVariableOpReadVariableOpdense_10219/kernel*
dtype0*
_output_shapes
:	
y
dense_10219/biasVarHandleOp*
shape:*!
shared_namedense_10219/bias*
dtype0*
_output_shapes
: 
r
$dense_10219/bias/Read/ReadVariableOpReadVariableOpdense_10219/bias*
dtype0*
_output_shapes	
:

dense_10220/kernelVarHandleOp*
shape:
*#
shared_namedense_10220/kernel*
dtype0*
_output_shapes
: 
{
&dense_10220/kernel/Read/ReadVariableOpReadVariableOpdense_10220/kernel*
dtype0* 
_output_shapes
:

y
dense_10220/biasVarHandleOp*
shape:*!
shared_namedense_10220/bias*
dtype0*
_output_shapes
: 
r
$dense_10220/bias/Read/ReadVariableOpReadVariableOpdense_10220/bias*
dtype0*
_output_shapes	
:

dense_10221/kernelVarHandleOp*
shape:	*#
shared_namedense_10221/kernel*
dtype0*
_output_shapes
: 
z
&dense_10221/kernel/Read/ReadVariableOpReadVariableOpdense_10221/kernel*
dtype0*
_output_shapes
:	
x
dense_10221/biasVarHandleOp*
shape:*!
shared_namedense_10221/bias*
dtype0*
_output_shapes
: 
q
$dense_10221/bias/Read/ReadVariableOpReadVariableOpdense_10221/bias*
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

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
©
&Adadelta/dense_10219/kernel/accum_gradVarHandleOp*
shape:	*7
shared_name(&Adadelta/dense_10219/kernel/accum_grad*
dtype0*
_output_shapes
: 
¢
:Adadelta/dense_10219/kernel/accum_grad/Read/ReadVariableOpReadVariableOp&Adadelta/dense_10219/kernel/accum_grad*
dtype0*
_output_shapes
:	
¡
$Adadelta/dense_10219/bias/accum_gradVarHandleOp*
shape:*5
shared_name&$Adadelta/dense_10219/bias/accum_grad*
dtype0*
_output_shapes
: 

8Adadelta/dense_10219/bias/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/dense_10219/bias/accum_grad*
dtype0*
_output_shapes	
:
ª
&Adadelta/dense_10220/kernel/accum_gradVarHandleOp*
shape:
*7
shared_name(&Adadelta/dense_10220/kernel/accum_grad*
dtype0*
_output_shapes
: 
£
:Adadelta/dense_10220/kernel/accum_grad/Read/ReadVariableOpReadVariableOp&Adadelta/dense_10220/kernel/accum_grad*
dtype0* 
_output_shapes
:

¡
$Adadelta/dense_10220/bias/accum_gradVarHandleOp*
shape:*5
shared_name&$Adadelta/dense_10220/bias/accum_grad*
dtype0*
_output_shapes
: 

8Adadelta/dense_10220/bias/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/dense_10220/bias/accum_grad*
dtype0*
_output_shapes	
:
©
&Adadelta/dense_10221/kernel/accum_gradVarHandleOp*
shape:	*7
shared_name(&Adadelta/dense_10221/kernel/accum_grad*
dtype0*
_output_shapes
: 
¢
:Adadelta/dense_10221/kernel/accum_grad/Read/ReadVariableOpReadVariableOp&Adadelta/dense_10221/kernel/accum_grad*
dtype0*
_output_shapes
:	
 
$Adadelta/dense_10221/bias/accum_gradVarHandleOp*
shape:*5
shared_name&$Adadelta/dense_10221/bias/accum_grad*
dtype0*
_output_shapes
: 

8Adadelta/dense_10221/bias/accum_grad/Read/ReadVariableOpReadVariableOp$Adadelta/dense_10221/bias/accum_grad*
dtype0*
_output_shapes
:
§
%Adadelta/dense_10219/kernel/accum_varVarHandleOp*
shape:	*6
shared_name'%Adadelta/dense_10219/kernel/accum_var*
dtype0*
_output_shapes
: 
 
9Adadelta/dense_10219/kernel/accum_var/Read/ReadVariableOpReadVariableOp%Adadelta/dense_10219/kernel/accum_var*
dtype0*
_output_shapes
:	

#Adadelta/dense_10219/bias/accum_varVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_10219/bias/accum_var*
dtype0*
_output_shapes
: 

7Adadelta/dense_10219/bias/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/dense_10219/bias/accum_var*
dtype0*
_output_shapes	
:
¨
%Adadelta/dense_10220/kernel/accum_varVarHandleOp*
shape:
*6
shared_name'%Adadelta/dense_10220/kernel/accum_var*
dtype0*
_output_shapes
: 
¡
9Adadelta/dense_10220/kernel/accum_var/Read/ReadVariableOpReadVariableOp%Adadelta/dense_10220/kernel/accum_var*
dtype0* 
_output_shapes
:


#Adadelta/dense_10220/bias/accum_varVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_10220/bias/accum_var*
dtype0*
_output_shapes
: 

7Adadelta/dense_10220/bias/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/dense_10220/bias/accum_var*
dtype0*
_output_shapes	
:
§
%Adadelta/dense_10221/kernel/accum_varVarHandleOp*
shape:	*6
shared_name'%Adadelta/dense_10221/kernel/accum_var*
dtype0*
_output_shapes
: 
 
9Adadelta/dense_10221/kernel/accum_var/Read/ReadVariableOpReadVariableOp%Adadelta/dense_10221/kernel/accum_var*
dtype0*
_output_shapes
:	

#Adadelta/dense_10221/bias/accum_varVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_10221/bias/accum_var*
dtype0*
_output_shapes
: 

7Adadelta/dense_10221/bias/accum_var/Read/ReadVariableOpReadVariableOp#Adadelta/dense_10221/bias/accum_var*
dtype0*
_output_shapes
:

NoOpNoOp
)
ConstConst"/device:CPU:0*Â(
value¸(Bµ( B®(
ó
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api

!iter
	"decay
#learning_rate
$rho
accum_gradE
accum_gradF
accum_gradG
accum_gradH
accum_gradI
accum_gradJ	accum_varK	accum_varL	accum_varM	accum_varN	accum_varO	accum_varP
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5

regularization_losses

%layers
	variables
&metrics
'layer_regularization_losses
trainable_variables
(non_trainable_variables
 
 
 
 

regularization_losses

)layers
	variables
*metrics
+layer_regularization_losses
trainable_variables
,non_trainable_variables
^\
VARIABLE_VALUEdense_10219/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_10219/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses

-layers
	variables
.metrics
/layer_regularization_losses
trainable_variables
0non_trainable_variables
^\
VARIABLE_VALUEdense_10220/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_10220/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses

1layers
	variables
2metrics
3layer_regularization_losses
trainable_variables
4non_trainable_variables
^\
VARIABLE_VALUEdense_10221/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_10221/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses

5layers
	variables
6metrics
7layer_regularization_losses
trainable_variables
8non_trainable_variables
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

0
1
2

90
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
	:total
	;count
<
_fn_kwargs
=regularization_losses
>	variables
?trainable_variables
@	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

:0
;1
 

=regularization_losses

Alayers
>	variables
Bmetrics
Clayer_regularization_losses
?trainable_variables
Dnon_trainable_variables
 
 
 

:0
;1

VARIABLE_VALUE&Adadelta/dense_10219/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_10219/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adadelta/dense_10220/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_10220/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adadelta/dense_10221/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_10221/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adadelta/dense_10219/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_10219/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adadelta/dense_10220/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_10220/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adadelta/dense_10221/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_10221/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

!serving_default_dense_10219_inputPlaceholder*
shape:ÿÿÿÿÿÿÿÿÿ*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¡
StatefulPartitionedCallStatefulPartitionedCall!serving_default_dense_10219_inputdense_10219/kerneldense_10219/biasdense_10220/kerneldense_10220/biasdense_10221/kerneldense_10221/bias*/
_gradient_op_typePartitionedCall-17375450*/
f*R(
&__inference_signature_wrapper_17375276*
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
:ÿÿÿÿÿÿÿÿÿ
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&dense_10219/kernel/Read/ReadVariableOp$dense_10219/bias/Read/ReadVariableOp&dense_10220/kernel/Read/ReadVariableOp$dense_10220/bias/Read/ReadVariableOp&dense_10221/kernel/Read/ReadVariableOp$dense_10221/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp:Adadelta/dense_10219/kernel/accum_grad/Read/ReadVariableOp8Adadelta/dense_10219/bias/accum_grad/Read/ReadVariableOp:Adadelta/dense_10220/kernel/accum_grad/Read/ReadVariableOp8Adadelta/dense_10220/bias/accum_grad/Read/ReadVariableOp:Adadelta/dense_10221/kernel/accum_grad/Read/ReadVariableOp8Adadelta/dense_10221/bias/accum_grad/Read/ReadVariableOp9Adadelta/dense_10219/kernel/accum_var/Read/ReadVariableOp7Adadelta/dense_10219/bias/accum_var/Read/ReadVariableOp9Adadelta/dense_10220/kernel/accum_var/Read/ReadVariableOp7Adadelta/dense_10220/bias/accum_var/Read/ReadVariableOp9Adadelta/dense_10221/kernel/accum_var/Read/ReadVariableOp7Adadelta/dense_10221/bias/accum_var/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-17375496**
f%R#
!__inference__traced_save_17375495*
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
°
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10219/kerneldense_10219/biasdense_10220/kerneldense_10220/biasdense_10221/kerneldense_10221/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcount&Adadelta/dense_10219/kernel/accum_grad$Adadelta/dense_10219/bias/accum_grad&Adadelta/dense_10220/kernel/accum_grad$Adadelta/dense_10220/bias/accum_grad&Adadelta/dense_10221/kernel/accum_grad$Adadelta/dense_10221/bias/accum_grad%Adadelta/dense_10219/kernel/accum_var#Adadelta/dense_10219/bias/accum_var%Adadelta/dense_10220/kernel/accum_var#Adadelta/dense_10220/bias/accum_var%Adadelta/dense_10221/kernel/accum_var#Adadelta/dense_10221/bias/accum_var*/
_gradient_op_typePartitionedCall-17375581*-
f(R&
$__inference__traced_restore_17375580*
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
:  Ä
¨	
Ã
2__inference_sequential_2057_layer_call_fn_17375335

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-17375224*V
fQRO
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375223*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
ý
ê
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375301

inputs.
*dense_10219_matmul_readvariableop_resource/
+dense_10219_biasadd_readvariableop_resource.
*dense_10220_matmul_readvariableop_resource/
+dense_10220_biasadd_readvariableop_resource.
*dense_10221_matmul_readvariableop_resource/
+dense_10221_biasadd_readvariableop_resource
identity¢"dense_10219/BiasAdd/ReadVariableOp¢!dense_10219/MatMul/ReadVariableOp¢"dense_10220/BiasAdd/ReadVariableOp¢!dense_10220/MatMul/ReadVariableOp¢"dense_10221/BiasAdd/ReadVariableOp¢!dense_10221/MatMul/ReadVariableOp»
!dense_10219/MatMul/ReadVariableOpReadVariableOp*dense_10219_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_10219/MatMulMatMulinputs)dense_10219/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_10219/BiasAdd/ReadVariableOpReadVariableOp+dense_10219_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_10219/BiasAddBiasAdddense_10219/MatMul:product:0*dense_10219/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_10220/MatMul/ReadVariableOpReadVariableOp*dense_10220_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_10220/MatMulMatMuldense_10219/BiasAdd:output:0)dense_10220/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_10220/BiasAdd/ReadVariableOpReadVariableOp+dense_10220_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_10220/BiasAddBiasAdddense_10220/MatMul:product:0*dense_10220/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
!dense_10221/MatMul/ReadVariableOpReadVariableOp*dense_10221_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_10221/MatMulMatMuldense_10220/BiasAdd:output:0)dense_10221/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
"dense_10221/BiasAdd/ReadVariableOpReadVariableOp+dense_10221_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_10221/BiasAddBiasAdddense_10221/MatMul:product:0*dense_10221/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_10221/ReluReludense_10221/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
IdentityIdentitydense_10221/Relu:activations:0#^dense_10219/BiasAdd/ReadVariableOp"^dense_10219/MatMul/ReadVariableOp#^dense_10220/BiasAdd/ReadVariableOp"^dense_10220/MatMul/ReadVariableOp#^dense_10221/BiasAdd/ReadVariableOp"^dense_10221/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_10221/BiasAdd/ReadVariableOp"dense_10221/BiasAdd/ReadVariableOp2H
"dense_10220/BiasAdd/ReadVariableOp"dense_10220/BiasAdd/ReadVariableOp2F
!dense_10220/MatMul/ReadVariableOp!dense_10220/MatMul/ReadVariableOp2H
"dense_10219/BiasAdd/ReadVariableOp"dense_10219/BiasAdd/ReadVariableOp2F
!dense_10219/MatMul/ReadVariableOp!dense_10219/MatMul/ReadVariableOp2F
!dense_10221/MatMul/ReadVariableOp!dense_10221/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
æ
¯
.__inference_dense_10219_layer_call_fn_17375363

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375125*R
fMRK
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375119*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
	
Â
&__inference_signature_wrapper_17375276
dense_10219_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCalldense_10219_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-17375267*,
f'R%
#__inference__wrapped_model_17375103*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :1 -
+
_user_specified_namedense_10219_input: : 
¶

M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375207
dense_10219_input.
*dense_10219_statefulpartitionedcall_args_1.
*dense_10219_statefulpartitionedcall_args_2.
*dense_10220_statefulpartitionedcall_args_1.
*dense_10220_statefulpartitionedcall_args_2.
*dense_10221_statefulpartitionedcall_args_1.
*dense_10221_statefulpartitionedcall_args_2
identity¢#dense_10219/StatefulPartitionedCall¢#dense_10220/StatefulPartitionedCall¢#dense_10221/StatefulPartitionedCall£
#dense_10219/StatefulPartitionedCallStatefulPartitionedCalldense_10219_input*dense_10219_statefulpartitionedcall_args_1*dense_10219_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375125*R
fMRK
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375119*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
#dense_10220/StatefulPartitionedCallStatefulPartitionedCall,dense_10219/StatefulPartitionedCall:output:0*dense_10220_statefulpartitionedcall_args_1*dense_10220_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375152*R
fMRK
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375146*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
#dense_10221/StatefulPartitionedCallStatefulPartitionedCall,dense_10220/StatefulPartitionedCall:output:0*dense_10221_statefulpartitionedcall_args_1*dense_10221_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375180*R
fMRK
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375174*
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
:ÿÿÿÿÿÿÿÿÿæ
IdentityIdentity,dense_10221/StatefulPartitionedCall:output:0$^dense_10219/StatefulPartitionedCall$^dense_10220/StatefulPartitionedCall$^dense_10221/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2J
#dense_10220/StatefulPartitionedCall#dense_10220/StatefulPartitionedCall2J
#dense_10221/StatefulPartitionedCall#dense_10221/StatefulPartitionedCall2J
#dense_10219/StatefulPartitionedCall#dense_10219/StatefulPartitionedCall: : : : :1 -
+
_user_specified_namedense_10219_input: : 
	
â
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375356

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
É	
Î
2__inference_sequential_2057_layer_call_fn_17375260
dense_10219_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_10219_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-17375251*V
fQRO
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375250*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :1 -
+
_user_specified_namedense_10219_input: : 

þ
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375250

inputs.
*dense_10219_statefulpartitionedcall_args_1.
*dense_10219_statefulpartitionedcall_args_2.
*dense_10220_statefulpartitionedcall_args_1.
*dense_10220_statefulpartitionedcall_args_2.
*dense_10221_statefulpartitionedcall_args_1.
*dense_10221_statefulpartitionedcall_args_2
identity¢#dense_10219/StatefulPartitionedCall¢#dense_10220/StatefulPartitionedCall¢#dense_10221/StatefulPartitionedCall
#dense_10219/StatefulPartitionedCallStatefulPartitionedCallinputs*dense_10219_statefulpartitionedcall_args_1*dense_10219_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375125*R
fMRK
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375119*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
#dense_10220/StatefulPartitionedCallStatefulPartitionedCall,dense_10219/StatefulPartitionedCall:output:0*dense_10220_statefulpartitionedcall_args_1*dense_10220_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375152*R
fMRK
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375146*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
#dense_10221/StatefulPartitionedCallStatefulPartitionedCall,dense_10220/StatefulPartitionedCall:output:0*dense_10221_statefulpartitionedcall_args_1*dense_10221_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375180*R
fMRK
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375174*
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
:ÿÿÿÿÿÿÿÿÿæ
IdentityIdentity,dense_10221/StatefulPartitionedCall:output:0$^dense_10219/StatefulPartitionedCall$^dense_10220/StatefulPartitionedCall$^dense_10221/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2J
#dense_10220/StatefulPartitionedCall#dense_10220/StatefulPartitionedCall2J
#dense_10221/StatefulPartitionedCall#dense_10221/StatefulPartitionedCall2J
#dense_10219/StatefulPartitionedCall#dense_10219/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
ì9
ÿ
!__inference__traced_save_17375495
file_prefix1
-savev2_dense_10219_kernel_read_readvariableop/
+savev2_dense_10219_bias_read_readvariableop1
-savev2_dense_10220_kernel_read_readvariableop/
+savev2_dense_10220_bias_read_readvariableop1
-savev2_dense_10221_kernel_read_readvariableop/
+savev2_dense_10221_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopE
Asavev2_adadelta_dense_10219_kernel_accum_grad_read_readvariableopC
?savev2_adadelta_dense_10219_bias_accum_grad_read_readvariableopE
Asavev2_adadelta_dense_10220_kernel_accum_grad_read_readvariableopC
?savev2_adadelta_dense_10220_bias_accum_grad_read_readvariableopE
Asavev2_adadelta_dense_10221_kernel_accum_grad_read_readvariableopC
?savev2_adadelta_dense_10221_bias_accum_grad_read_readvariableopD
@savev2_adadelta_dense_10219_kernel_accum_var_read_readvariableopB
>savev2_adadelta_dense_10219_bias_accum_var_read_readvariableopD
@savev2_adadelta_dense_10220_kernel_accum_var_read_readvariableopB
>savev2_adadelta_dense_10220_bias_accum_var_read_readvariableopD
@savev2_adadelta_dense_10221_kernel_accum_var_read_readvariableopB
>savev2_adadelta_dense_10221_bias_accum_var_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_7bb9337ab74042beb9c2e42602dbb495/part*
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
: 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¡
SaveV2/tensor_namesConst"/device:CPU:0*Ê
valueÀB½B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
SaveV2/shape_and_slicesConst"/device:CPU:0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Ô
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_dense_10219_kernel_read_readvariableop+savev2_dense_10219_bias_read_readvariableop-savev2_dense_10220_kernel_read_readvariableop+savev2_dense_10220_bias_read_readvariableop-savev2_dense_10221_kernel_read_readvariableop+savev2_dense_10221_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopAsavev2_adadelta_dense_10219_kernel_accum_grad_read_readvariableop?savev2_adadelta_dense_10219_bias_accum_grad_read_readvariableopAsavev2_adadelta_dense_10220_kernel_accum_grad_read_readvariableop?savev2_adadelta_dense_10220_bias_accum_grad_read_readvariableopAsavev2_adadelta_dense_10221_kernel_accum_grad_read_readvariableop?savev2_adadelta_dense_10221_bias_accum_grad_read_readvariableop@savev2_adadelta_dense_10219_kernel_accum_var_read_readvariableop>savev2_adadelta_dense_10219_bias_accum_var_read_readvariableop@savev2_adadelta_dense_10220_kernel_accum_var_read_readvariableop>savev2_adadelta_dense_10220_bias_accum_var_read_readvariableop@savev2_adadelta_dense_10221_kernel_accum_var_read_readvariableop>savev2_adadelta_dense_10221_bias_accum_var_read_readvariableop"/device:CPU:0*&
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
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
:Ã
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ¹
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
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

identity_1Identity_1:output:0*Ç
_input_shapesµ
²: :	::
::	:: : : : : : :	::
::	::	::
::	:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : :
 
	
â
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375373

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¤
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
¶

M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375192
dense_10219_input.
*dense_10219_statefulpartitionedcall_args_1.
*dense_10219_statefulpartitionedcall_args_2.
*dense_10220_statefulpartitionedcall_args_1.
*dense_10220_statefulpartitionedcall_args_2.
*dense_10221_statefulpartitionedcall_args_1.
*dense_10221_statefulpartitionedcall_args_2
identity¢#dense_10219/StatefulPartitionedCall¢#dense_10220/StatefulPartitionedCall¢#dense_10221/StatefulPartitionedCall£
#dense_10219/StatefulPartitionedCallStatefulPartitionedCalldense_10219_input*dense_10219_statefulpartitionedcall_args_1*dense_10219_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375125*R
fMRK
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375119*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
#dense_10220/StatefulPartitionedCallStatefulPartitionedCall,dense_10219/StatefulPartitionedCall:output:0*dense_10220_statefulpartitionedcall_args_1*dense_10220_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375152*R
fMRK
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375146*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
#dense_10221/StatefulPartitionedCallStatefulPartitionedCall,dense_10220/StatefulPartitionedCall:output:0*dense_10221_statefulpartitionedcall_args_1*dense_10221_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375180*R
fMRK
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375174*
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
:ÿÿÿÿÿÿÿÿÿæ
IdentityIdentity,dense_10221/StatefulPartitionedCall:output:0$^dense_10219/StatefulPartitionedCall$^dense_10220/StatefulPartitionedCall$^dense_10221/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2J
#dense_10220/StatefulPartitionedCall#dense_10220/StatefulPartitionedCall2J
#dense_10221/StatefulPartitionedCall#dense_10221/StatefulPartitionedCall2J
#dense_10219/StatefulPartitionedCall#dense_10219/StatefulPartitionedCall: : : : :1 -
+
_user_specified_namedense_10219_input: : 
ç
¯
.__inference_dense_10220_layer_call_fn_17375380

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375152*R
fMRK
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375146*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
×	
â
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375174

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
å
¯
.__inference_dense_10221_layer_call_fn_17375398

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375180*R
fMRK
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375174*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 

þ
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375223

inputs.
*dense_10219_statefulpartitionedcall_args_1.
*dense_10219_statefulpartitionedcall_args_2.
*dense_10220_statefulpartitionedcall_args_1.
*dense_10220_statefulpartitionedcall_args_2.
*dense_10221_statefulpartitionedcall_args_1.
*dense_10221_statefulpartitionedcall_args_2
identity¢#dense_10219/StatefulPartitionedCall¢#dense_10220/StatefulPartitionedCall¢#dense_10221/StatefulPartitionedCall
#dense_10219/StatefulPartitionedCallStatefulPartitionedCallinputs*dense_10219_statefulpartitionedcall_args_1*dense_10219_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375125*R
fMRK
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375119*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
#dense_10220/StatefulPartitionedCallStatefulPartitionedCall,dense_10219/StatefulPartitionedCall:output:0*dense_10220_statefulpartitionedcall_args_1*dense_10220_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375152*R
fMRK
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375146*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
#dense_10221/StatefulPartitionedCallStatefulPartitionedCall,dense_10220/StatefulPartitionedCall:output:0*dense_10221_statefulpartitionedcall_args_1*dense_10221_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-17375180*R
fMRK
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375174*
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
:ÿÿÿÿÿÿÿÿÿæ
IdentityIdentity,dense_10221/StatefulPartitionedCall:output:0$^dense_10219/StatefulPartitionedCall$^dense_10220/StatefulPartitionedCall$^dense_10221/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2J
#dense_10220/StatefulPartitionedCall#dense_10220/StatefulPartitionedCall2J
#dense_10221/StatefulPartitionedCall#dense_10221/StatefulPartitionedCall2J
#dense_10219/StatefulPartitionedCall#dense_10219/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
×	
â
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375391

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
	
â
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375119

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
¨	
Ã
2__inference_sequential_2057_layer_call_fn_17375346

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-17375251*V
fQRO
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375250*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
	
â
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375146

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¤
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
c
°
$__inference__traced_restore_17375580
file_prefix'
#assignvariableop_dense_10219_kernel'
#assignvariableop_1_dense_10219_bias)
%assignvariableop_2_dense_10220_kernel'
#assignvariableop_3_dense_10220_bias)
%assignvariableop_4_dense_10221_kernel'
#assignvariableop_5_dense_10221_bias$
 assignvariableop_6_adadelta_iter%
!assignvariableop_7_adadelta_decay-
)assignvariableop_8_adadelta_learning_rate#
assignvariableop_9_adadelta_rho
assignvariableop_10_total
assignvariableop_11_count>
:assignvariableop_12_adadelta_dense_10219_kernel_accum_grad<
8assignvariableop_13_adadelta_dense_10219_bias_accum_grad>
:assignvariableop_14_adadelta_dense_10220_kernel_accum_grad<
8assignvariableop_15_adadelta_dense_10220_bias_accum_grad>
:assignvariableop_16_adadelta_dense_10221_kernel_accum_grad<
8assignvariableop_17_adadelta_dense_10221_bias_accum_grad=
9assignvariableop_18_adadelta_dense_10219_kernel_accum_var;
7assignvariableop_19_adadelta_dense_10219_bias_accum_var=
9assignvariableop_20_adadelta_dense_10220_kernel_accum_var;
7assignvariableop_21_adadelta_dense_10220_bias_accum_var=
9assignvariableop_22_adadelta_dense_10221_kernel_accum_var;
7assignvariableop_23_adadelta_dense_10221_bias_accum_var
identity_25¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1¤
RestoreV2/tensor_namesConst"/device:CPU:0*Ê
valueÀB½B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
RestoreV2/shape_and_slicesConst"/device:CPU:0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*&
dtypes
2	*t
_output_shapesb
`::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_dense_10219_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_10219_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_10220_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_10220_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp%assignvariableop_4_dense_10221_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_10221_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOp assignvariableop_6_adadelta_iterIdentity_6:output:0*
dtype0	*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp!assignvariableop_7_adadelta_decayIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
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
:
AssignVariableOp_12AssignVariableOp:assignvariableop_12_adadelta_dense_10219_kernel_accum_gradIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp8assignvariableop_13_adadelta_dense_10219_bias_accum_gradIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp:assignvariableop_14_adadelta_dense_10220_kernel_accum_gradIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp8assignvariableop_15_adadelta_dense_10220_bias_accum_gradIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp:assignvariableop_16_adadelta_dense_10221_kernel_accum_gradIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp8assignvariableop_17_adadelta_dense_10221_bias_accum_gradIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp9assignvariableop_18_adadelta_dense_10219_kernel_accum_varIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adadelta_dense_10219_bias_accum_varIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp9assignvariableop_20_adadelta_dense_10220_kernel_accum_varIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adadelta_dense_10220_bias_accum_varIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp9assignvariableop_22_adadelta_dense_10221_kernel_accum_varIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adadelta_dense_10221_bias_accum_varIdentity_23:output:0*
dtype0*
_output_shapes
 
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
:µ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ß
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ì
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2(
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
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192$
AssignVariableOpAssignVariableOp: : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : :
 
É	
Î
2__inference_sequential_2057_layer_call_fn_17375233
dense_10219_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_10219_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-17375224*V
fQRO
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375223*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :1 -
+
_user_specified_namedense_10219_input: : 
Õ$

#__inference__wrapped_model_17375103
dense_10219_input>
:sequential_2057_dense_10219_matmul_readvariableop_resource?
;sequential_2057_dense_10219_biasadd_readvariableop_resource>
:sequential_2057_dense_10220_matmul_readvariableop_resource?
;sequential_2057_dense_10220_biasadd_readvariableop_resource>
:sequential_2057_dense_10221_matmul_readvariableop_resource?
;sequential_2057_dense_10221_biasadd_readvariableop_resource
identity¢2sequential_2057/dense_10219/BiasAdd/ReadVariableOp¢1sequential_2057/dense_10219/MatMul/ReadVariableOp¢2sequential_2057/dense_10220/BiasAdd/ReadVariableOp¢1sequential_2057/dense_10220/MatMul/ReadVariableOp¢2sequential_2057/dense_10221/BiasAdd/ReadVariableOp¢1sequential_2057/dense_10221/MatMul/ReadVariableOpÛ
1sequential_2057/dense_10219/MatMul/ReadVariableOpReadVariableOp:sequential_2057_dense_10219_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	­
"sequential_2057/dense_10219/MatMulMatMuldense_10219_input9sequential_2057/dense_10219/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2057/dense_10219/BiasAdd/ReadVariableOpReadVariableOp;sequential_2057_dense_10219_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2057/dense_10219/BiasAddBiasAdd,sequential_2057/dense_10219/MatMul:product:0:sequential_2057/dense_10219/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
1sequential_2057/dense_10220/MatMul/ReadVariableOpReadVariableOp:sequential_2057_dense_10220_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
È
"sequential_2057/dense_10220/MatMulMatMul,sequential_2057/dense_10219/BiasAdd:output:09sequential_2057/dense_10220/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2057/dense_10220/BiasAdd/ReadVariableOpReadVariableOp;sequential_2057_dense_10220_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2057/dense_10220/BiasAddBiasAdd,sequential_2057/dense_10220/MatMul:product:0:sequential_2057/dense_10220/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
1sequential_2057/dense_10221/MatMul/ReadVariableOpReadVariableOp:sequential_2057_dense_10221_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ç
"sequential_2057/dense_10221/MatMulMatMul,sequential_2057/dense_10220/BiasAdd:output:09sequential_2057/dense_10221/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
2sequential_2057/dense_10221/BiasAdd/ReadVariableOpReadVariableOp;sequential_2057_dense_10221_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ê
#sequential_2057/dense_10221/BiasAddBiasAdd,sequential_2057/dense_10221/MatMul:product:0:sequential_2057/dense_10221/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 sequential_2057/dense_10221/ReluRelu,sequential_2057/dense_10221/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
IdentityIdentity.sequential_2057/dense_10221/Relu:activations:03^sequential_2057/dense_10219/BiasAdd/ReadVariableOp2^sequential_2057/dense_10219/MatMul/ReadVariableOp3^sequential_2057/dense_10220/BiasAdd/ReadVariableOp2^sequential_2057/dense_10220/MatMul/ReadVariableOp3^sequential_2057/dense_10221/BiasAdd/ReadVariableOp2^sequential_2057/dense_10221/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2h
2sequential_2057/dense_10219/BiasAdd/ReadVariableOp2sequential_2057/dense_10219/BiasAdd/ReadVariableOp2f
1sequential_2057/dense_10220/MatMul/ReadVariableOp1sequential_2057/dense_10220/MatMul/ReadVariableOp2f
1sequential_2057/dense_10219/MatMul/ReadVariableOp1sequential_2057/dense_10219/MatMul/ReadVariableOp2f
1sequential_2057/dense_10221/MatMul/ReadVariableOp1sequential_2057/dense_10221/MatMul/ReadVariableOp2h
2sequential_2057/dense_10221/BiasAdd/ReadVariableOp2sequential_2057/dense_10221/BiasAdd/ReadVariableOp2h
2sequential_2057/dense_10220/BiasAdd/ReadVariableOp2sequential_2057/dense_10220/BiasAdd/ReadVariableOp: : : : :1 -
+
_user_specified_namedense_10219_input: : 
ý
ê
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375324

inputs.
*dense_10219_matmul_readvariableop_resource/
+dense_10219_biasadd_readvariableop_resource.
*dense_10220_matmul_readvariableop_resource/
+dense_10220_biasadd_readvariableop_resource.
*dense_10221_matmul_readvariableop_resource/
+dense_10221_biasadd_readvariableop_resource
identity¢"dense_10219/BiasAdd/ReadVariableOp¢!dense_10219/MatMul/ReadVariableOp¢"dense_10220/BiasAdd/ReadVariableOp¢!dense_10220/MatMul/ReadVariableOp¢"dense_10221/BiasAdd/ReadVariableOp¢!dense_10221/MatMul/ReadVariableOp»
!dense_10219/MatMul/ReadVariableOpReadVariableOp*dense_10219_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_10219/MatMulMatMulinputs)dense_10219/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_10219/BiasAdd/ReadVariableOpReadVariableOp+dense_10219_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_10219/BiasAddBiasAdddense_10219/MatMul:product:0*dense_10219/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_10220/MatMul/ReadVariableOpReadVariableOp*dense_10220_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_10220/MatMulMatMuldense_10219/BiasAdd:output:0)dense_10220/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_10220/BiasAdd/ReadVariableOpReadVariableOp+dense_10220_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_10220/BiasAddBiasAdddense_10220/MatMul:product:0*dense_10220/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
!dense_10221/MatMul/ReadVariableOpReadVariableOp*dense_10221_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_10221/MatMulMatMuldense_10220/BiasAdd:output:0)dense_10221/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
"dense_10221/BiasAdd/ReadVariableOpReadVariableOp+dense_10221_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_10221/BiasAddBiasAdddense_10221/MatMul:product:0*dense_10221/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_10221/ReluReludense_10221/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
IdentityIdentitydense_10221/Relu:activations:0#^dense_10219/BiasAdd/ReadVariableOp"^dense_10219/MatMul/ReadVariableOp#^dense_10220/BiasAdd/ReadVariableOp"^dense_10220/MatMul/ReadVariableOp#^dense_10221/BiasAdd/ReadVariableOp"^dense_10221/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_10221/BiasAdd/ReadVariableOp"dense_10221/BiasAdd/ReadVariableOp2H
"dense_10220/BiasAdd/ReadVariableOp"dense_10220/BiasAdd/ReadVariableOp2F
!dense_10220/MatMul/ReadVariableOp!dense_10220/MatMul/ReadVariableOp2F
!dense_10219/MatMul/ReadVariableOp!dense_10219/MatMul/ReadVariableOp2H
"dense_10219/BiasAdd/ReadVariableOp"dense_10219/BiasAdd/ReadVariableOp2F
!dense_10221/MatMul/ReadVariableOp!dense_10221/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*Â
serving_default®
O
dense_10219_input:
#serving_default_dense_10219_input:0ÿÿÿÿÿÿÿÿÿ?
dense_102210
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ý

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
regularization_losses
	variables
trainable_variables
		keras_api


signatures
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses"Ê
_tf_keras_sequential«{"class_name": "Sequential", "name": "sequential_2057", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2057", "layers": [{"class_name": "Dense", "config": {"name": "dense_10219", "trainable": true, "batch_input_shape": [null, 23], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10220", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10221", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2057", "layers": [{"class_name": "Dense", "config": {"name": "dense_10219", "trainable": true, "batch_input_shape": [null, 23], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10220", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_10221", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
·
regularization_losses
	variables
trainable_variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "InputLayer", "name": "dense_10219_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 23], "config": {"batch_input_shape": [null, 23], "dtype": "float32", "sparse": false, "name": "dense_10219_input"}}
¿

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Dense", "name": "dense_10219", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 23], "config": {"name": "dense_10219", "trainable": true, "batch_input_shape": [null, 23], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 23}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "Dense", "name": "dense_10220", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_10220", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "Dense", "name": "dense_10221", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_10221", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}

!iter
	"decay
#learning_rate
$rho
accum_gradE
accum_gradF
accum_gradG
accum_gradH
accum_gradI
accum_gradJ	accum_varK	accum_varL	accum_varM	accum_varN	accum_varO	accum_varP"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
·
regularization_losses

%layers
	variables
&metrics
'layer_regularization_losses
trainable_variables
(non_trainable_variables
Q__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
\serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

regularization_losses

)layers
	variables
*metrics
+layer_regularization_losses
trainable_variables
,non_trainable_variables
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
%:#	2dense_10219/kernel
:2dense_10219/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

regularization_losses

-layers
	variables
.metrics
/layer_regularization_losses
trainable_variables
0non_trainable_variables
V__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
&:$
2dense_10220/kernel
:2dense_10220/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

regularization_losses

1layers
	variables
2metrics
3layer_regularization_losses
trainable_variables
4non_trainable_variables
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
%:#	2dense_10221/kernel
:2dense_10221/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

regularization_losses

5layers
	variables
6metrics
7layer_regularization_losses
trainable_variables
8non_trainable_variables
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
5
0
1
2"
trackable_list_wrapper
'
90"
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

	:total
	;count
<
_fn_kwargs
=regularization_losses
>	variables
?trainable_variables
@	keras_api
]__call__
*^&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper

=regularization_losses

Alayers
>	variables
Bmetrics
Clayer_regularization_losses
?trainable_variables
Dnon_trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
7:5	2&Adadelta/dense_10219/kernel/accum_grad
1:/2$Adadelta/dense_10219/bias/accum_grad
8:6
2&Adadelta/dense_10220/kernel/accum_grad
1:/2$Adadelta/dense_10220/bias/accum_grad
7:5	2&Adadelta/dense_10221/kernel/accum_grad
0:.2$Adadelta/dense_10221/bias/accum_grad
6:4	2%Adadelta/dense_10219/kernel/accum_var
0:.2#Adadelta/dense_10219/bias/accum_var
7:5
2%Adadelta/dense_10220/kernel/accum_var
0:.2#Adadelta/dense_10220/bias/accum_var
6:4	2%Adadelta/dense_10221/kernel/accum_var
/:-2#Adadelta/dense_10221/bias/accum_var
2
2__inference_sequential_2057_layer_call_fn_17375346
2__inference_sequential_2057_layer_call_fn_17375233
2__inference_sequential_2057_layer_call_fn_17375335
2__inference_sequential_2057_layer_call_fn_17375260À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ë2è
#__inference__wrapped_model_17375103À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
dense_10219_inputÿÿÿÿÿÿÿÿÿ
2ÿ
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375207
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375324
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375192
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375301À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ø2Õ
.__inference_dense_10219_layer_call_fn_17375363¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375356¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_dense_10220_layer_call_fn_17375380¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375373¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_dense_10221_layer_call_fn_17375398¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375391¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
?B=
&__inference_signature_wrapper_17375276dense_10219_input
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 ¿
&__inference_signature_wrapper_17375276O¢L
¢ 
EªB
@
dense_10219_input+(
dense_10219_inputÿÿÿÿÿÿÿÿÿ"9ª6
4
dense_10221%"
dense_10221ÿÿÿÿÿÿÿÿÿ
2__inference_sequential_2057_layer_call_fn_17375233fB¢?
8¢5
+(
dense_10219_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_sequential_2057_layer_call_fn_17375346[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¹
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375324h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375207sB¢?
8¢5
+(
dense_10219_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_10221_layer_call_fn_17375398P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
I__inference_dense_10219_layer_call_and_return_conditional_losses_17375356]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dense_10220_layer_call_and_return_conditional_losses_17375373^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ª
I__inference_dense_10221_layer_call_and_return_conditional_losses_17375391]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_10220_layer_call_fn_17375380Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_sequential_2057_layer_call_fn_17375260fB¢?
8¢5
+(
dense_10219_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÄ
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375192sB¢?
8¢5
+(
dense_10219_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¹
M__inference_sequential_2057_layer_call_and_return_conditional_losses_17375301h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_10219_layer_call_fn_17375363P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
#__inference__wrapped_model_17375103:¢7
0¢-
+(
dense_10219_inputÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
dense_10221%"
dense_10221ÿÿÿÿÿÿÿÿÿ
2__inference_sequential_2057_layer_call_fn_17375335[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ