½
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
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8ì

dense_9976/kernelVarHandleOp*
shape:	*"
shared_namedense_9976/kernel*
dtype0*
_output_shapes
: 
x
%dense_9976/kernel/Read/ReadVariableOpReadVariableOpdense_9976/kernel*
dtype0*
_output_shapes
:	
w
dense_9976/biasVarHandleOp*
shape:* 
shared_namedense_9976/bias*
dtype0*
_output_shapes
: 
p
#dense_9976/bias/Read/ReadVariableOpReadVariableOpdense_9976/bias*
dtype0*
_output_shapes	
:

dense_9977/kernelVarHandleOp*
shape:
*"
shared_namedense_9977/kernel*
dtype0*
_output_shapes
: 
y
%dense_9977/kernel/Read/ReadVariableOpReadVariableOpdense_9977/kernel*
dtype0* 
_output_shapes
:

w
dense_9977/biasVarHandleOp*
shape:* 
shared_namedense_9977/bias*
dtype0*
_output_shapes
: 
p
#dense_9977/bias/Read/ReadVariableOpReadVariableOpdense_9977/bias*
dtype0*
_output_shapes	
:

dense_9978/kernelVarHandleOp*
shape:	*"
shared_namedense_9978/kernel*
dtype0*
_output_shapes
: 
x
%dense_9978/kernel/Read/ReadVariableOpReadVariableOpdense_9978/kernel*
dtype0*
_output_shapes
:	
v
dense_9978/biasVarHandleOp*
shape:* 
shared_namedense_9978/bias*
dtype0*
_output_shapes
: 
o
#dense_9978/bias/Read/ReadVariableOpReadVariableOpdense_9978/bias*
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
§
%Adadelta/dense_9976/kernel/accum_gradVarHandleOp*
shape:	*6
shared_name'%Adadelta/dense_9976/kernel/accum_grad*
dtype0*
_output_shapes
: 
 
9Adadelta/dense_9976/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_9976/kernel/accum_grad*
dtype0*
_output_shapes
:	

#Adadelta/dense_9976/bias/accum_gradVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_9976/bias/accum_grad*
dtype0*
_output_shapes
: 

7Adadelta/dense_9976/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_9976/bias/accum_grad*
dtype0*
_output_shapes	
:
¨
%Adadelta/dense_9977/kernel/accum_gradVarHandleOp*
shape:
*6
shared_name'%Adadelta/dense_9977/kernel/accum_grad*
dtype0*
_output_shapes
: 
¡
9Adadelta/dense_9977/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_9977/kernel/accum_grad*
dtype0* 
_output_shapes
:


#Adadelta/dense_9977/bias/accum_gradVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_9977/bias/accum_grad*
dtype0*
_output_shapes
: 

7Adadelta/dense_9977/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_9977/bias/accum_grad*
dtype0*
_output_shapes	
:
§
%Adadelta/dense_9978/kernel/accum_gradVarHandleOp*
shape:	*6
shared_name'%Adadelta/dense_9978/kernel/accum_grad*
dtype0*
_output_shapes
: 
 
9Adadelta/dense_9978/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_9978/kernel/accum_grad*
dtype0*
_output_shapes
:	

#Adadelta/dense_9978/bias/accum_gradVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_9978/bias/accum_grad*
dtype0*
_output_shapes
: 

7Adadelta/dense_9978/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_9978/bias/accum_grad*
dtype0*
_output_shapes
:
¥
$Adadelta/dense_9976/kernel/accum_varVarHandleOp*
shape:	*5
shared_name&$Adadelta/dense_9976/kernel/accum_var*
dtype0*
_output_shapes
: 

8Adadelta/dense_9976/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_9976/kernel/accum_var*
dtype0*
_output_shapes
:	

"Adadelta/dense_9976/bias/accum_varVarHandleOp*
shape:*3
shared_name$"Adadelta/dense_9976/bias/accum_var*
dtype0*
_output_shapes
: 

6Adadelta/dense_9976/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_9976/bias/accum_var*
dtype0*
_output_shapes	
:
¦
$Adadelta/dense_9977/kernel/accum_varVarHandleOp*
shape:
*5
shared_name&$Adadelta/dense_9977/kernel/accum_var*
dtype0*
_output_shapes
: 

8Adadelta/dense_9977/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_9977/kernel/accum_var*
dtype0* 
_output_shapes
:


"Adadelta/dense_9977/bias/accum_varVarHandleOp*
shape:*3
shared_name$"Adadelta/dense_9977/bias/accum_var*
dtype0*
_output_shapes
: 

6Adadelta/dense_9977/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_9977/bias/accum_var*
dtype0*
_output_shapes	
:
¥
$Adadelta/dense_9978/kernel/accum_varVarHandleOp*
shape:	*5
shared_name&$Adadelta/dense_9978/kernel/accum_var*
dtype0*
_output_shapes
: 

8Adadelta/dense_9978/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_9978/kernel/accum_var*
dtype0*
_output_shapes
:	

"Adadelta/dense_9978/bias/accum_varVarHandleOp*
shape:*3
shared_name$"Adadelta/dense_9978/bias/accum_var*
dtype0*
_output_shapes
: 

6Adadelta/dense_9978/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_9978/bias/accum_var*
dtype0*
_output_shapes
:

NoOpNoOp
+
ConstConst"/device:CPU:0*Ã*
value¹*B¶* B¯*

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

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

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

regularization_losses

.layers
	variables
/metrics
0layer_regularization_losses
trainable_variables
1non_trainable_variables
][
VARIABLE_VALUEdense_9976/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9976/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses

2layers
	variables
3metrics
4layer_regularization_losses
trainable_variables
5non_trainable_variables
][
VARIABLE_VALUEdense_9977/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9977/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

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

regularization_losses

:layers
	variables
;metrics
<layer_regularization_losses
trainable_variables
=non_trainable_variables
][
VARIABLE_VALUEdense_9978/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_9978/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1

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

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

VARIABLE_VALUE%Adadelta/dense_9976/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_9976/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adadelta/dense_9977/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_9977/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adadelta/dense_9978/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_9978/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_9976/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/dense_9976/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_9977/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/dense_9977/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_9978/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/dense_9978/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

 serving_default_dense_9976_inputPlaceholder*
shape:ÿÿÿÿÿÿÿÿÿ*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_9976_inputdense_9976/kerneldense_9976/biasdense_9977/kerneldense_9977/biasdense_9978/kerneldense_9978/bias*/
_gradient_op_typePartitionedCall-16962262*/
f*R(
&__inference_signature_wrapper_16962030*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_9976/kernel/Read/ReadVariableOp#dense_9976/bias/Read/ReadVariableOp%dense_9977/kernel/Read/ReadVariableOp#dense_9977/bias/Read/ReadVariableOp%dense_9978/kernel/Read/ReadVariableOp#dense_9978/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp9Adadelta/dense_9976/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_9976/bias/accum_grad/Read/ReadVariableOp9Adadelta/dense_9977/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_9977/bias/accum_grad/Read/ReadVariableOp9Adadelta/dense_9978/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_9978/bias/accum_grad/Read/ReadVariableOp8Adadelta/dense_9976/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_9976/bias/accum_var/Read/ReadVariableOp8Adadelta/dense_9977/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_9977/bias/accum_var/Read/ReadVariableOp8Adadelta/dense_9978/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_9978/bias/accum_var/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-16962308**
f%R#
!__inference__traced_save_16962307*
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

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9976/kerneldense_9976/biasdense_9977/kerneldense_9977/biasdense_9978/kerneldense_9978/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcount%Adadelta/dense_9976/kernel/accum_grad#Adadelta/dense_9976/bias/accum_grad%Adadelta/dense_9977/kernel/accum_grad#Adadelta/dense_9977/bias/accum_grad%Adadelta/dense_9978/kernel/accum_grad#Adadelta/dense_9978/bias/accum_grad$Adadelta/dense_9976/kernel/accum_var"Adadelta/dense_9976/bias/accum_var$Adadelta/dense_9977/kernel/accum_var"Adadelta/dense_9977/bias/accum_var$Adadelta/dense_9978/kernel/accum_var"Adadelta/dense_9978/bias/accum_var*/
_gradient_op_typePartitionedCall-16962393*-
f(R&
$__inference__traced_restore_16962392*
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
: Ü
Æ	
Í
2__inference_sequential_2008_layer_call_fn_16961986
dense_9976_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_9976_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-16961977*V
fQRO
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961976*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9976_input: : 
Ã'
þ
#__inference__wrapped_model_16961808
dense_9976_input=
9sequential_2008_dense_9976_matmul_readvariableop_resource>
:sequential_2008_dense_9976_biasadd_readvariableop_resource=
9sequential_2008_dense_9977_matmul_readvariableop_resource>
:sequential_2008_dense_9977_biasadd_readvariableop_resource=
9sequential_2008_dense_9978_matmul_readvariableop_resource>
:sequential_2008_dense_9978_biasadd_readvariableop_resource
identity¢1sequential_2008/dense_9976/BiasAdd/ReadVariableOp¢0sequential_2008/dense_9976/MatMul/ReadVariableOp¢1sequential_2008/dense_9977/BiasAdd/ReadVariableOp¢0sequential_2008/dense_9977/MatMul/ReadVariableOp¢1sequential_2008/dense_9978/BiasAdd/ReadVariableOp¢0sequential_2008/dense_9978/MatMul/ReadVariableOpÙ
0sequential_2008/dense_9976/MatMul/ReadVariableOpReadVariableOp9sequential_2008_dense_9976_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ª
!sequential_2008/dense_9976/MatMulMatMuldense_9976_input8sequential_2008/dense_9976/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
1sequential_2008/dense_9976/BiasAdd/ReadVariableOpReadVariableOp:sequential_2008_dense_9976_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:È
"sequential_2008/dense_9976/BiasAddBiasAdd+sequential_2008/dense_9976/MatMul:product:09sequential_2008/dense_9976/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2008/dense_9976/ReluRelu+sequential_2008/dense_9976/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
0sequential_2008/dense_9977/MatMul/ReadVariableOpReadVariableOp9sequential_2008_dense_9977_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Ç
!sequential_2008/dense_9977/MatMulMatMul-sequential_2008/dense_9976/Relu:activations:08sequential_2008/dense_9977/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
1sequential_2008/dense_9977/BiasAdd/ReadVariableOpReadVariableOp:sequential_2008_dense_9977_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:È
"sequential_2008/dense_9977/BiasAddBiasAdd+sequential_2008/dense_9977/MatMul:product:09sequential_2008/dense_9977/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2008/dense_9977/ReluRelu+sequential_2008/dense_9977/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%sequential_2008/dropout_3978/IdentityIdentity-sequential_2008/dense_9977/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
0sequential_2008/dense_9978/MatMul/ReadVariableOpReadVariableOp9sequential_2008_dense_9978_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ç
!sequential_2008/dense_9978/MatMulMatMul.sequential_2008/dropout_3978/Identity:output:08sequential_2008/dense_9978/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
1sequential_2008/dense_9978/BiasAdd/ReadVariableOpReadVariableOp:sequential_2008_dense_9978_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ç
"sequential_2008/dense_9978/BiasAddBiasAdd+sequential_2008/dense_9978/MatMul:product:09sequential_2008/dense_9978/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_2008/dense_9978/ReluRelu+sequential_2008/dense_9978/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
IdentityIdentity-sequential_2008/dense_9978/Relu:activations:02^sequential_2008/dense_9976/BiasAdd/ReadVariableOp1^sequential_2008/dense_9976/MatMul/ReadVariableOp2^sequential_2008/dense_9977/BiasAdd/ReadVariableOp1^sequential_2008/dense_9977/MatMul/ReadVariableOp2^sequential_2008/dense_9978/BiasAdd/ReadVariableOp1^sequential_2008/dense_9978/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2d
0sequential_2008/dense_9977/MatMul/ReadVariableOp0sequential_2008/dense_9977/MatMul/ReadVariableOp2f
1sequential_2008/dense_9978/BiasAdd/ReadVariableOp1sequential_2008/dense_9978/BiasAdd/ReadVariableOp2d
0sequential_2008/dense_9976/MatMul/ReadVariableOp0sequential_2008/dense_9976/MatMul/ReadVariableOp2f
1sequential_2008/dense_9977/BiasAdd/ReadVariableOp1sequential_2008/dense_9977/BiasAdd/ReadVariableOp2d
0sequential_2008/dense_9978/MatMul/ReadVariableOp0sequential_2008/dense_9978/MatMul/ReadVariableOp2f
1sequential_2008/dense_9976/BiasAdd/ReadVariableOp1sequential_2008/dense_9976/BiasAdd/ReadVariableOp: : : : :0 ,
*
_user_specified_namedense_9976_input: : 
õb

$__inference__traced_restore_16962392
file_prefix&
"assignvariableop_dense_9976_kernel&
"assignvariableop_1_dense_9976_bias(
$assignvariableop_2_dense_9977_kernel&
"assignvariableop_3_dense_9977_bias(
$assignvariableop_4_dense_9978_kernel&
"assignvariableop_5_dense_9978_bias$
 assignvariableop_6_adadelta_iter%
!assignvariableop_7_adadelta_decay-
)assignvariableop_8_adadelta_learning_rate#
assignvariableop_9_adadelta_rho
assignvariableop_10_total
assignvariableop_11_count=
9assignvariableop_12_adadelta_dense_9976_kernel_accum_grad;
7assignvariableop_13_adadelta_dense_9976_bias_accum_grad=
9assignvariableop_14_adadelta_dense_9977_kernel_accum_grad;
7assignvariableop_15_adadelta_dense_9977_bias_accum_grad=
9assignvariableop_16_adadelta_dense_9978_kernel_accum_grad;
7assignvariableop_17_adadelta_dense_9978_bias_accum_grad<
8assignvariableop_18_adadelta_dense_9976_kernel_accum_var:
6assignvariableop_19_adadelta_dense_9976_bias_accum_var<
8assignvariableop_20_adadelta_dense_9977_kernel_accum_var:
6assignvariableop_21_adadelta_dense_9977_bias_accum_var<
8assignvariableop_22_adadelta_dense_9978_kernel_accum_var:
6assignvariableop_23_adadelta_dense_9978_bias_accum_var
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
:~
AssignVariableOpAssignVariableOp"assignvariableop_dense_9976_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_9976_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_9977_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_9977_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_9978_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_9978_biasIdentity_5:output:0*
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
:
AssignVariableOp_12AssignVariableOp9assignvariableop_12_adadelta_dense_9976_kernel_accum_gradIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp7assignvariableop_13_adadelta_dense_9976_bias_accum_gradIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp9assignvariableop_14_adadelta_dense_9977_kernel_accum_gradIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adadelta_dense_9977_bias_accum_gradIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp9assignvariableop_16_adadelta_dense_9978_kernel_accum_gradIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adadelta_dense_9978_bias_accum_gradIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adadelta_dense_9976_kernel_accum_varIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adadelta_dense_9976_bias_accum_varIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adadelta_dense_9977_kernel_accum_varIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adadelta_dense_9977_bias_accum_varIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adadelta_dense_9978_kernel_accum_varIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adadelta_dense_9978_bias_accum_varIdentity_23:output:0*
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
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192$
AssignVariableOpAssignVariableOp: : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : :
 

Þ
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962099

inputs-
)dense_9976_matmul_readvariableop_resource.
*dense_9976_biasadd_readvariableop_resource-
)dense_9977_matmul_readvariableop_resource.
*dense_9977_biasadd_readvariableop_resource-
)dense_9978_matmul_readvariableop_resource.
*dense_9978_biasadd_readvariableop_resource
identity¢!dense_9976/BiasAdd/ReadVariableOp¢ dense_9976/MatMul/ReadVariableOp¢!dense_9977/BiasAdd/ReadVariableOp¢ dense_9977/MatMul/ReadVariableOp¢!dense_9978/BiasAdd/ReadVariableOp¢ dense_9978/MatMul/ReadVariableOp¹
 dense_9976/MatMul/ReadVariableOpReadVariableOp)dense_9976_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_9976/MatMulMatMulinputs(dense_9976/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_9976/BiasAdd/ReadVariableOpReadVariableOp*dense_9976_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_9976/BiasAddBiasAdddense_9976/MatMul:product:0)dense_9976/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dense_9976/ReluReludense_9976/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_9977/MatMul/ReadVariableOpReadVariableOp)dense_9977_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_9977/MatMulMatMuldense_9976/Relu:activations:0(dense_9977/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_9977/BiasAdd/ReadVariableOpReadVariableOp*dense_9977_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_9977/BiasAddBiasAdddense_9977/MatMul:product:0)dense_9977/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dense_9977/ReluReludense_9977/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_3978/IdentityIdentitydense_9977/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 dense_9978/MatMul/ReadVariableOpReadVariableOp)dense_9978_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_9978/MatMulMatMuldropout_3978/Identity:output:0(dense_9978/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!dense_9978/BiasAdd/ReadVariableOpReadVariableOp*dense_9978_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_9978/BiasAddBiasAdddense_9978/MatMul:product:0)dense_9978/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9978/ReluReludense_9978/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
IdentityIdentitydense_9978/Relu:activations:0"^dense_9976/BiasAdd/ReadVariableOp!^dense_9976/MatMul/ReadVariableOp"^dense_9977/BiasAdd/ReadVariableOp!^dense_9977/MatMul/ReadVariableOp"^dense_9978/BiasAdd/ReadVariableOp!^dense_9978/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_9976/MatMul/ReadVariableOp dense_9976/MatMul/ReadVariableOp2D
 dense_9978/MatMul/ReadVariableOp dense_9978/MatMul/ReadVariableOp2F
!dense_9978/BiasAdd/ReadVariableOp!dense_9978/BiasAdd/ReadVariableOp2F
!dense_9977/BiasAdd/ReadVariableOp!dense_9977/BiasAdd/ReadVariableOp2F
!dense_9976/BiasAdd/ReadVariableOp!dense_9976/BiasAdd/ReadVariableOp2D
 dense_9977/MatMul/ReadVariableOp dense_9977/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 

h
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961897

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
å
®
-__inference_dense_9977_layer_call_fn_16962157

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961859*Q
fLRJ
H__inference_dense_9977_layer_call_and_return_conditional_losses_16961853*
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
ö/
Þ
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962073

inputs-
)dense_9976_matmul_readvariableop_resource.
*dense_9976_biasadd_readvariableop_resource-
)dense_9977_matmul_readvariableop_resource.
*dense_9977_biasadd_readvariableop_resource-
)dense_9978_matmul_readvariableop_resource.
*dense_9978_biasadd_readvariableop_resource
identity¢!dense_9976/BiasAdd/ReadVariableOp¢ dense_9976/MatMul/ReadVariableOp¢!dense_9977/BiasAdd/ReadVariableOp¢ dense_9977/MatMul/ReadVariableOp¢!dense_9978/BiasAdd/ReadVariableOp¢ dense_9978/MatMul/ReadVariableOp¹
 dense_9976/MatMul/ReadVariableOpReadVariableOp)dense_9976_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_9976/MatMulMatMulinputs(dense_9976/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_9976/BiasAdd/ReadVariableOpReadVariableOp*dense_9976_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_9976/BiasAddBiasAdddense_9976/MatMul:product:0)dense_9976/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dense_9976/ReluReludense_9976/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_9977/MatMul/ReadVariableOpReadVariableOp)dense_9977_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_9977/MatMulMatMuldense_9976/Relu:activations:0(dense_9977/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_9977/BiasAdd/ReadVariableOpReadVariableOp*dense_9977_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_9977/BiasAddBiasAdddense_9977/MatMul:product:0)dense_9977/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
dense_9977/ReluReludense_9977/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dropout_3978/dropout/rateConst*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: g
dropout_3978/dropout/ShapeShapedense_9977/Relu:activations:0*
T0*
_output_shapes
:l
'dropout_3978/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_3978/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: §
1dropout_3978/dropout/random_uniform/RandomUniformRandomUniform#dropout_3978/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
'dropout_3978/dropout/random_uniform/subSub0dropout_3978/dropout/random_uniform/max:output:00dropout_3978/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ê
'dropout_3978/dropout/random_uniform/mulMul:dropout_3978/dropout/random_uniform/RandomUniform:output:0+dropout_3978/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
#dropout_3978/dropout/random_uniformAdd+dropout_3978/dropout/random_uniform/mul:z:00dropout_3978/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_3978/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_3978/dropout/subSub#dropout_3978/dropout/sub/x:output:0"dropout_3978/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_3978/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_3978/dropout/truedivRealDiv'dropout_3978/dropout/truediv/x:output:0dropout_3978/dropout/sub:z:0*
T0*
_output_shapes
: ±
!dropout_3978/dropout/GreaterEqualGreaterEqual'dropout_3978/dropout/random_uniform:z:0"dropout_3978/dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3978/dropout/mulMuldense_9977/Relu:activations:0 dropout_3978/dropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3978/dropout/CastCast%dropout_3978/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_3978/dropout/mul_1Muldropout_3978/dropout/mul:z:0dropout_3978/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 dense_9978/MatMul/ReadVariableOpReadVariableOp)dense_9978_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_9978/MatMulMatMuldropout_3978/dropout/mul_1:z:0(dense_9978/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!dense_9978/BiasAdd/ReadVariableOpReadVariableOp*dense_9978_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_9978/BiasAddBiasAdddense_9978/MatMul:product:0)dense_9978/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_9978/ReluReludense_9978/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
IdentityIdentitydense_9978/Relu:activations:0"^dense_9976/BiasAdd/ReadVariableOp!^dense_9976/MatMul/ReadVariableOp"^dense_9977/BiasAdd/ReadVariableOp!^dense_9977/MatMul/ReadVariableOp"^dense_9978/BiasAdd/ReadVariableOp!^dense_9978/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_9976/MatMul/ReadVariableOp dense_9976/MatMul/ReadVariableOp2F
!dense_9978/BiasAdd/ReadVariableOp!dense_9978/BiasAdd/ReadVariableOp2D
 dense_9978/MatMul/ReadVariableOp dense_9978/MatMul/ReadVariableOp2F
!dense_9977/BiasAdd/ReadVariableOp!dense_9977/BiasAdd/ReadVariableOp2F
!dense_9976/BiasAdd/ReadVariableOp!dense_9976/BiasAdd/ReadVariableOp2D
 dense_9977/MatMul/ReadVariableOp dense_9977/MatMul/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 

¦
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961943
dense_9976_input-
)dense_9976_statefulpartitionedcall_args_1-
)dense_9976_statefulpartitionedcall_args_2-
)dense_9977_statefulpartitionedcall_args_1-
)dense_9977_statefulpartitionedcall_args_2-
)dense_9978_statefulpartitionedcall_args_1-
)dense_9978_statefulpartitionedcall_args_2
identity¢"dense_9976/StatefulPartitionedCall¢"dense_9977/StatefulPartitionedCall¢"dense_9978/StatefulPartitionedCall¢$dropout_3978/StatefulPartitionedCall
"dense_9976/StatefulPartitionedCallStatefulPartitionedCalldense_9976_input)dense_9976_statefulpartitionedcall_args_1)dense_9976_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961831*Q
fLRJ
H__inference_dense_9976_layer_call_and_return_conditional_losses_16961825*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_9977/StatefulPartitionedCallStatefulPartitionedCall+dense_9976/StatefulPartitionedCall:output:0)dense_9977_statefulpartitionedcall_args_1)dense_9977_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961859*Q
fLRJ
H__inference_dense_9977_layer_call_and_return_conditional_losses_16961853*
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
:ÿÿÿÿÿÿÿÿÿå
$dropout_3978/StatefulPartitionedCallStatefulPartitionedCall+dense_9977/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-16961901*S
fNRL
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961890*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"dense_9978/StatefulPartitionedCallStatefulPartitionedCall-dropout_3978/StatefulPartitionedCall:output:0)dense_9978_statefulpartitionedcall_args_1)dense_9978_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961931*Q
fLRJ
H__inference_dense_9978_layer_call_and_return_conditional_losses_16961925*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity+dense_9978/StatefulPartitionedCall:output:0#^dense_9976/StatefulPartitionedCall#^dense_9977/StatefulPartitionedCall#^dense_9978/StatefulPartitionedCall%^dropout_3978/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2L
$dropout_3978/StatefulPartitionedCall$dropout_3978/StatefulPartitionedCall2H
"dense_9976/StatefulPartitionedCall"dense_9976/StatefulPartitionedCall2H
"dense_9977/StatefulPartitionedCall"dense_9977/StatefulPartitionedCall2H
"dense_9978/StatefulPartitionedCall"dense_9978/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9976_input: : 
È9
í
!__inference__traced_save_16962307
file_prefix0
,savev2_dense_9976_kernel_read_readvariableop.
*savev2_dense_9976_bias_read_readvariableop0
,savev2_dense_9977_kernel_read_readvariableop.
*savev2_dense_9977_bias_read_readvariableop0
,savev2_dense_9978_kernel_read_readvariableop.
*savev2_dense_9978_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopD
@savev2_adadelta_dense_9976_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_9976_bias_accum_grad_read_readvariableopD
@savev2_adadelta_dense_9977_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_9977_bias_accum_grad_read_readvariableopD
@savev2_adadelta_dense_9978_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_9978_bias_accum_grad_read_readvariableopC
?savev2_adadelta_dense_9976_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_9976_bias_accum_var_read_readvariableopC
?savev2_adadelta_dense_9977_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_9977_bias_accum_var_read_readvariableopC
?savev2_adadelta_dense_9978_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_9978_bias_accum_var_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_6877ae69912b436086188132d7fcc169/part*
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
:Â
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_9976_kernel_read_readvariableop*savev2_dense_9976_bias_read_readvariableop,savev2_dense_9977_kernel_read_readvariableop*savev2_dense_9977_bias_read_readvariableop,savev2_dense_9978_kernel_read_readvariableop*savev2_dense_9978_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop@savev2_adadelta_dense_9976_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_9976_bias_accum_grad_read_readvariableop@savev2_adadelta_dense_9977_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_9977_bias_accum_grad_read_readvariableop@savev2_adadelta_dense_9978_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_9978_bias_accum_grad_read_readvariableop?savev2_adadelta_dense_9976_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_9976_bias_accum_var_read_readvariableop?savev2_adadelta_dense_9977_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_9977_bias_accum_var_read_readvariableop?savev2_adadelta_dense_9978_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_9978_bias_accum_var_read_readvariableop"/device:CPU:0*&
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
²: :	::
::	:: : : : : : :	::
::	::	::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : :
 
Ê
h
/__inference_dropout_3978_layer_call_fn_16962187

inputs
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputs*/
_gradient_op_typePartitionedCall-16961901*S
fNRL
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961890*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Æ	
Í
2__inference_sequential_2008_layer_call_fn_16962014
dense_9976_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_9976_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-16962005*V
fQRO
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962004*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9976_input: : 
Ú	
á
H__inference_dense_9976_layer_call_and_return_conditional_losses_16962132

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
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
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Ö	
á
H__inference_dense_9978_layer_call_and_return_conditional_losses_16961925

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
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ä
ÿ
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961959
dense_9976_input-
)dense_9976_statefulpartitionedcall_args_1-
)dense_9976_statefulpartitionedcall_args_2-
)dense_9977_statefulpartitionedcall_args_1-
)dense_9977_statefulpartitionedcall_args_2-
)dense_9978_statefulpartitionedcall_args_1-
)dense_9978_statefulpartitionedcall_args_2
identity¢"dense_9976/StatefulPartitionedCall¢"dense_9977/StatefulPartitionedCall¢"dense_9978/StatefulPartitionedCall
"dense_9976/StatefulPartitionedCallStatefulPartitionedCalldense_9976_input)dense_9976_statefulpartitionedcall_args_1)dense_9976_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961831*Q
fLRJ
H__inference_dense_9976_layer_call_and_return_conditional_losses_16961825*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_9977/StatefulPartitionedCallStatefulPartitionedCall+dense_9976/StatefulPartitionedCall:output:0)dense_9977_statefulpartitionedcall_args_1)dense_9977_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961859*Q
fLRJ
H__inference_dense_9977_layer_call_and_return_conditional_losses_16961853*
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
:ÿÿÿÿÿÿÿÿÿÕ
dropout_3978/PartitionedCallPartitionedCall+dense_9977/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-16961909*S
fNRL
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961897*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
"dense_9978/StatefulPartitionedCallStatefulPartitionedCall%dropout_3978/PartitionedCall:output:0)dense_9978_statefulpartitionedcall_args_1)dense_9978_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961931*Q
fLRJ
H__inference_dense_9978_layer_call_and_return_conditional_losses_16961925*
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
:ÿÿÿÿÿÿÿÿÿâ
IdentityIdentity+dense_9978/StatefulPartitionedCall:output:0#^dense_9976/StatefulPartitionedCall#^dense_9977/StatefulPartitionedCall#^dense_9978/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_9976/StatefulPartitionedCall"dense_9976/StatefulPartitionedCall2H
"dense_9977/StatefulPartitionedCall"dense_9977/StatefulPartitionedCall2H
"dense_9978/StatefulPartitionedCall"dense_9978/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9976_input: : 
Ü	
á
H__inference_dense_9977_layer_call_and_return_conditional_losses_16962150

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
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
·
i
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16962177

inputs
identityQ
dropout/rateConst*
valueB
 *ÍÌÌ=*
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
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: £
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
	
Á
&__inference_signature_wrapper_16962030
dense_9976_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCalldense_9976_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-16962021*,
f'R%
#__inference__wrapped_model_16961808*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_9976_input: : 
Ü	
á
H__inference_dense_9977_layer_call_and_return_conditional_losses_16961853

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
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
·
i
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961890

inputs
identityQ
dropout/rateConst*
valueB
 *ÍÌÌ=*
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
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: £
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
ú

M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961976

inputs-
)dense_9976_statefulpartitionedcall_args_1-
)dense_9976_statefulpartitionedcall_args_2-
)dense_9977_statefulpartitionedcall_args_1-
)dense_9977_statefulpartitionedcall_args_2-
)dense_9978_statefulpartitionedcall_args_1-
)dense_9978_statefulpartitionedcall_args_2
identity¢"dense_9976/StatefulPartitionedCall¢"dense_9977/StatefulPartitionedCall¢"dense_9978/StatefulPartitionedCall¢$dropout_3978/StatefulPartitionedCall
"dense_9976/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_9976_statefulpartitionedcall_args_1)dense_9976_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961831*Q
fLRJ
H__inference_dense_9976_layer_call_and_return_conditional_losses_16961825*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_9977/StatefulPartitionedCallStatefulPartitionedCall+dense_9976/StatefulPartitionedCall:output:0)dense_9977_statefulpartitionedcall_args_1)dense_9977_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961859*Q
fLRJ
H__inference_dense_9977_layer_call_and_return_conditional_losses_16961853*
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
:ÿÿÿÿÿÿÿÿÿå
$dropout_3978/StatefulPartitionedCallStatefulPartitionedCall+dense_9977/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-16961901*S
fNRL
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961890*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"dense_9978/StatefulPartitionedCallStatefulPartitionedCall-dropout_3978/StatefulPartitionedCall:output:0)dense_9978_statefulpartitionedcall_args_1)dense_9978_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961931*Q
fLRJ
H__inference_dense_9978_layer_call_and_return_conditional_losses_16961925*
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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity+dense_9978/StatefulPartitionedCall:output:0#^dense_9976/StatefulPartitionedCall#^dense_9977/StatefulPartitionedCall#^dense_9978/StatefulPartitionedCall%^dropout_3978/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2L
$dropout_3978/StatefulPartitionedCall$dropout_3978/StatefulPartitionedCall2H
"dense_9976/StatefulPartitionedCall"dense_9976/StatefulPartitionedCall2H
"dense_9977/StatefulPartitionedCall"dense_9977/StatefulPartitionedCall2H
"dense_9978/StatefulPartitionedCall"dense_9978/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
¨	
Ã
2__inference_sequential_2008_layer_call_fn_16962110

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-16961977*V
fQRO
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961976*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
¨	
Ã
2__inference_sequential_2008_layer_call_fn_16962121

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*/
_gradient_op_typePartitionedCall-16962005*V
fQRO
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962004*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
Æ
K
/__inference_dropout_3978_layer_call_fn_16962192

inputs
identity£
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-16961909*S
fNRL
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961897*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
Ú	
á
H__inference_dense_9976_layer_call_and_return_conditional_losses_16961825

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
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
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Æ
õ
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962004

inputs-
)dense_9976_statefulpartitionedcall_args_1-
)dense_9976_statefulpartitionedcall_args_2-
)dense_9977_statefulpartitionedcall_args_1-
)dense_9977_statefulpartitionedcall_args_2-
)dense_9978_statefulpartitionedcall_args_1-
)dense_9978_statefulpartitionedcall_args_2
identity¢"dense_9976/StatefulPartitionedCall¢"dense_9977/StatefulPartitionedCall¢"dense_9978/StatefulPartitionedCall
"dense_9976/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_9976_statefulpartitionedcall_args_1)dense_9976_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961831*Q
fLRJ
H__inference_dense_9976_layer_call_and_return_conditional_losses_16961825*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_9977/StatefulPartitionedCallStatefulPartitionedCall+dense_9976/StatefulPartitionedCall:output:0)dense_9977_statefulpartitionedcall_args_1)dense_9977_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961859*Q
fLRJ
H__inference_dense_9977_layer_call_and_return_conditional_losses_16961853*
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
:ÿÿÿÿÿÿÿÿÿÕ
dropout_3978/PartitionedCallPartitionedCall+dense_9977/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-16961909*S
fNRL
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16961897*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ²
"dense_9978/StatefulPartitionedCallStatefulPartitionedCall%dropout_3978/PartitionedCall:output:0)dense_9978_statefulpartitionedcall_args_1)dense_9978_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961931*Q
fLRJ
H__inference_dense_9978_layer_call_and_return_conditional_losses_16961925*
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
:ÿÿÿÿÿÿÿÿÿâ
IdentityIdentity+dense_9978/StatefulPartitionedCall:output:0#^dense_9976/StatefulPartitionedCall#^dense_9977/StatefulPartitionedCall#^dense_9978/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_9976/StatefulPartitionedCall"dense_9976/StatefulPartitionedCall2H
"dense_9977/StatefulPartitionedCall"dense_9977/StatefulPartitionedCall2H
"dense_9978/StatefulPartitionedCall"dense_9978/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
ã
®
-__inference_dense_9978_layer_call_fn_16962210

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961931*Q
fLRJ
H__inference_dense_9978_layer_call_and_return_conditional_losses_16961925*
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

h
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16962182

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:& "
 
_user_specified_nameinputs
Ö	
á
H__inference_dense_9978_layer_call_and_return_conditional_losses_16962203

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
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ä
®
-__inference_dense_9976_layer_call_fn_16962139

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-16961831*Q
fLRJ
H__inference_dense_9976_layer_call_and_return_conditional_losses_16961825*
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
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*¿
serving_default«
M
dense_9976_input9
"serving_default_dense_9976_input:0ÿÿÿÿÿÿÿÿÿ>

dense_99780
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ó£
¿!
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
*\&call_and_return_all_conditional_losses"å
_tf_keras_sequentialÆ{"class_name": "Sequential", "name": "sequential_2008", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2008", "layers": [{"class_name": "Dense", "config": {"name": "dense_9976", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9977", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3978", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_9978", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2008", "layers": [{"class_name": "Dense", "config": {"name": "dense_9976", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_9977", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_3978", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_9978", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
³
regularization_losses
	variables
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"¤
_tf_keras_layer{"class_name": "InputLayer", "name": "dense_9976_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 8], "config": {"batch_input_shape": [null, 8], "dtype": "float32", "sparse": false, "name": "dense_9976_input"}}
¸

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layerù{"class_name": "Dense", "name": "dense_9976", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 8], "config": {"name": "dense_9976", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "Dense", "name": "dense_9977", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9977", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
µ
regularization_losses
	variables
trainable_variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"¦
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_3978", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_3978", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}


 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
e__call__
*f&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Dense", "name": "dense_9978", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_9978", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}

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
·
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

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
$:"	2dense_9976/kernel
:2dense_9976/bias
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

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
%:#
2dense_9977/kernel
:2dense_9977/bias
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

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

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
$:"	2dense_9978/kernel
:2dense_9978/bias
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

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

	Ctotal
	Dcount
E
_fn_kwargs
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h__call__
*i&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float32"}}
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

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
6:4	2%Adadelta/dense_9976/kernel/accum_grad
0:.2#Adadelta/dense_9976/bias/accum_grad
7:5
2%Adadelta/dense_9977/kernel/accum_grad
0:.2#Adadelta/dense_9977/bias/accum_grad
6:4	2%Adadelta/dense_9978/kernel/accum_grad
/:-2#Adadelta/dense_9978/bias/accum_grad
5:3	2$Adadelta/dense_9976/kernel/accum_var
/:-2"Adadelta/dense_9976/bias/accum_var
6:4
2$Adadelta/dense_9977/kernel/accum_var
/:-2"Adadelta/dense_9977/bias/accum_var
5:3	2$Adadelta/dense_9978/kernel/accum_var
.:,2"Adadelta/dense_9978/bias/accum_var
2
2__inference_sequential_2008_layer_call_fn_16962014
2__inference_sequential_2008_layer_call_fn_16961986
2__inference_sequential_2008_layer_call_fn_16962110
2__inference_sequential_2008_layer_call_fn_16962121À
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
ê2ç
#__inference__wrapped_model_16961808¿
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
annotationsª */¢,
*'
dense_9976_inputÿÿÿÿÿÿÿÿÿ
2ÿ
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962099
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962073
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961943
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961959À
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
×2Ô
-__inference_dense_9976_layer_call_fn_16962139¢
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
ò2ï
H__inference_dense_9976_layer_call_and_return_conditional_losses_16962132¢
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
×2Ô
-__inference_dense_9977_layer_call_fn_16962157¢
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
ò2ï
H__inference_dense_9977_layer_call_and_return_conditional_losses_16962150¢
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
2
/__inference_dropout_3978_layer_call_fn_16962192
/__inference_dropout_3978_layer_call_fn_16962187´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16962177
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16962182´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
×2Ô
-__inference_dense_9978_layer_call_fn_16962210¢
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
ò2ï
H__inference_dense_9978_layer_call_and_return_conditional_losses_16962203¢
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
>B<
&__inference_signature_wrapper_16962030dense_9976_input
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
 £
#__inference__wrapped_model_16961808| !9¢6
/¢,
*'
dense_9976_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

dense_9978$!

dense_9978ÿÿÿÿÿÿÿÿÿ
/__inference_dropout_3978_layer_call_fn_16962192Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_dropout_3978_layer_call_fn_16962187Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ»
&__inference_signature_wrapper_16962030 !M¢J
¢ 
Cª@
>
dense_9976_input*'
dense_9976_inputÿÿÿÿÿÿÿÿÿ"7ª4
2

dense_9978$!

dense_9978ÿÿÿÿÿÿÿÿÿ
2__inference_sequential_2008_layer_call_fn_16961986e !A¢>
7¢4
*'
dense_9976_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962073h !7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961959r !A¢>
7¢4
*'
dense_9976_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ©
H__inference_dense_9978_layer_call_and_return_conditional_losses_16962203] !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ©
H__inference_dense_9976_layer_call_and_return_conditional_losses_16962132]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ª
H__inference_dense_9977_layer_call_and_return_conditional_losses_16962150^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dense_9977_layer_call_fn_16962157Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
2__inference_sequential_2008_layer_call_fn_16962110[ !7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16962099h !7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_sequential_2008_layer_call_fn_16962121[ !7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_dense_9976_layer_call_fn_16962139P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_dense_9978_layer_call_fn_16962210P !0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16962177^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¬
J__inference_dropout_3978_layer_call_and_return_conditional_losses_16962182^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ã
M__inference_sequential_2008_layer_call_and_return_conditional_losses_16961943r !A¢>
7¢4
*'
dense_9976_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_sequential_2008_layer_call_fn_16962014e !A¢>
7¢4
*'
dense_9976_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ