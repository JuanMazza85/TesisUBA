øÊ
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
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8è³

dense_3889/kernelVarHandleOp*
shape:	*"
shared_namedense_3889/kernel*
dtype0*
_output_shapes
: 
x
%dense_3889/kernel/Read/ReadVariableOpReadVariableOpdense_3889/kernel*
dtype0*
_output_shapes
:	
w
dense_3889/biasVarHandleOp*
shape:* 
shared_namedense_3889/bias*
dtype0*
_output_shapes
: 
p
#dense_3889/bias/Read/ReadVariableOpReadVariableOpdense_3889/bias*
dtype0*
_output_shapes	
:

dense_3890/kernelVarHandleOp*
shape:
*"
shared_namedense_3890/kernel*
dtype0*
_output_shapes
: 
y
%dense_3890/kernel/Read/ReadVariableOpReadVariableOpdense_3890/kernel*
dtype0* 
_output_shapes
:

w
dense_3890/biasVarHandleOp*
shape:* 
shared_namedense_3890/bias*
dtype0*
_output_shapes
: 
p
#dense_3890/bias/Read/ReadVariableOpReadVariableOpdense_3890/bias*
dtype0*
_output_shapes	
:

dense_3891/kernelVarHandleOp*
shape:	*"
shared_namedense_3891/kernel*
dtype0*
_output_shapes
: 
x
%dense_3891/kernel/Read/ReadVariableOpReadVariableOpdense_3891/kernel*
dtype0*
_output_shapes
:	
v
dense_3891/biasVarHandleOp*
shape:* 
shared_namedense_3891/bias*
dtype0*
_output_shapes
: 
o
#dense_3891/bias/Read/ReadVariableOpReadVariableOpdense_3891/bias*
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
%Adadelta/dense_3889/kernel/accum_gradVarHandleOp*
shape:	*6
shared_name'%Adadelta/dense_3889/kernel/accum_grad*
dtype0*
_output_shapes
: 
 
9Adadelta/dense_3889/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_3889/kernel/accum_grad*
dtype0*
_output_shapes
:	

#Adadelta/dense_3889/bias/accum_gradVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_3889/bias/accum_grad*
dtype0*
_output_shapes
: 

7Adadelta/dense_3889/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_3889/bias/accum_grad*
dtype0*
_output_shapes	
:
¨
%Adadelta/dense_3890/kernel/accum_gradVarHandleOp*
shape:
*6
shared_name'%Adadelta/dense_3890/kernel/accum_grad*
dtype0*
_output_shapes
: 
¡
9Adadelta/dense_3890/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_3890/kernel/accum_grad*
dtype0* 
_output_shapes
:


#Adadelta/dense_3890/bias/accum_gradVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_3890/bias/accum_grad*
dtype0*
_output_shapes
: 

7Adadelta/dense_3890/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_3890/bias/accum_grad*
dtype0*
_output_shapes	
:
§
%Adadelta/dense_3891/kernel/accum_gradVarHandleOp*
shape:	*6
shared_name'%Adadelta/dense_3891/kernel/accum_grad*
dtype0*
_output_shapes
: 
 
9Adadelta/dense_3891/kernel/accum_grad/Read/ReadVariableOpReadVariableOp%Adadelta/dense_3891/kernel/accum_grad*
dtype0*
_output_shapes
:	

#Adadelta/dense_3891/bias/accum_gradVarHandleOp*
shape:*4
shared_name%#Adadelta/dense_3891/bias/accum_grad*
dtype0*
_output_shapes
: 

7Adadelta/dense_3891/bias/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_3891/bias/accum_grad*
dtype0*
_output_shapes
:
¥
$Adadelta/dense_3889/kernel/accum_varVarHandleOp*
shape:	*5
shared_name&$Adadelta/dense_3889/kernel/accum_var*
dtype0*
_output_shapes
: 

8Adadelta/dense_3889/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_3889/kernel/accum_var*
dtype0*
_output_shapes
:	

"Adadelta/dense_3889/bias/accum_varVarHandleOp*
shape:*3
shared_name$"Adadelta/dense_3889/bias/accum_var*
dtype0*
_output_shapes
: 

6Adadelta/dense_3889/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_3889/bias/accum_var*
dtype0*
_output_shapes	
:
¦
$Adadelta/dense_3890/kernel/accum_varVarHandleOp*
shape:
*5
shared_name&$Adadelta/dense_3890/kernel/accum_var*
dtype0*
_output_shapes
: 

8Adadelta/dense_3890/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_3890/kernel/accum_var*
dtype0* 
_output_shapes
:


"Adadelta/dense_3890/bias/accum_varVarHandleOp*
shape:*3
shared_name$"Adadelta/dense_3890/bias/accum_var*
dtype0*
_output_shapes
: 

6Adadelta/dense_3890/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_3890/bias/accum_var*
dtype0*
_output_shapes	
:
¥
$Adadelta/dense_3891/kernel/accum_varVarHandleOp*
shape:	*5
shared_name&$Adadelta/dense_3891/kernel/accum_var*
dtype0*
_output_shapes
: 

8Adadelta/dense_3891/kernel/accum_var/Read/ReadVariableOpReadVariableOp$Adadelta/dense_3891/kernel/accum_var*
dtype0*
_output_shapes
:	

"Adadelta/dense_3891/bias/accum_varVarHandleOp*
shape:*3
shared_name$"Adadelta/dense_3891/bias/accum_var*
dtype0*
_output_shapes
: 

6Adadelta/dense_3891/bias/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_3891/bias/accum_var*
dtype0*
_output_shapes
:

NoOpNoOp
õ(
ConstConst"/device:CPU:0*°(
value¦(B£( B(
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
][
VARIABLE_VALUEdense_3889/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3889/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
][
VARIABLE_VALUEdense_3890/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3890/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
][
VARIABLE_VALUEdense_3891/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3891/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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

VARIABLE_VALUE%Adadelta/dense_3889/kernel/accum_grad[layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_3889/bias/accum_gradYlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adadelta/dense_3890/kernel/accum_grad[layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_3890/bias/accum_gradYlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE%Adadelta/dense_3891/kernel/accum_grad[layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adadelta/dense_3891/bias/accum_gradYlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_3889/kernel/accum_varZlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/dense_3889/bias/accum_varXlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_3890/kernel/accum_varZlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/dense_3890/bias/accum_varXlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE$Adadelta/dense_3891/kernel/accum_varZlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adadelta/dense_3891/bias/accum_varXlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

 serving_default_dense_3889_inputPlaceholder*
shape:ÿÿÿÿÿÿÿÿÿ*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_3889_inputdense_3889/kerneldense_3889/biasdense_3890/kerneldense_3890/biasdense_3891/kerneldense_3891/bias*.
_gradient_op_typePartitionedCall-6618206*.
f)R'
%__inference_signature_wrapper_6618032*
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

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_3889/kernel/Read/ReadVariableOp#dense_3889/bias/Read/ReadVariableOp%dense_3890/kernel/Read/ReadVariableOp#dense_3890/bias/Read/ReadVariableOp%dense_3891/kernel/Read/ReadVariableOp#dense_3891/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp9Adadelta/dense_3889/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_3889/bias/accum_grad/Read/ReadVariableOp9Adadelta/dense_3890/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_3890/bias/accum_grad/Read/ReadVariableOp9Adadelta/dense_3891/kernel/accum_grad/Read/ReadVariableOp7Adadelta/dense_3891/bias/accum_grad/Read/ReadVariableOp8Adadelta/dense_3889/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_3889/bias/accum_var/Read/ReadVariableOp8Adadelta/dense_3890/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_3890/bias/accum_var/Read/ReadVariableOp8Adadelta/dense_3891/kernel/accum_var/Read/ReadVariableOp6Adadelta/dense_3891/bias/accum_var/Read/ReadVariableOpConst*.
_gradient_op_typePartitionedCall-6618252*)
f$R"
 __inference__traced_save_6618251*
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

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3889/kerneldense_3889/biasdense_3890/kerneldense_3890/biasdense_3891/kerneldense_3891/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcount%Adadelta/dense_3889/kernel/accum_grad#Adadelta/dense_3889/bias/accum_grad%Adadelta/dense_3890/kernel/accum_grad#Adadelta/dense_3890/bias/accum_grad%Adadelta/dense_3891/kernel/accum_grad#Adadelta/dense_3891/bias/accum_grad$Adadelta/dense_3889/kernel/accum_var"Adadelta/dense_3889/bias/accum_var$Adadelta/dense_3890/kernel/accum_var"Adadelta/dense_3890/bias/accum_var$Adadelta/dense_3891/kernel/accum_var"Adadelta/dense_3891/bias/accum_var*.
_gradient_op_typePartitionedCall-6618337*,
f'R%
#__inference__traced_restore_6618336*
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
: ñ¿
	
À
%__inference_signature_wrapper_6618032
dense_3889_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCalldense_3889_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-6618023*+
f&R$
"__inference__wrapped_model_6617859*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_3889_input: : 
£	
Á
0__inference_sequential_783_layer_call_fn_6618102

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-6618007*T
fORM
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618006*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 

ý
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617948
dense_3889_input-
)dense_3889_statefulpartitionedcall_args_1-
)dense_3889_statefulpartitionedcall_args_2-
)dense_3890_statefulpartitionedcall_args_1-
)dense_3890_statefulpartitionedcall_args_2-
)dense_3891_statefulpartitionedcall_args_1-
)dense_3891_statefulpartitionedcall_args_2
identity¢"dense_3889/StatefulPartitionedCall¢"dense_3890/StatefulPartitionedCall¢"dense_3891/StatefulPartitionedCall
"dense_3889/StatefulPartitionedCallStatefulPartitionedCalldense_3889_input)dense_3889_statefulpartitionedcall_args_1)dense_3889_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617881*P
fKRI
G__inference_dense_3889_layer_call_and_return_conditional_losses_6617875*
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
:ÿÿÿÿÿÿÿÿÿ·
"dense_3890/StatefulPartitionedCallStatefulPartitionedCall+dense_3889/StatefulPartitionedCall:output:0)dense_3890_statefulpartitionedcall_args_1)dense_3890_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617908*P
fKRI
G__inference_dense_3890_layer_call_and_return_conditional_losses_6617902*
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
:ÿÿÿÿÿÿÿÿÿ¶
"dense_3891/StatefulPartitionedCallStatefulPartitionedCall+dense_3890/StatefulPartitionedCall:output:0)dense_3891_statefulpartitionedcall_args_1)dense_3891_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617936*P
fKRI
G__inference_dense_3891_layer_call_and_return_conditional_losses_6617930*
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
IdentityIdentity+dense_3891/StatefulPartitionedCall:output:0#^dense_3889/StatefulPartitionedCall#^dense_3890/StatefulPartitionedCall#^dense_3891/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_3890/StatefulPartitionedCall"dense_3890/StatefulPartitionedCall2H
"dense_3891/StatefulPartitionedCall"dense_3891/StatefulPartitionedCall2H
"dense_3889/StatefulPartitionedCall"dense_3889/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_3889_input: : 
Õ	
à
G__inference_dense_3891_layer_call_and_return_conditional_losses_6617930

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
½
Ü
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618080

inputs-
)dense_3889_matmul_readvariableop_resource.
*dense_3889_biasadd_readvariableop_resource-
)dense_3890_matmul_readvariableop_resource.
*dense_3890_biasadd_readvariableop_resource-
)dense_3891_matmul_readvariableop_resource.
*dense_3891_biasadd_readvariableop_resource
identity¢!dense_3889/BiasAdd/ReadVariableOp¢ dense_3889/MatMul/ReadVariableOp¢!dense_3890/BiasAdd/ReadVariableOp¢ dense_3890/MatMul/ReadVariableOp¢!dense_3891/BiasAdd/ReadVariableOp¢ dense_3891/MatMul/ReadVariableOp¹
 dense_3889/MatMul/ReadVariableOpReadVariableOp)dense_3889_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3889/MatMulMatMulinputs(dense_3889/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3889/BiasAdd/ReadVariableOpReadVariableOp*dense_3889_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3889/BiasAddBiasAdddense_3889/MatMul:product:0)dense_3889/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3890/MatMul/ReadVariableOpReadVariableOp)dense_3890_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3890/MatMulMatMuldense_3889/BiasAdd:output:0(dense_3890/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3890/BiasAdd/ReadVariableOpReadVariableOp*dense_3890_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3890/BiasAddBiasAdddense_3890/MatMul:product:0)dense_3890/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 dense_3891/MatMul/ReadVariableOpReadVariableOp)dense_3891_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3891/MatMulMatMuldense_3890/BiasAdd:output:0(dense_3891/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!dense_3891/BiasAdd/ReadVariableOpReadVariableOp*dense_3891_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_3891/BiasAddBiasAdddense_3891/MatMul:product:0)dense_3891/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3891/ReluReludense_3891/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
IdentityIdentitydense_3891/Relu:activations:0"^dense_3889/BiasAdd/ReadVariableOp!^dense_3889/MatMul/ReadVariableOp"^dense_3890/BiasAdd/ReadVariableOp!^dense_3890/MatMul/ReadVariableOp"^dense_3891/BiasAdd/ReadVariableOp!^dense_3891/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2F
!dense_3890/BiasAdd/ReadVariableOp!dense_3890/BiasAdd/ReadVariableOp2D
 dense_3890/MatMul/ReadVariableOp dense_3890/MatMul/ReadVariableOp2F
!dense_3889/BiasAdd/ReadVariableOp!dense_3889/BiasAdd/ReadVariableOp2D
 dense_3889/MatMul/ReadVariableOp dense_3889/MatMul/ReadVariableOp2D
 dense_3891/MatMul/ReadVariableOp dense_3891/MatMul/ReadVariableOp2F
!dense_3891/BiasAdd/ReadVariableOp!dense_3891/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
â
­
,__inference_dense_3890_layer_call_fn_6618136

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617908*P
fKRI
G__inference_dense_3890_layer_call_and_return_conditional_losses_6617902*
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
à
­
,__inference_dense_3891_layer_call_fn_6618154

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617936*P
fKRI
G__inference_dense_3891_layer_call_and_return_conditional_losses_6617930*
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
Õ#
ñ
"__inference__wrapped_model_6617859
dense_3889_input<
8sequential_783_dense_3889_matmul_readvariableop_resource=
9sequential_783_dense_3889_biasadd_readvariableop_resource<
8sequential_783_dense_3890_matmul_readvariableop_resource=
9sequential_783_dense_3890_biasadd_readvariableop_resource<
8sequential_783_dense_3891_matmul_readvariableop_resource=
9sequential_783_dense_3891_biasadd_readvariableop_resource
identity¢0sequential_783/dense_3889/BiasAdd/ReadVariableOp¢/sequential_783/dense_3889/MatMul/ReadVariableOp¢0sequential_783/dense_3890/BiasAdd/ReadVariableOp¢/sequential_783/dense_3890/MatMul/ReadVariableOp¢0sequential_783/dense_3891/BiasAdd/ReadVariableOp¢/sequential_783/dense_3891/MatMul/ReadVariableOp×
/sequential_783/dense_3889/MatMul/ReadVariableOpReadVariableOp8sequential_783_dense_3889_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	¨
 sequential_783/dense_3889/MatMulMatMuldense_3889_input7sequential_783/dense_3889/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_783/dense_3889/BiasAdd/ReadVariableOpReadVariableOp9sequential_783_dense_3889_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_783/dense_3889/BiasAddBiasAdd*sequential_783/dense_3889/MatMul:product:08sequential_783/dense_3889/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
/sequential_783/dense_3890/MatMul/ReadVariableOpReadVariableOp8sequential_783_dense_3890_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Â
 sequential_783/dense_3890/MatMulMatMul*sequential_783/dense_3889/BiasAdd:output:07sequential_783/dense_3890/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_783/dense_3890/BiasAdd/ReadVariableOpReadVariableOp9sequential_783_dense_3890_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_783/dense_3890/BiasAddBiasAdd*sequential_783/dense_3890/MatMul:product:08sequential_783/dense_3890/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
/sequential_783/dense_3891/MatMul/ReadVariableOpReadVariableOp8sequential_783_dense_3891_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Á
 sequential_783/dense_3891/MatMulMatMul*sequential_783/dense_3890/BiasAdd:output:07sequential_783/dense_3891/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
0sequential_783/dense_3891/BiasAdd/ReadVariableOpReadVariableOp9sequential_783_dense_3891_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ä
!sequential_783/dense_3891/BiasAddBiasAdd*sequential_783/dense_3891/MatMul:product:08sequential_783/dense_3891/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_783/dense_3891/ReluRelu*sequential_783/dense_3891/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
IdentityIdentity,sequential_783/dense_3891/Relu:activations:01^sequential_783/dense_3889/BiasAdd/ReadVariableOp0^sequential_783/dense_3889/MatMul/ReadVariableOp1^sequential_783/dense_3890/BiasAdd/ReadVariableOp0^sequential_783/dense_3890/MatMul/ReadVariableOp1^sequential_783/dense_3891/BiasAdd/ReadVariableOp0^sequential_783/dense_3891/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2b
/sequential_783/dense_3889/MatMul/ReadVariableOp/sequential_783/dense_3889/MatMul/ReadVariableOp2d
0sequential_783/dense_3891/BiasAdd/ReadVariableOp0sequential_783/dense_3891/BiasAdd/ReadVariableOp2d
0sequential_783/dense_3890/BiasAdd/ReadVariableOp0sequential_783/dense_3890/BiasAdd/ReadVariableOp2b
/sequential_783/dense_3891/MatMul/ReadVariableOp/sequential_783/dense_3891/MatMul/ReadVariableOp2d
0sequential_783/dense_3889/BiasAdd/ReadVariableOp0sequential_783/dense_3889/BiasAdd/ReadVariableOp2b
/sequential_783/dense_3890/MatMul/ReadVariableOp/sequential_783/dense_3890/MatMul/ReadVariableOp: : : : :0 ,
*
_user_specified_namedense_3889_input: : 
Á	
Ë
0__inference_sequential_783_layer_call_fn_6618016
dense_3889_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3889_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-6618007*T
fORM
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618006*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_3889_input: : 
ì
ó
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617979

inputs-
)dense_3889_statefulpartitionedcall_args_1-
)dense_3889_statefulpartitionedcall_args_2-
)dense_3890_statefulpartitionedcall_args_1-
)dense_3890_statefulpartitionedcall_args_2-
)dense_3891_statefulpartitionedcall_args_1-
)dense_3891_statefulpartitionedcall_args_2
identity¢"dense_3889/StatefulPartitionedCall¢"dense_3890/StatefulPartitionedCall¢"dense_3891/StatefulPartitionedCall
"dense_3889/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_3889_statefulpartitionedcall_args_1)dense_3889_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617881*P
fKRI
G__inference_dense_3889_layer_call_and_return_conditional_losses_6617875*
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
:ÿÿÿÿÿÿÿÿÿ·
"dense_3890/StatefulPartitionedCallStatefulPartitionedCall+dense_3889/StatefulPartitionedCall:output:0)dense_3890_statefulpartitionedcall_args_1)dense_3890_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617908*P
fKRI
G__inference_dense_3890_layer_call_and_return_conditional_losses_6617902*
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
:ÿÿÿÿÿÿÿÿÿ¶
"dense_3891/StatefulPartitionedCallStatefulPartitionedCall+dense_3890/StatefulPartitionedCall:output:0)dense_3891_statefulpartitionedcall_args_1)dense_3891_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617936*P
fKRI
G__inference_dense_3891_layer_call_and_return_conditional_losses_6617930*
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
IdentityIdentity+dense_3891/StatefulPartitionedCall:output:0#^dense_3889/StatefulPartitionedCall#^dense_3890/StatefulPartitionedCall#^dense_3891/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_3890/StatefulPartitionedCall"dense_3890/StatefulPartitionedCall2H
"dense_3891/StatefulPartitionedCall"dense_3891/StatefulPartitionedCall2H
"dense_3889/StatefulPartitionedCall"dense_3889/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
½
Ü
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618057

inputs-
)dense_3889_matmul_readvariableop_resource.
*dense_3889_biasadd_readvariableop_resource-
)dense_3890_matmul_readvariableop_resource.
*dense_3890_biasadd_readvariableop_resource-
)dense_3891_matmul_readvariableop_resource.
*dense_3891_biasadd_readvariableop_resource
identity¢!dense_3889/BiasAdd/ReadVariableOp¢ dense_3889/MatMul/ReadVariableOp¢!dense_3890/BiasAdd/ReadVariableOp¢ dense_3890/MatMul/ReadVariableOp¢!dense_3891/BiasAdd/ReadVariableOp¢ dense_3891/MatMul/ReadVariableOp¹
 dense_3889/MatMul/ReadVariableOpReadVariableOp)dense_3889_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3889/MatMulMatMulinputs(dense_3889/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3889/BiasAdd/ReadVariableOpReadVariableOp*dense_3889_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3889/BiasAddBiasAdddense_3889/MatMul:product:0)dense_3889/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3890/MatMul/ReadVariableOpReadVariableOp)dense_3890_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3890/MatMulMatMuldense_3889/BiasAdd:output:0(dense_3890/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3890/BiasAdd/ReadVariableOpReadVariableOp*dense_3890_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3890/BiasAddBiasAdddense_3890/MatMul:product:0)dense_3890/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 dense_3891/MatMul/ReadVariableOpReadVariableOp)dense_3891_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3891/MatMulMatMuldense_3890/BiasAdd:output:0(dense_3891/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!dense_3891/BiasAdd/ReadVariableOpReadVariableOp*dense_3891_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_3891/BiasAddBiasAdddense_3891/MatMul:product:0)dense_3891/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3891/ReluReludense_3891/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
IdentityIdentitydense_3891/Relu:activations:0"^dense_3889/BiasAdd/ReadVariableOp!^dense_3889/MatMul/ReadVariableOp"^dense_3890/BiasAdd/ReadVariableOp!^dense_3890/MatMul/ReadVariableOp"^dense_3891/BiasAdd/ReadVariableOp!^dense_3891/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2F
!dense_3890/BiasAdd/ReadVariableOp!dense_3890/BiasAdd/ReadVariableOp2F
!dense_3889/BiasAdd/ReadVariableOp!dense_3889/BiasAdd/ReadVariableOp2D
 dense_3890/MatMul/ReadVariableOp dense_3890/MatMul/ReadVariableOp2D
 dense_3889/MatMul/ReadVariableOp dense_3889/MatMul/ReadVariableOp2D
 dense_3891/MatMul/ReadVariableOp dense_3891/MatMul/ReadVariableOp2F
!dense_3891/BiasAdd/ReadVariableOp!dense_3891/BiasAdd/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs: : 
ì
ó
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618006

inputs-
)dense_3889_statefulpartitionedcall_args_1-
)dense_3889_statefulpartitionedcall_args_2-
)dense_3890_statefulpartitionedcall_args_1-
)dense_3890_statefulpartitionedcall_args_2-
)dense_3891_statefulpartitionedcall_args_1-
)dense_3891_statefulpartitionedcall_args_2
identity¢"dense_3889/StatefulPartitionedCall¢"dense_3890/StatefulPartitionedCall¢"dense_3891/StatefulPartitionedCall
"dense_3889/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_3889_statefulpartitionedcall_args_1)dense_3889_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617881*P
fKRI
G__inference_dense_3889_layer_call_and_return_conditional_losses_6617875*
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
:ÿÿÿÿÿÿÿÿÿ·
"dense_3890/StatefulPartitionedCallStatefulPartitionedCall+dense_3889/StatefulPartitionedCall:output:0)dense_3890_statefulpartitionedcall_args_1)dense_3890_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617908*P
fKRI
G__inference_dense_3890_layer_call_and_return_conditional_losses_6617902*
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
:ÿÿÿÿÿÿÿÿÿ¶
"dense_3891/StatefulPartitionedCallStatefulPartitionedCall+dense_3890/StatefulPartitionedCall:output:0)dense_3891_statefulpartitionedcall_args_1)dense_3891_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617936*P
fKRI
G__inference_dense_3891_layer_call_and_return_conditional_losses_6617930*
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
IdentityIdentity+dense_3891/StatefulPartitionedCall:output:0#^dense_3889/StatefulPartitionedCall#^dense_3890/StatefulPartitionedCall#^dense_3891/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_3890/StatefulPartitionedCall"dense_3890/StatefulPartitionedCall2H
"dense_3891/StatefulPartitionedCall"dense_3891/StatefulPartitionedCall2H
"dense_3889/StatefulPartitionedCall"dense_3889/StatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 
	
à
G__inference_dense_3890_layer_call_and_return_conditional_losses_6617902

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
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
	
à
G__inference_dense_3890_layer_call_and_return_conditional_losses_6618129

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
Á	
Ë
0__inference_sequential_783_layer_call_fn_6617989
dense_3889_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3889_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-6617980*T
fORM
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617979*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_3889_input: : 
	
à
G__inference_dense_3889_layer_call_and_return_conditional_losses_6618112

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
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
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
£	
Á
0__inference_sequential_783_layer_call_fn_6618091

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6*.
_gradient_op_typePartitionedCall-6617980*T
fORM
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617979*
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
+:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs: : 

ý
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617963
dense_3889_input-
)dense_3889_statefulpartitionedcall_args_1-
)dense_3889_statefulpartitionedcall_args_2-
)dense_3890_statefulpartitionedcall_args_1-
)dense_3890_statefulpartitionedcall_args_2-
)dense_3891_statefulpartitionedcall_args_1-
)dense_3891_statefulpartitionedcall_args_2
identity¢"dense_3889/StatefulPartitionedCall¢"dense_3890/StatefulPartitionedCall¢"dense_3891/StatefulPartitionedCall
"dense_3889/StatefulPartitionedCallStatefulPartitionedCalldense_3889_input)dense_3889_statefulpartitionedcall_args_1)dense_3889_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617881*P
fKRI
G__inference_dense_3889_layer_call_and_return_conditional_losses_6617875*
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
:ÿÿÿÿÿÿÿÿÿ·
"dense_3890/StatefulPartitionedCallStatefulPartitionedCall+dense_3889/StatefulPartitionedCall:output:0)dense_3890_statefulpartitionedcall_args_1)dense_3890_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617908*P
fKRI
G__inference_dense_3890_layer_call_and_return_conditional_losses_6617902*
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
:ÿÿÿÿÿÿÿÿÿ¶
"dense_3891/StatefulPartitionedCallStatefulPartitionedCall+dense_3890/StatefulPartitionedCall:output:0)dense_3891_statefulpartitionedcall_args_1)dense_3891_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617936*P
fKRI
G__inference_dense_3891_layer_call_and_return_conditional_losses_6617930*
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
IdentityIdentity+dense_3891/StatefulPartitionedCall:output:0#^dense_3889/StatefulPartitionedCall#^dense_3890/StatefulPartitionedCall#^dense_3891/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ::::::2H
"dense_3890/StatefulPartitionedCall"dense_3890/StatefulPartitionedCall2H
"dense_3891/StatefulPartitionedCall"dense_3891/StatefulPartitionedCall2H
"dense_3889/StatefulPartitionedCall"dense_3889/StatefulPartitionedCall: : : : :0 ,
*
_user_specified_namedense_3889_input: : 
á
­
,__inference_dense_3889_layer_call_fn_6618119

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6617881*P
fKRI
G__inference_dense_3889_layer_call_and_return_conditional_losses_6617875*
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
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Õ	
à
G__inference_dense_3891_layer_call_and_return_conditional_losses_6618147

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
ôb

#__inference__traced_restore_6618336
file_prefix&
"assignvariableop_dense_3889_kernel&
"assignvariableop_1_dense_3889_bias(
$assignvariableop_2_dense_3890_kernel&
"assignvariableop_3_dense_3890_bias(
$assignvariableop_4_dense_3891_kernel&
"assignvariableop_5_dense_3891_bias$
 assignvariableop_6_adadelta_iter%
!assignvariableop_7_adadelta_decay-
)assignvariableop_8_adadelta_learning_rate#
assignvariableop_9_adadelta_rho
assignvariableop_10_total
assignvariableop_11_count=
9assignvariableop_12_adadelta_dense_3889_kernel_accum_grad;
7assignvariableop_13_adadelta_dense_3889_bias_accum_grad=
9assignvariableop_14_adadelta_dense_3890_kernel_accum_grad;
7assignvariableop_15_adadelta_dense_3890_bias_accum_grad=
9assignvariableop_16_adadelta_dense_3891_kernel_accum_grad;
7assignvariableop_17_adadelta_dense_3891_bias_accum_grad<
8assignvariableop_18_adadelta_dense_3889_kernel_accum_var:
6assignvariableop_19_adadelta_dense_3889_bias_accum_var<
8assignvariableop_20_adadelta_dense_3890_kernel_accum_var:
6assignvariableop_21_adadelta_dense_3890_bias_accum_var<
8assignvariableop_22_adadelta_dense_3891_kernel_accum_var:
6assignvariableop_23_adadelta_dense_3891_bias_accum_var
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
AssignVariableOpAssignVariableOp"assignvariableop_dense_3889_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_3889_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_3890_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_3890_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_3891_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_3891_biasIdentity_5:output:0*
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
AssignVariableOp_12AssignVariableOp9assignvariableop_12_adadelta_dense_3889_kernel_accum_gradIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp7assignvariableop_13_adadelta_dense_3889_bias_accum_gradIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp9assignvariableop_14_adadelta_dense_3890_kernel_accum_gradIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp7assignvariableop_15_adadelta_dense_3890_bias_accum_gradIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp9assignvariableop_16_adadelta_dense_3891_kernel_accum_gradIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp7assignvariableop_17_adadelta_dense_3891_bias_accum_gradIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp8assignvariableop_18_adadelta_dense_3889_kernel_accum_varIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adadelta_dense_3889_bias_accum_varIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adadelta_dense_3890_kernel_accum_varIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adadelta_dense_3890_bias_accum_varIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp8assignvariableop_22_adadelta_dense_3891_kernel_accum_varIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adadelta_dense_3891_bias_accum_varIdentity_23:output:0*
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
Ç9
ì
 __inference__traced_save_6618251
file_prefix0
,savev2_dense_3889_kernel_read_readvariableop.
*savev2_dense_3889_bias_read_readvariableop0
,savev2_dense_3890_kernel_read_readvariableop.
*savev2_dense_3890_bias_read_readvariableop0
,savev2_dense_3891_kernel_read_readvariableop.
*savev2_dense_3891_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopD
@savev2_adadelta_dense_3889_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_3889_bias_accum_grad_read_readvariableopD
@savev2_adadelta_dense_3890_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_3890_bias_accum_grad_read_readvariableopD
@savev2_adadelta_dense_3891_kernel_accum_grad_read_readvariableopB
>savev2_adadelta_dense_3891_bias_accum_grad_read_readvariableopC
?savev2_adadelta_dense_3889_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_3889_bias_accum_var_read_readvariableopC
?savev2_adadelta_dense_3890_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_3890_bias_accum_var_read_readvariableopC
?savev2_adadelta_dense_3891_kernel_accum_var_read_readvariableopA
=savev2_adadelta_dense_3891_bias_accum_var_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_6e3bcae8ce3249fc912d65db0968b90b/part*
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_3889_kernel_read_readvariableop*savev2_dense_3889_bias_read_readvariableop,savev2_dense_3890_kernel_read_readvariableop*savev2_dense_3890_bias_read_readvariableop,savev2_dense_3891_kernel_read_readvariableop*savev2_dense_3891_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop@savev2_adadelta_dense_3889_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_3889_bias_accum_grad_read_readvariableop@savev2_adadelta_dense_3890_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_3890_bias_accum_grad_read_readvariableop@savev2_adadelta_dense_3891_kernel_accum_grad_read_readvariableop>savev2_adadelta_dense_3891_bias_accum_grad_read_readvariableop?savev2_adadelta_dense_3889_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_3889_bias_accum_var_read_readvariableop?savev2_adadelta_dense_3890_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_3890_bias_accum_var_read_readvariableop?savev2_adadelta_dense_3891_kernel_accum_var_read_readvariableop=savev2_adadelta_dense_3891_bias_accum_var_read_readvariableop"/device:CPU:0*&
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
²: :	::
::	:: : : : : : :	::
::	::	::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : : : : : : : : :
 
	
à
G__inference_dense_3889_layer_call_and_return_conditional_losses_6617875

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
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
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*¿
serving_default«
M
dense_3889_input9
"serving_default_dense_3889_input:0ÿÿÿÿÿÿÿÿÿ>

dense_38910
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:é

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
*S&call_and_return_all_conditional_losses"¾
_tf_keras_sequential{"class_name": "Sequential", "name": "sequential_783", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_783", "layers": [{"class_name": "Dense", "config": {"name": "dense_3889", "trainable": true, "batch_input_shape": [null, 3], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3890", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3891", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_783", "layers": [{"class_name": "Dense", "config": {"name": "dense_3889", "trainable": true, "batch_input_shape": [null, 3], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3890", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3891", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adadelta", "config": {"name": "Adadelta", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.949999988079071, "epsilon": 1e-07}}}}
³
regularization_losses
	variables
trainable_variables
	keras_api
T__call__
*U&call_and_return_all_conditional_losses"¤
_tf_keras_layer{"class_name": "InputLayer", "name": "dense_3889_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 3], "config": {"batch_input_shape": [null, 3], "dtype": "float32", "sparse": false, "name": "dense_3889_input"}}
º

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
V__call__
*W&call_and_return_all_conditional_losses"
_tf_keras_layerû{"class_name": "Dense", "name": "dense_3889", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 3], "config": {"name": "dense_3889", "trainable": true, "batch_input_shape": [null, 3], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "Dense", "name": "dense_3890", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3890", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
 	keras_api
Z__call__
*[&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Dense", "name": "dense_3891", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3891", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
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
$:"	2dense_3889/kernel
:2dense_3889/bias
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
%:#
2dense_3890/kernel
:2dense_3890/bias
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
$:"	2dense_3891/kernel
:2dense_3891/bias
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
6:4	2%Adadelta/dense_3889/kernel/accum_grad
0:.2#Adadelta/dense_3889/bias/accum_grad
7:5
2%Adadelta/dense_3890/kernel/accum_grad
0:.2#Adadelta/dense_3890/bias/accum_grad
6:4	2%Adadelta/dense_3891/kernel/accum_grad
/:-2#Adadelta/dense_3891/bias/accum_grad
5:3	2$Adadelta/dense_3889/kernel/accum_var
/:-2"Adadelta/dense_3889/bias/accum_var
6:4
2$Adadelta/dense_3890/kernel/accum_var
/:-2"Adadelta/dense_3890/bias/accum_var
5:3	2$Adadelta/dense_3891/kernel/accum_var
.:,2"Adadelta/dense_3891/bias/accum_var
2
0__inference_sequential_783_layer_call_fn_6618016
0__inference_sequential_783_layer_call_fn_6618091
0__inference_sequential_783_layer_call_fn_6617989
0__inference_sequential_783_layer_call_fn_6618102À
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
é2æ
"__inference__wrapped_model_6617859¿
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
dense_3889_inputÿÿÿÿÿÿÿÿÿ
ú2÷
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618057
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617948
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618080
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617963À
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
Ö2Ó
,__inference_dense_3889_layer_call_fn_6618119¢
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
ñ2î
G__inference_dense_3889_layer_call_and_return_conditional_losses_6618112¢
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
Ö2Ó
,__inference_dense_3890_layer_call_fn_6618136¢
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
ñ2î
G__inference_dense_3890_layer_call_and_return_conditional_losses_6618129¢
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
Ö2Ó
,__inference_dense_3891_layer_call_fn_6618154¢
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
ñ2î
G__inference_dense_3891_layer_call_and_return_conditional_losses_6618147¢
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
=B;
%__inference_signature_wrapper_6618032dense_3889_input
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
 
,__inference_dense_3890_layer_call_fn_6618136Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_783_layer_call_fn_6617989eA¢>
7¢4
*'
dense_3889_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¢
"__inference__wrapped_model_6617859|9¢6
/¢,
*'
dense_3889_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

dense_3891$!

dense_3891ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_783_layer_call_fn_6618102[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿº
%__inference_signature_wrapper_6618032M¢J
¢ 
Cª@
>
dense_3889_input*'
dense_3889_inputÿÿÿÿÿÿÿÿÿ"7ª4
2

dense_3891$!

dense_3891ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_3890_layer_call_and_return_conditional_losses_6618129^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_3891_layer_call_fn_6618154P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ·
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618057h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617948rA¢>
7¢4
*'
dense_3889_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_sequential_783_layer_call_fn_6618016eA¢>
7¢4
*'
dense_3889_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_dense_3889_layer_call_and_return_conditional_losses_6618112]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_783_layer_call_and_return_conditional_losses_6617963rA¢>
7¢4
*'
dense_3889_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_sequential_783_layer_call_fn_6618091[7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_dense_3891_layer_call_and_return_conditional_losses_6618147]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_3889_layer_call_fn_6618119P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ·
K__inference_sequential_783_layer_call_and_return_conditional_losses_6618080h7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 