ê

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
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8Öã

dense_11681/kernelVarHandleOp*
shape:	*#
shared_namedense_11681/kernel*
dtype0*
_output_shapes
: 
z
&dense_11681/kernel/Read/ReadVariableOpReadVariableOpdense_11681/kernel*
dtype0*
_output_shapes
:	
y
dense_11681/biasVarHandleOp*
shape:*!
shared_namedense_11681/bias*
dtype0*
_output_shapes
: 
r
$dense_11681/bias/Read/ReadVariableOpReadVariableOpdense_11681/bias*
dtype0*
_output_shapes	
:

dense_11682/kernelVarHandleOp*
shape:
*#
shared_namedense_11682/kernel*
dtype0*
_output_shapes
: 
{
&dense_11682/kernel/Read/ReadVariableOpReadVariableOpdense_11682/kernel*
dtype0* 
_output_shapes
:

y
dense_11682/biasVarHandleOp*
shape:*!
shared_namedense_11682/bias*
dtype0*
_output_shapes
: 
r
$dense_11682/bias/Read/ReadVariableOpReadVariableOpdense_11682/bias*
dtype0*
_output_shapes	
:

dense_11683/kernelVarHandleOp*
shape:
*#
shared_namedense_11683/kernel*
dtype0*
_output_shapes
: 
{
&dense_11683/kernel/Read/ReadVariableOpReadVariableOpdense_11683/kernel*
dtype0* 
_output_shapes
:

y
dense_11683/biasVarHandleOp*
shape:*!
shared_namedense_11683/bias*
dtype0*
_output_shapes
: 
r
$dense_11683/bias/Read/ReadVariableOpReadVariableOpdense_11683/bias*
dtype0*
_output_shapes	
:

dense_11684/kernelVarHandleOp*
shape:
*#
shared_namedense_11684/kernel*
dtype0*
_output_shapes
: 
{
&dense_11684/kernel/Read/ReadVariableOpReadVariableOpdense_11684/kernel*
dtype0* 
_output_shapes
:

y
dense_11684/biasVarHandleOp*
shape:*!
shared_namedense_11684/bias*
dtype0*
_output_shapes
: 
r
$dense_11684/bias/Read/ReadVariableOpReadVariableOpdense_11684/bias*
dtype0*
_output_shapes	
:

dense_11685/kernelVarHandleOp*
shape:
*#
shared_namedense_11685/kernel*
dtype0*
_output_shapes
: 
{
&dense_11685/kernel/Read/ReadVariableOpReadVariableOpdense_11685/kernel*
dtype0* 
_output_shapes
:

y
dense_11685/biasVarHandleOp*
shape:*!
shared_namedense_11685/bias*
dtype0*
_output_shapes
: 
r
$dense_11685/bias/Read/ReadVariableOpReadVariableOpdense_11685/bias*
dtype0*
_output_shapes	
:

dense_11686/kernelVarHandleOp*
shape:
*#
shared_namedense_11686/kernel*
dtype0*
_output_shapes
: 
{
&dense_11686/kernel/Read/ReadVariableOpReadVariableOpdense_11686/kernel*
dtype0* 
_output_shapes
:

y
dense_11686/biasVarHandleOp*
shape:*!
shared_namedense_11686/bias*
dtype0*
_output_shapes
: 
r
$dense_11686/bias/Read/ReadVariableOpReadVariableOpdense_11686/bias*
dtype0*
_output_shapes	
:

dense_11687/kernelVarHandleOp*
shape:	*#
shared_namedense_11687/kernel*
dtype0*
_output_shapes
: 
z
&dense_11687/kernel/Read/ReadVariableOpReadVariableOpdense_11687/kernel*
dtype0*
_output_shapes
:	
x
dense_11687/biasVarHandleOp*
shape:*!
shared_namedense_11687/bias*
dtype0*
_output_shapes
: 
q
$dense_11687/bias/Read/ReadVariableOpReadVariableOpdense_11687/bias*
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

Nadam/dense_11681/kernel/mVarHandleOp*
shape:	*+
shared_nameNadam/dense_11681/kernel/m*
dtype0*
_output_shapes
: 

.Nadam/dense_11681/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11681/kernel/m*
dtype0*
_output_shapes
:	

Nadam/dense_11681/bias/mVarHandleOp*
shape:*)
shared_nameNadam/dense_11681/bias/m*
dtype0*
_output_shapes
: 

,Nadam/dense_11681/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11681/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_11682/kernel/mVarHandleOp*
shape:
*+
shared_nameNadam/dense_11682/kernel/m*
dtype0*
_output_shapes
: 

.Nadam/dense_11682/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11682/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_11682/bias/mVarHandleOp*
shape:*)
shared_nameNadam/dense_11682/bias/m*
dtype0*
_output_shapes
: 

,Nadam/dense_11682/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11682/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_11683/kernel/mVarHandleOp*
shape:
*+
shared_nameNadam/dense_11683/kernel/m*
dtype0*
_output_shapes
: 

.Nadam/dense_11683/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11683/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_11683/bias/mVarHandleOp*
shape:*)
shared_nameNadam/dense_11683/bias/m*
dtype0*
_output_shapes
: 

,Nadam/dense_11683/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11683/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_11684/kernel/mVarHandleOp*
shape:
*+
shared_nameNadam/dense_11684/kernel/m*
dtype0*
_output_shapes
: 

.Nadam/dense_11684/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11684/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_11684/bias/mVarHandleOp*
shape:*)
shared_nameNadam/dense_11684/bias/m*
dtype0*
_output_shapes
: 

,Nadam/dense_11684/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11684/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_11685/kernel/mVarHandleOp*
shape:
*+
shared_nameNadam/dense_11685/kernel/m*
dtype0*
_output_shapes
: 

.Nadam/dense_11685/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11685/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_11685/bias/mVarHandleOp*
shape:*)
shared_nameNadam/dense_11685/bias/m*
dtype0*
_output_shapes
: 

,Nadam/dense_11685/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11685/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_11686/kernel/mVarHandleOp*
shape:
*+
shared_nameNadam/dense_11686/kernel/m*
dtype0*
_output_shapes
: 

.Nadam/dense_11686/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11686/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_11686/bias/mVarHandleOp*
shape:*)
shared_nameNadam/dense_11686/bias/m*
dtype0*
_output_shapes
: 

,Nadam/dense_11686/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11686/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_11687/kernel/mVarHandleOp*
shape:	*+
shared_nameNadam/dense_11687/kernel/m*
dtype0*
_output_shapes
: 

.Nadam/dense_11687/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_11687/kernel/m*
dtype0*
_output_shapes
:	

Nadam/dense_11687/bias/mVarHandleOp*
shape:*)
shared_nameNadam/dense_11687/bias/m*
dtype0*
_output_shapes
: 

,Nadam/dense_11687/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_11687/bias/m*
dtype0*
_output_shapes
:

Nadam/dense_11681/kernel/vVarHandleOp*
shape:	*+
shared_nameNadam/dense_11681/kernel/v*
dtype0*
_output_shapes
: 

.Nadam/dense_11681/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11681/kernel/v*
dtype0*
_output_shapes
:	

Nadam/dense_11681/bias/vVarHandleOp*
shape:*)
shared_nameNadam/dense_11681/bias/v*
dtype0*
_output_shapes
: 

,Nadam/dense_11681/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11681/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_11682/kernel/vVarHandleOp*
shape:
*+
shared_nameNadam/dense_11682/kernel/v*
dtype0*
_output_shapes
: 

.Nadam/dense_11682/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11682/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_11682/bias/vVarHandleOp*
shape:*)
shared_nameNadam/dense_11682/bias/v*
dtype0*
_output_shapes
: 

,Nadam/dense_11682/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11682/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_11683/kernel/vVarHandleOp*
shape:
*+
shared_nameNadam/dense_11683/kernel/v*
dtype0*
_output_shapes
: 

.Nadam/dense_11683/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11683/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_11683/bias/vVarHandleOp*
shape:*)
shared_nameNadam/dense_11683/bias/v*
dtype0*
_output_shapes
: 

,Nadam/dense_11683/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11683/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_11684/kernel/vVarHandleOp*
shape:
*+
shared_nameNadam/dense_11684/kernel/v*
dtype0*
_output_shapes
: 

.Nadam/dense_11684/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11684/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_11684/bias/vVarHandleOp*
shape:*)
shared_nameNadam/dense_11684/bias/v*
dtype0*
_output_shapes
: 

,Nadam/dense_11684/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11684/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_11685/kernel/vVarHandleOp*
shape:
*+
shared_nameNadam/dense_11685/kernel/v*
dtype0*
_output_shapes
: 

.Nadam/dense_11685/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11685/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_11685/bias/vVarHandleOp*
shape:*)
shared_nameNadam/dense_11685/bias/v*
dtype0*
_output_shapes
: 

,Nadam/dense_11685/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11685/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_11686/kernel/vVarHandleOp*
shape:
*+
shared_nameNadam/dense_11686/kernel/v*
dtype0*
_output_shapes
: 

.Nadam/dense_11686/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11686/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_11686/bias/vVarHandleOp*
shape:*)
shared_nameNadam/dense_11686/bias/v*
dtype0*
_output_shapes
: 

,Nadam/dense_11686/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11686/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_11687/kernel/vVarHandleOp*
shape:	*+
shared_nameNadam/dense_11687/kernel/v*
dtype0*
_output_shapes
: 

.Nadam/dense_11687/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_11687/kernel/v*
dtype0*
_output_shapes
:	

Nadam/dense_11687/bias/vVarHandleOp*
shape:*)
shared_nameNadam/dense_11687/bias/v*
dtype0*
_output_shapes
: 

,Nadam/dense_11687/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_11687/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
£K
ConstConst"/device:CPU:0*ÞJ
valueÔJBÑJ BÊJ

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
ß
=iter

>beta_1

?beta_2
	@decay
Alearning_rate
Bmomentum_cachemsmtmumvmw mx%my&mz+m{,m|1m}2m~7m8mvvvvv v%v&v+v,v1v2v7v8v
 
f
0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813
f
0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813


regularization_losses

Clayers
	variables
Dmetrics
Elayer_regularization_losses
trainable_variables
Fnon_trainable_variables
 
 
 
 

regularization_losses

Glayers
	variables
Hmetrics
Ilayer_regularization_losses
trainable_variables
Jnon_trainable_variables
^\
VARIABLE_VALUEdense_11681/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_11681/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses

Klayers
	variables
Lmetrics
Mlayer_regularization_losses
trainable_variables
Nnon_trainable_variables
^\
VARIABLE_VALUEdense_11682/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_11682/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses

Olayers
	variables
Pmetrics
Qlayer_regularization_losses
trainable_variables
Rnon_trainable_variables
^\
VARIABLE_VALUEdense_11683/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_11683/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1

!regularization_losses

Slayers
"	variables
Tmetrics
Ulayer_regularization_losses
#trainable_variables
Vnon_trainable_variables
^\
VARIABLE_VALUEdense_11684/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_11684/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1

'regularization_losses

Wlayers
(	variables
Xmetrics
Ylayer_regularization_losses
)trainable_variables
Znon_trainable_variables
^\
VARIABLE_VALUEdense_11685/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_11685/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1

-regularization_losses

[layers
.	variables
\metrics
]layer_regularization_losses
/trainable_variables
^non_trainable_variables
^\
VARIABLE_VALUEdense_11686/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_11686/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21

3regularization_losses

_layers
4	variables
`metrics
alayer_regularization_losses
5trainable_variables
bnon_trainable_variables
^\
VARIABLE_VALUEdense_11687/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEdense_11687/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81

9regularization_losses

clayers
:	variables
dmetrics
elayer_regularization_losses
;trainable_variables
fnon_trainable_variables
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
1
0
1
2
3
4
5
6

g0
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
	htotal
	icount
j
_fn_kwargs
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

h0
i1
 

kregularization_losses

olayers
l	variables
pmetrics
qlayer_regularization_losses
mtrainable_variables
rnon_trainable_variables
 
 
 

h0
i1

VARIABLE_VALUENadam/dense_11681/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11681/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11682/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11682/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11683/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11683/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11684/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11684/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11685/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11685/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11686/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11686/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11687/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11687/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11681/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11681/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11682/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11682/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11683/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11683/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11684/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11684/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11685/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11685/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11686/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11686/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_11687/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUENadam/dense_11687/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

!serving_default_dense_11681_inputPlaceholder*
shape:ÿÿÿÿÿÿÿÿÿ*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCall!serving_default_dense_11681_inputdense_11681/kerneldense_11681/biasdense_11682/kerneldense_11682/biasdense_11683/kerneldense_11683/biasdense_11684/kerneldense_11684/biasdense_11685/kerneldense_11685/biasdense_11686/kerneldense_11686/biasdense_11687/kerneldense_11687/bias*/
_gradient_op_typePartitionedCall-19860363*/
f*R(
&__inference_signature_wrapper_19860005*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&dense_11681/kernel/Read/ReadVariableOp$dense_11681/bias/Read/ReadVariableOp&dense_11682/kernel/Read/ReadVariableOp$dense_11682/bias/Read/ReadVariableOp&dense_11683/kernel/Read/ReadVariableOp$dense_11683/bias/Read/ReadVariableOp&dense_11684/kernel/Read/ReadVariableOp$dense_11684/bias/Read/ReadVariableOp&dense_11685/kernel/Read/ReadVariableOp$dense_11685/bias/Read/ReadVariableOp&dense_11686/kernel/Read/ReadVariableOp$dense_11686/bias/Read/ReadVariableOp&dense_11687/kernel/Read/ReadVariableOp$dense_11687/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Nadam/dense_11681/kernel/m/Read/ReadVariableOp,Nadam/dense_11681/bias/m/Read/ReadVariableOp.Nadam/dense_11682/kernel/m/Read/ReadVariableOp,Nadam/dense_11682/bias/m/Read/ReadVariableOp.Nadam/dense_11683/kernel/m/Read/ReadVariableOp,Nadam/dense_11683/bias/m/Read/ReadVariableOp.Nadam/dense_11684/kernel/m/Read/ReadVariableOp,Nadam/dense_11684/bias/m/Read/ReadVariableOp.Nadam/dense_11685/kernel/m/Read/ReadVariableOp,Nadam/dense_11685/bias/m/Read/ReadVariableOp.Nadam/dense_11686/kernel/m/Read/ReadVariableOp,Nadam/dense_11686/bias/m/Read/ReadVariableOp.Nadam/dense_11687/kernel/m/Read/ReadVariableOp,Nadam/dense_11687/bias/m/Read/ReadVariableOp.Nadam/dense_11681/kernel/v/Read/ReadVariableOp,Nadam/dense_11681/bias/v/Read/ReadVariableOp.Nadam/dense_11682/kernel/v/Read/ReadVariableOp,Nadam/dense_11682/bias/v/Read/ReadVariableOp.Nadam/dense_11683/kernel/v/Read/ReadVariableOp,Nadam/dense_11683/bias/v/Read/ReadVariableOp.Nadam/dense_11684/kernel/v/Read/ReadVariableOp,Nadam/dense_11684/bias/v/Read/ReadVariableOp.Nadam/dense_11685/kernel/v/Read/ReadVariableOp,Nadam/dense_11685/bias/v/Read/ReadVariableOp.Nadam/dense_11686/kernel/v/Read/ReadVariableOp,Nadam/dense_11686/bias/v/Read/ReadVariableOp.Nadam/dense_11687/kernel/v/Read/ReadVariableOp,Nadam/dense_11687/bias/v/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-19860435**
f%R#
!__inference__traced_save_19860434*
Tout
2**
config_proto

CPU

GPU 2J 8*?
Tin8
624	*
_output_shapes
: 
£
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11681/kerneldense_11681/biasdense_11682/kerneldense_11682/biasdense_11683/kerneldense_11683/biasdense_11684/kerneldense_11684/biasdense_11685/kerneldense_11685/biasdense_11686/kerneldense_11686/biasdense_11687/kerneldense_11687/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/dense_11681/kernel/mNadam/dense_11681/bias/mNadam/dense_11682/kernel/mNadam/dense_11682/bias/mNadam/dense_11683/kernel/mNadam/dense_11683/bias/mNadam/dense_11684/kernel/mNadam/dense_11684/bias/mNadam/dense_11685/kernel/mNadam/dense_11685/bias/mNadam/dense_11686/kernel/mNadam/dense_11686/bias/mNadam/dense_11687/kernel/mNadam/dense_11687/bias/mNadam/dense_11681/kernel/vNadam/dense_11681/bias/vNadam/dense_11682/kernel/vNadam/dense_11682/bias/vNadam/dense_11683/kernel/vNadam/dense_11683/bias/vNadam/dense_11684/kernel/vNadam/dense_11684/bias/vNadam/dense_11685/kernel/vNadam/dense_11685/bias/vNadam/dense_11686/kernel/vNadam/dense_11686/bias/vNadam/dense_11687/kernel/vNadam/dense_11687/bias/v*/
_gradient_op_typePartitionedCall-19860598*-
f(R&
$__inference__traced_restore_19860597*
Tout
2**
config_proto

CPU

GPU 2J 8*>
Tin7
523*
_output_shapes
: Æ
ç
¯
.__inference_dense_11686_layer_call_fn_19860241

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859820*R
fMRK
I__inference_dense_11686_layer_call_and_return_conditional_losses_19859814*
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
	
â
I__inference_dense_11683_layer_call_and_return_conditional_losses_19859733

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
ª
è
2__inference_sequential_2351_layer_call_fn_19860120

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*/
_gradient_op_typePartitionedCall-19859916*V
fQRO
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859915*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
ç
¯
.__inference_dense_11684_layer_call_fn_19860207

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859766*R
fMRK
I__inference_dense_11684_layer_call_and_return_conditional_losses_19859760*
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
	
â
I__inference_dense_11685_layer_call_and_return_conditional_losses_19859787

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
æ
¯
.__inference_dense_11681_layer_call_fn_19860156

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859685*R
fMRK
I__inference_dense_11681_layer_call_and_return_conditional_losses_19859679*
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
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
	
â
I__inference_dense_11681_layer_call_and_return_conditional_losses_19860149

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
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
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
	
â
I__inference_dense_11682_layer_call_and_return_conditional_losses_19860166

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
Q
³
#__inference__wrapped_model_19859663
dense_11681_input>
:sequential_2351_dense_11681_matmul_readvariableop_resource?
;sequential_2351_dense_11681_biasadd_readvariableop_resource>
:sequential_2351_dense_11682_matmul_readvariableop_resource?
;sequential_2351_dense_11682_biasadd_readvariableop_resource>
:sequential_2351_dense_11683_matmul_readvariableop_resource?
;sequential_2351_dense_11683_biasadd_readvariableop_resource>
:sequential_2351_dense_11684_matmul_readvariableop_resource?
;sequential_2351_dense_11684_biasadd_readvariableop_resource>
:sequential_2351_dense_11685_matmul_readvariableop_resource?
;sequential_2351_dense_11685_biasadd_readvariableop_resource>
:sequential_2351_dense_11686_matmul_readvariableop_resource?
;sequential_2351_dense_11686_biasadd_readvariableop_resource>
:sequential_2351_dense_11687_matmul_readvariableop_resource?
;sequential_2351_dense_11687_biasadd_readvariableop_resource
identity¢2sequential_2351/dense_11681/BiasAdd/ReadVariableOp¢1sequential_2351/dense_11681/MatMul/ReadVariableOp¢2sequential_2351/dense_11682/BiasAdd/ReadVariableOp¢1sequential_2351/dense_11682/MatMul/ReadVariableOp¢2sequential_2351/dense_11683/BiasAdd/ReadVariableOp¢1sequential_2351/dense_11683/MatMul/ReadVariableOp¢2sequential_2351/dense_11684/BiasAdd/ReadVariableOp¢1sequential_2351/dense_11684/MatMul/ReadVariableOp¢2sequential_2351/dense_11685/BiasAdd/ReadVariableOp¢1sequential_2351/dense_11685/MatMul/ReadVariableOp¢2sequential_2351/dense_11686/BiasAdd/ReadVariableOp¢1sequential_2351/dense_11686/MatMul/ReadVariableOp¢2sequential_2351/dense_11687/BiasAdd/ReadVariableOp¢1sequential_2351/dense_11687/MatMul/ReadVariableOpÛ
1sequential_2351/dense_11681/MatMul/ReadVariableOpReadVariableOp:sequential_2351_dense_11681_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	­
"sequential_2351/dense_11681/MatMulMatMuldense_11681_input9sequential_2351/dense_11681/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2351/dense_11681/BiasAdd/ReadVariableOpReadVariableOp;sequential_2351_dense_11681_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2351/dense_11681/BiasAddBiasAdd,sequential_2351/dense_11681/MatMul:product:0:sequential_2351/dense_11681/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
1sequential_2351/dense_11682/MatMul/ReadVariableOpReadVariableOp:sequential_2351_dense_11682_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
È
"sequential_2351/dense_11682/MatMulMatMul,sequential_2351/dense_11681/BiasAdd:output:09sequential_2351/dense_11682/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2351/dense_11682/BiasAdd/ReadVariableOpReadVariableOp;sequential_2351_dense_11682_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2351/dense_11682/BiasAddBiasAdd,sequential_2351/dense_11682/MatMul:product:0:sequential_2351/dense_11682/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
1sequential_2351/dense_11683/MatMul/ReadVariableOpReadVariableOp:sequential_2351_dense_11683_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
È
"sequential_2351/dense_11683/MatMulMatMul,sequential_2351/dense_11682/BiasAdd:output:09sequential_2351/dense_11683/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2351/dense_11683/BiasAdd/ReadVariableOpReadVariableOp;sequential_2351_dense_11683_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2351/dense_11683/BiasAddBiasAdd,sequential_2351/dense_11683/MatMul:product:0:sequential_2351/dense_11683/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
1sequential_2351/dense_11684/MatMul/ReadVariableOpReadVariableOp:sequential_2351_dense_11684_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
È
"sequential_2351/dense_11684/MatMulMatMul,sequential_2351/dense_11683/BiasAdd:output:09sequential_2351/dense_11684/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2351/dense_11684/BiasAdd/ReadVariableOpReadVariableOp;sequential_2351_dense_11684_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2351/dense_11684/BiasAddBiasAdd,sequential_2351/dense_11684/MatMul:product:0:sequential_2351/dense_11684/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
1sequential_2351/dense_11685/MatMul/ReadVariableOpReadVariableOp:sequential_2351_dense_11685_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
È
"sequential_2351/dense_11685/MatMulMatMul,sequential_2351/dense_11684/BiasAdd:output:09sequential_2351/dense_11685/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2351/dense_11685/BiasAdd/ReadVariableOpReadVariableOp;sequential_2351_dense_11685_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2351/dense_11685/BiasAddBiasAdd,sequential_2351/dense_11685/MatMul:product:0:sequential_2351/dense_11685/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ
1sequential_2351/dense_11686/MatMul/ReadVariableOpReadVariableOp:sequential_2351_dense_11686_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
È
"sequential_2351/dense_11686/MatMulMatMul,sequential_2351/dense_11685/BiasAdd:output:09sequential_2351/dense_11686/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
2sequential_2351/dense_11686/BiasAdd/ReadVariableOpReadVariableOp;sequential_2351_dense_11686_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Ë
#sequential_2351/dense_11686/BiasAddBiasAdd,sequential_2351/dense_11686/MatMul:product:0:sequential_2351/dense_11686/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÛ
1sequential_2351/dense_11687/MatMul/ReadVariableOpReadVariableOp:sequential_2351_dense_11687_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ç
"sequential_2351/dense_11687/MatMulMatMul,sequential_2351/dense_11686/BiasAdd:output:09sequential_2351/dense_11687/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
2sequential_2351/dense_11687/BiasAdd/ReadVariableOpReadVariableOp;sequential_2351_dense_11687_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ê
#sequential_2351/dense_11687/BiasAddBiasAdd,sequential_2351/dense_11687/MatMul:product:0:sequential_2351/dense_11687/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 sequential_2351/dense_11687/ReluRelu,sequential_2351/dense_11687/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
IdentityIdentity.sequential_2351/dense_11687/Relu:activations:03^sequential_2351/dense_11681/BiasAdd/ReadVariableOp2^sequential_2351/dense_11681/MatMul/ReadVariableOp3^sequential_2351/dense_11682/BiasAdd/ReadVariableOp2^sequential_2351/dense_11682/MatMul/ReadVariableOp3^sequential_2351/dense_11683/BiasAdd/ReadVariableOp2^sequential_2351/dense_11683/MatMul/ReadVariableOp3^sequential_2351/dense_11684/BiasAdd/ReadVariableOp2^sequential_2351/dense_11684/MatMul/ReadVariableOp3^sequential_2351/dense_11685/BiasAdd/ReadVariableOp2^sequential_2351/dense_11685/MatMul/ReadVariableOp3^sequential_2351/dense_11686/BiasAdd/ReadVariableOp2^sequential_2351/dense_11686/MatMul/ReadVariableOp3^sequential_2351/dense_11687/BiasAdd/ReadVariableOp2^sequential_2351/dense_11687/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2f
1sequential_2351/dense_11683/MatMul/ReadVariableOp1sequential_2351/dense_11683/MatMul/ReadVariableOp2h
2sequential_2351/dense_11685/BiasAdd/ReadVariableOp2sequential_2351/dense_11685/BiasAdd/ReadVariableOp2f
1sequential_2351/dense_11687/MatMul/ReadVariableOp1sequential_2351/dense_11687/MatMul/ReadVariableOp2h
2sequential_2351/dense_11683/BiasAdd/ReadVariableOp2sequential_2351/dense_11683/BiasAdd/ReadVariableOp2f
1sequential_2351/dense_11684/MatMul/ReadVariableOp1sequential_2351/dense_11684/MatMul/ReadVariableOp2h
2sequential_2351/dense_11681/BiasAdd/ReadVariableOp2sequential_2351/dense_11681/BiasAdd/ReadVariableOp2f
1sequential_2351/dense_11681/MatMul/ReadVariableOp1sequential_2351/dense_11681/MatMul/ReadVariableOp2h
2sequential_2351/dense_11686/BiasAdd/ReadVariableOp2sequential_2351/dense_11686/BiasAdd/ReadVariableOp2f
1sequential_2351/dense_11685/MatMul/ReadVariableOp1sequential_2351/dense_11685/MatMul/ReadVariableOp2h
2sequential_2351/dense_11684/BiasAdd/ReadVariableOp2sequential_2351/dense_11684/BiasAdd/ReadVariableOp2f
1sequential_2351/dense_11682/MatMul/ReadVariableOp1sequential_2351/dense_11682/MatMul/ReadVariableOp2f
1sequential_2351/dense_11686/MatMul/ReadVariableOp1sequential_2351/dense_11686/MatMul/ReadVariableOp2h
2sequential_2351/dense_11682/BiasAdd/ReadVariableOp2sequential_2351/dense_11682/BiasAdd/ReadVariableOp2h
2sequential_2351/dense_11687/BiasAdd/ReadVariableOp2sequential_2351/dense_11687/BiasAdd/ReadVariableOp: : : : : :	 : : : :1 -
+
_user_specified_namedense_11681_input: : : : :
 
ç
¯
.__inference_dense_11685_layer_call_fn_19860224

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859793*R
fMRK
I__inference_dense_11685_layer_call_and_return_conditional_losses_19859787*
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
É)

M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859915

inputs.
*dense_11681_statefulpartitionedcall_args_1.
*dense_11681_statefulpartitionedcall_args_2.
*dense_11682_statefulpartitionedcall_args_1.
*dense_11682_statefulpartitionedcall_args_2.
*dense_11683_statefulpartitionedcall_args_1.
*dense_11683_statefulpartitionedcall_args_2.
*dense_11684_statefulpartitionedcall_args_1.
*dense_11684_statefulpartitionedcall_args_2.
*dense_11685_statefulpartitionedcall_args_1.
*dense_11685_statefulpartitionedcall_args_2.
*dense_11686_statefulpartitionedcall_args_1.
*dense_11686_statefulpartitionedcall_args_2.
*dense_11687_statefulpartitionedcall_args_1.
*dense_11687_statefulpartitionedcall_args_2
identity¢#dense_11681/StatefulPartitionedCall¢#dense_11682/StatefulPartitionedCall¢#dense_11683/StatefulPartitionedCall¢#dense_11684/StatefulPartitionedCall¢#dense_11685/StatefulPartitionedCall¢#dense_11686/StatefulPartitionedCall¢#dense_11687/StatefulPartitionedCall
#dense_11681/StatefulPartitionedCallStatefulPartitionedCallinputs*dense_11681_statefulpartitionedcall_args_1*dense_11681_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859685*R
fMRK
I__inference_dense_11681_layer_call_and_return_conditional_losses_19859679*
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
#dense_11682/StatefulPartitionedCallStatefulPartitionedCall,dense_11681/StatefulPartitionedCall:output:0*dense_11682_statefulpartitionedcall_args_1*dense_11682_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859712*R
fMRK
I__inference_dense_11682_layer_call_and_return_conditional_losses_19859706*
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
#dense_11683/StatefulPartitionedCallStatefulPartitionedCall,dense_11682/StatefulPartitionedCall:output:0*dense_11683_statefulpartitionedcall_args_1*dense_11683_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859739*R
fMRK
I__inference_dense_11683_layer_call_and_return_conditional_losses_19859733*
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
#dense_11684/StatefulPartitionedCallStatefulPartitionedCall,dense_11683/StatefulPartitionedCall:output:0*dense_11684_statefulpartitionedcall_args_1*dense_11684_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859766*R
fMRK
I__inference_dense_11684_layer_call_and_return_conditional_losses_19859760*
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
#dense_11685/StatefulPartitionedCallStatefulPartitionedCall,dense_11684/StatefulPartitionedCall:output:0*dense_11685_statefulpartitionedcall_args_1*dense_11685_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859793*R
fMRK
I__inference_dense_11685_layer_call_and_return_conditional_losses_19859787*
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
#dense_11686/StatefulPartitionedCallStatefulPartitionedCall,dense_11685/StatefulPartitionedCall:output:0*dense_11686_statefulpartitionedcall_args_1*dense_11686_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859820*R
fMRK
I__inference_dense_11686_layer_call_and_return_conditional_losses_19859814*
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
#dense_11687/StatefulPartitionedCallStatefulPartitionedCall,dense_11686/StatefulPartitionedCall:output:0*dense_11687_statefulpartitionedcall_args_1*dense_11687_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859848*R
fMRK
I__inference_dense_11687_layer_call_and_return_conditional_losses_19859842*
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
:ÿÿÿÿÿÿÿÿÿþ
IdentityIdentity,dense_11687/StatefulPartitionedCall:output:0$^dense_11681/StatefulPartitionedCall$^dense_11682/StatefulPartitionedCall$^dense_11683/StatefulPartitionedCall$^dense_11684/StatefulPartitionedCall$^dense_11685/StatefulPartitionedCall$^dense_11686/StatefulPartitionedCall$^dense_11687/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2J
#dense_11681/StatefulPartitionedCall#dense_11681/StatefulPartitionedCall2J
#dense_11682/StatefulPartitionedCall#dense_11682/StatefulPartitionedCall2J
#dense_11683/StatefulPartitionedCall#dense_11683/StatefulPartitionedCall2J
#dense_11684/StatefulPartitionedCall#dense_11684/StatefulPartitionedCall2J
#dense_11685/StatefulPartitionedCall#dense_11685/StatefulPartitionedCall2J
#dense_11686/StatefulPartitionedCall#dense_11686/StatefulPartitionedCall2J
#dense_11687/StatefulPartitionedCall#dense_11687/StatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
å
¯
.__inference_dense_11687_layer_call_fn_19860259

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859848*R
fMRK
I__inference_dense_11687_layer_call_and_return_conditional_losses_19859842*
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
Ë
ó
2__inference_sequential_2351_layer_call_fn_19859980
dense_11681_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_11681_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*/
_gradient_op_typePartitionedCall-19859963*V
fQRO
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859962*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :1 -
+
_user_specified_namedense_11681_input: : : : :
 
ê)
¡
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859887
dense_11681_input.
*dense_11681_statefulpartitionedcall_args_1.
*dense_11681_statefulpartitionedcall_args_2.
*dense_11682_statefulpartitionedcall_args_1.
*dense_11682_statefulpartitionedcall_args_2.
*dense_11683_statefulpartitionedcall_args_1.
*dense_11683_statefulpartitionedcall_args_2.
*dense_11684_statefulpartitionedcall_args_1.
*dense_11684_statefulpartitionedcall_args_2.
*dense_11685_statefulpartitionedcall_args_1.
*dense_11685_statefulpartitionedcall_args_2.
*dense_11686_statefulpartitionedcall_args_1.
*dense_11686_statefulpartitionedcall_args_2.
*dense_11687_statefulpartitionedcall_args_1.
*dense_11687_statefulpartitionedcall_args_2
identity¢#dense_11681/StatefulPartitionedCall¢#dense_11682/StatefulPartitionedCall¢#dense_11683/StatefulPartitionedCall¢#dense_11684/StatefulPartitionedCall¢#dense_11685/StatefulPartitionedCall¢#dense_11686/StatefulPartitionedCall¢#dense_11687/StatefulPartitionedCall£
#dense_11681/StatefulPartitionedCallStatefulPartitionedCalldense_11681_input*dense_11681_statefulpartitionedcall_args_1*dense_11681_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859685*R
fMRK
I__inference_dense_11681_layer_call_and_return_conditional_losses_19859679*
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
#dense_11682/StatefulPartitionedCallStatefulPartitionedCall,dense_11681/StatefulPartitionedCall:output:0*dense_11682_statefulpartitionedcall_args_1*dense_11682_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859712*R
fMRK
I__inference_dense_11682_layer_call_and_return_conditional_losses_19859706*
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
#dense_11683/StatefulPartitionedCallStatefulPartitionedCall,dense_11682/StatefulPartitionedCall:output:0*dense_11683_statefulpartitionedcall_args_1*dense_11683_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859739*R
fMRK
I__inference_dense_11683_layer_call_and_return_conditional_losses_19859733*
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
#dense_11684/StatefulPartitionedCallStatefulPartitionedCall,dense_11683/StatefulPartitionedCall:output:0*dense_11684_statefulpartitionedcall_args_1*dense_11684_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859766*R
fMRK
I__inference_dense_11684_layer_call_and_return_conditional_losses_19859760*
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
#dense_11685/StatefulPartitionedCallStatefulPartitionedCall,dense_11684/StatefulPartitionedCall:output:0*dense_11685_statefulpartitionedcall_args_1*dense_11685_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859793*R
fMRK
I__inference_dense_11685_layer_call_and_return_conditional_losses_19859787*
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
#dense_11686/StatefulPartitionedCallStatefulPartitionedCall,dense_11685/StatefulPartitionedCall:output:0*dense_11686_statefulpartitionedcall_args_1*dense_11686_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859820*R
fMRK
I__inference_dense_11686_layer_call_and_return_conditional_losses_19859814*
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
#dense_11687/StatefulPartitionedCallStatefulPartitionedCall,dense_11686/StatefulPartitionedCall:output:0*dense_11687_statefulpartitionedcall_args_1*dense_11687_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859848*R
fMRK
I__inference_dense_11687_layer_call_and_return_conditional_losses_19859842*
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
:ÿÿÿÿÿÿÿÿÿþ
IdentityIdentity,dense_11687/StatefulPartitionedCall:output:0$^dense_11681/StatefulPartitionedCall$^dense_11682/StatefulPartitionedCall$^dense_11683/StatefulPartitionedCall$^dense_11684/StatefulPartitionedCall$^dense_11685/StatefulPartitionedCall$^dense_11686/StatefulPartitionedCall$^dense_11687/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2J
#dense_11681/StatefulPartitionedCall#dense_11681/StatefulPartitionedCall2J
#dense_11682/StatefulPartitionedCall#dense_11682/StatefulPartitionedCall2J
#dense_11683/StatefulPartitionedCall#dense_11683/StatefulPartitionedCall2J
#dense_11684/StatefulPartitionedCall#dense_11684/StatefulPartitionedCall2J
#dense_11685/StatefulPartitionedCall#dense_11685/StatefulPartitionedCall2J
#dense_11686/StatefulPartitionedCall#dense_11686/StatefulPartitionedCall2J
#dense_11687/StatefulPartitionedCall#dense_11687/StatefulPartitionedCall: : : : : :	 : : : :1 -
+
_user_specified_namedense_11681_input: : : : :
 
ç
¯
.__inference_dense_11683_layer_call_fn_19860190

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859739*R
fMRK
I__inference_dense_11683_layer_call_and_return_conditional_losses_19859733*
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
ç
¯
.__inference_dense_11682_layer_call_fn_19860173

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859712*R
fMRK
I__inference_dense_11682_layer_call_and_return_conditional_losses_19859706*
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
±?


M__inference_sequential_2351_layer_call_and_return_conditional_losses_19860054

inputs.
*dense_11681_matmul_readvariableop_resource/
+dense_11681_biasadd_readvariableop_resource.
*dense_11682_matmul_readvariableop_resource/
+dense_11682_biasadd_readvariableop_resource.
*dense_11683_matmul_readvariableop_resource/
+dense_11683_biasadd_readvariableop_resource.
*dense_11684_matmul_readvariableop_resource/
+dense_11684_biasadd_readvariableop_resource.
*dense_11685_matmul_readvariableop_resource/
+dense_11685_biasadd_readvariableop_resource.
*dense_11686_matmul_readvariableop_resource/
+dense_11686_biasadd_readvariableop_resource.
*dense_11687_matmul_readvariableop_resource/
+dense_11687_biasadd_readvariableop_resource
identity¢"dense_11681/BiasAdd/ReadVariableOp¢!dense_11681/MatMul/ReadVariableOp¢"dense_11682/BiasAdd/ReadVariableOp¢!dense_11682/MatMul/ReadVariableOp¢"dense_11683/BiasAdd/ReadVariableOp¢!dense_11683/MatMul/ReadVariableOp¢"dense_11684/BiasAdd/ReadVariableOp¢!dense_11684/MatMul/ReadVariableOp¢"dense_11685/BiasAdd/ReadVariableOp¢!dense_11685/MatMul/ReadVariableOp¢"dense_11686/BiasAdd/ReadVariableOp¢!dense_11686/MatMul/ReadVariableOp¢"dense_11687/BiasAdd/ReadVariableOp¢!dense_11687/MatMul/ReadVariableOp»
!dense_11681/MatMul/ReadVariableOpReadVariableOp*dense_11681_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_11681/MatMulMatMulinputs)dense_11681/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11681/BiasAdd/ReadVariableOpReadVariableOp+dense_11681_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11681/BiasAddBiasAdddense_11681/MatMul:product:0*dense_11681/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11682/MatMul/ReadVariableOpReadVariableOp*dense_11682_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11682/MatMulMatMuldense_11681/BiasAdd:output:0)dense_11682/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11682/BiasAdd/ReadVariableOpReadVariableOp+dense_11682_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11682/BiasAddBiasAdddense_11682/MatMul:product:0*dense_11682/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11683/MatMul/ReadVariableOpReadVariableOp*dense_11683_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11683/MatMulMatMuldense_11682/BiasAdd:output:0)dense_11683/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11683/BiasAdd/ReadVariableOpReadVariableOp+dense_11683_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11683/BiasAddBiasAdddense_11683/MatMul:product:0*dense_11683/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11684/MatMul/ReadVariableOpReadVariableOp*dense_11684_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11684/MatMulMatMuldense_11683/BiasAdd:output:0)dense_11684/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11684/BiasAdd/ReadVariableOpReadVariableOp+dense_11684_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11684/BiasAddBiasAdddense_11684/MatMul:product:0*dense_11684/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11685/MatMul/ReadVariableOpReadVariableOp*dense_11685_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11685/MatMulMatMuldense_11684/BiasAdd:output:0)dense_11685/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11685/BiasAdd/ReadVariableOpReadVariableOp+dense_11685_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11685/BiasAddBiasAdddense_11685/MatMul:product:0*dense_11685/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11686/MatMul/ReadVariableOpReadVariableOp*dense_11686_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11686/MatMulMatMuldense_11685/BiasAdd:output:0)dense_11686/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11686/BiasAdd/ReadVariableOpReadVariableOp+dense_11686_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11686/BiasAddBiasAdddense_11686/MatMul:product:0*dense_11686/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
!dense_11687/MatMul/ReadVariableOpReadVariableOp*dense_11687_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_11687/MatMulMatMuldense_11686/BiasAdd:output:0)dense_11687/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
"dense_11687/BiasAdd/ReadVariableOpReadVariableOp+dense_11687_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_11687/BiasAddBiasAdddense_11687/MatMul:product:0*dense_11687/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_11687/ReluReludense_11687/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
IdentityIdentitydense_11687/Relu:activations:0#^dense_11681/BiasAdd/ReadVariableOp"^dense_11681/MatMul/ReadVariableOp#^dense_11682/BiasAdd/ReadVariableOp"^dense_11682/MatMul/ReadVariableOp#^dense_11683/BiasAdd/ReadVariableOp"^dense_11683/MatMul/ReadVariableOp#^dense_11684/BiasAdd/ReadVariableOp"^dense_11684/MatMul/ReadVariableOp#^dense_11685/BiasAdd/ReadVariableOp"^dense_11685/MatMul/ReadVariableOp#^dense_11686/BiasAdd/ReadVariableOp"^dense_11686/MatMul/ReadVariableOp#^dense_11687/BiasAdd/ReadVariableOp"^dense_11687/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"dense_11682/BiasAdd/ReadVariableOp"dense_11682/BiasAdd/ReadVariableOp2F
!dense_11686/MatMul/ReadVariableOp!dense_11686/MatMul/ReadVariableOp2H
"dense_11687/BiasAdd/ReadVariableOp"dense_11687/BiasAdd/ReadVariableOp2F
!dense_11683/MatMul/ReadVariableOp!dense_11683/MatMul/ReadVariableOp2H
"dense_11685/BiasAdd/ReadVariableOp"dense_11685/BiasAdd/ReadVariableOp2F
!dense_11687/MatMul/ReadVariableOp!dense_11687/MatMul/ReadVariableOp2H
"dense_11683/BiasAdd/ReadVariableOp"dense_11683/BiasAdd/ReadVariableOp2F
!dense_11684/MatMul/ReadVariableOp!dense_11684/MatMul/ReadVariableOp2H
"dense_11681/BiasAdd/ReadVariableOp"dense_11681/BiasAdd/ReadVariableOp2F
!dense_11681/MatMul/ReadVariableOp!dense_11681/MatMul/ReadVariableOp2H
"dense_11686/BiasAdd/ReadVariableOp"dense_11686/BiasAdd/ReadVariableOp2F
!dense_11685/MatMul/ReadVariableOp!dense_11685/MatMul/ReadVariableOp2H
"dense_11684/BiasAdd/ReadVariableOp"dense_11684/BiasAdd/ReadVariableOp2F
!dense_11682/MatMul/ReadVariableOp!dense_11682/MatMul/ReadVariableOp: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
	
â
I__inference_dense_11684_layer_call_and_return_conditional_losses_19859760

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
	
â
I__inference_dense_11683_layer_call_and_return_conditional_losses_19860183

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
»^
þ
!__inference__traced_save_19860434
file_prefix1
-savev2_dense_11681_kernel_read_readvariableop/
+savev2_dense_11681_bias_read_readvariableop1
-savev2_dense_11682_kernel_read_readvariableop/
+savev2_dense_11682_bias_read_readvariableop1
-savev2_dense_11683_kernel_read_readvariableop/
+savev2_dense_11683_bias_read_readvariableop1
-savev2_dense_11684_kernel_read_readvariableop/
+savev2_dense_11684_bias_read_readvariableop1
-savev2_dense_11685_kernel_read_readvariableop/
+savev2_dense_11685_bias_read_readvariableop1
-savev2_dense_11686_kernel_read_readvariableop/
+savev2_dense_11686_bias_read_readvariableop1
-savev2_dense_11687_kernel_read_readvariableop/
+savev2_dense_11687_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_nadam_dense_11681_kernel_m_read_readvariableop7
3savev2_nadam_dense_11681_bias_m_read_readvariableop9
5savev2_nadam_dense_11682_kernel_m_read_readvariableop7
3savev2_nadam_dense_11682_bias_m_read_readvariableop9
5savev2_nadam_dense_11683_kernel_m_read_readvariableop7
3savev2_nadam_dense_11683_bias_m_read_readvariableop9
5savev2_nadam_dense_11684_kernel_m_read_readvariableop7
3savev2_nadam_dense_11684_bias_m_read_readvariableop9
5savev2_nadam_dense_11685_kernel_m_read_readvariableop7
3savev2_nadam_dense_11685_bias_m_read_readvariableop9
5savev2_nadam_dense_11686_kernel_m_read_readvariableop7
3savev2_nadam_dense_11686_bias_m_read_readvariableop9
5savev2_nadam_dense_11687_kernel_m_read_readvariableop7
3savev2_nadam_dense_11687_bias_m_read_readvariableop9
5savev2_nadam_dense_11681_kernel_v_read_readvariableop7
3savev2_nadam_dense_11681_bias_v_read_readvariableop9
5savev2_nadam_dense_11682_kernel_v_read_readvariableop7
3savev2_nadam_dense_11682_bias_v_read_readvariableop9
5savev2_nadam_dense_11683_kernel_v_read_readvariableop7
3savev2_nadam_dense_11683_bias_v_read_readvariableop9
5savev2_nadam_dense_11684_kernel_v_read_readvariableop7
3savev2_nadam_dense_11684_bias_v_read_readvariableop9
5savev2_nadam_dense_11685_kernel_v_read_readvariableop7
3savev2_nadam_dense_11685_bias_v_read_readvariableop9
5savev2_nadam_dense_11686_kernel_v_read_readvariableop7
3savev2_nadam_dense_11686_bias_v_read_readvariableop9
5savev2_nadam_dense_11687_kernel_v_read_readvariableop7
3savev2_nadam_dense_11687_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_9e25c6a095934185b44c1d7b4065cc1d/part*
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
: 
SaveV2/tensor_namesConst"/device:CPU:0*±
value§B¤2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:2Ñ
SaveV2/shape_and_slicesConst"/device:CPU:0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_dense_11681_kernel_read_readvariableop+savev2_dense_11681_bias_read_readvariableop-savev2_dense_11682_kernel_read_readvariableop+savev2_dense_11682_bias_read_readvariableop-savev2_dense_11683_kernel_read_readvariableop+savev2_dense_11683_bias_read_readvariableop-savev2_dense_11684_kernel_read_readvariableop+savev2_dense_11684_bias_read_readvariableop-savev2_dense_11685_kernel_read_readvariableop+savev2_dense_11685_bias_read_readvariableop-savev2_dense_11686_kernel_read_readvariableop+savev2_dense_11686_bias_read_readvariableop-savev2_dense_11687_kernel_read_readvariableop+savev2_dense_11687_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_nadam_dense_11681_kernel_m_read_readvariableop3savev2_nadam_dense_11681_bias_m_read_readvariableop5savev2_nadam_dense_11682_kernel_m_read_readvariableop3savev2_nadam_dense_11682_bias_m_read_readvariableop5savev2_nadam_dense_11683_kernel_m_read_readvariableop3savev2_nadam_dense_11683_bias_m_read_readvariableop5savev2_nadam_dense_11684_kernel_m_read_readvariableop3savev2_nadam_dense_11684_bias_m_read_readvariableop5savev2_nadam_dense_11685_kernel_m_read_readvariableop3savev2_nadam_dense_11685_bias_m_read_readvariableop5savev2_nadam_dense_11686_kernel_m_read_readvariableop3savev2_nadam_dense_11686_bias_m_read_readvariableop5savev2_nadam_dense_11687_kernel_m_read_readvariableop3savev2_nadam_dense_11687_bias_m_read_readvariableop5savev2_nadam_dense_11681_kernel_v_read_readvariableop3savev2_nadam_dense_11681_bias_v_read_readvariableop5savev2_nadam_dense_11682_kernel_v_read_readvariableop3savev2_nadam_dense_11682_bias_v_read_readvariableop5savev2_nadam_dense_11683_kernel_v_read_readvariableop3savev2_nadam_dense_11683_bias_v_read_readvariableop5savev2_nadam_dense_11684_kernel_v_read_readvariableop3savev2_nadam_dense_11684_bias_v_read_readvariableop5savev2_nadam_dense_11685_kernel_v_read_readvariableop3savev2_nadam_dense_11685_bias_v_read_readvariableop5savev2_nadam_dense_11686_kernel_v_read_readvariableop3savev2_nadam_dense_11686_bias_v_read_readvariableop5savev2_nadam_dense_11687_kernel_v_read_readvariableop3savev2_nadam_dense_11687_bias_v_read_readvariableop"/device:CPU:0*@
dtypes6
422	*
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

identity_1Identity_1:output:0*¯
_input_shapes
: :	::
::
::
::
::
::	:: : : : : : : : :	::
::
::
::
::
::	::	::
::
::
::
::
::	:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1: : :0 :# : :	 :+ : :+ '
%
_user_specified_namefile_prefix:3 :" : : :* :% : : :2 :- : : :$ : : :, : :
 : :' : : :/ : : : :& : : :. : : :! : : :) : : :1 :  : : :( 
×	
â
I__inference_dense_11687_layer_call_and_return_conditional_losses_19860252

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
	
â
I__inference_dense_11682_layer_call_and_return_conditional_losses_19859706

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
ê)
¡
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859860
dense_11681_input.
*dense_11681_statefulpartitionedcall_args_1.
*dense_11681_statefulpartitionedcall_args_2.
*dense_11682_statefulpartitionedcall_args_1.
*dense_11682_statefulpartitionedcall_args_2.
*dense_11683_statefulpartitionedcall_args_1.
*dense_11683_statefulpartitionedcall_args_2.
*dense_11684_statefulpartitionedcall_args_1.
*dense_11684_statefulpartitionedcall_args_2.
*dense_11685_statefulpartitionedcall_args_1.
*dense_11685_statefulpartitionedcall_args_2.
*dense_11686_statefulpartitionedcall_args_1.
*dense_11686_statefulpartitionedcall_args_2.
*dense_11687_statefulpartitionedcall_args_1.
*dense_11687_statefulpartitionedcall_args_2
identity¢#dense_11681/StatefulPartitionedCall¢#dense_11682/StatefulPartitionedCall¢#dense_11683/StatefulPartitionedCall¢#dense_11684/StatefulPartitionedCall¢#dense_11685/StatefulPartitionedCall¢#dense_11686/StatefulPartitionedCall¢#dense_11687/StatefulPartitionedCall£
#dense_11681/StatefulPartitionedCallStatefulPartitionedCalldense_11681_input*dense_11681_statefulpartitionedcall_args_1*dense_11681_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859685*R
fMRK
I__inference_dense_11681_layer_call_and_return_conditional_losses_19859679*
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
#dense_11682/StatefulPartitionedCallStatefulPartitionedCall,dense_11681/StatefulPartitionedCall:output:0*dense_11682_statefulpartitionedcall_args_1*dense_11682_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859712*R
fMRK
I__inference_dense_11682_layer_call_and_return_conditional_losses_19859706*
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
#dense_11683/StatefulPartitionedCallStatefulPartitionedCall,dense_11682/StatefulPartitionedCall:output:0*dense_11683_statefulpartitionedcall_args_1*dense_11683_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859739*R
fMRK
I__inference_dense_11683_layer_call_and_return_conditional_losses_19859733*
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
#dense_11684/StatefulPartitionedCallStatefulPartitionedCall,dense_11683/StatefulPartitionedCall:output:0*dense_11684_statefulpartitionedcall_args_1*dense_11684_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859766*R
fMRK
I__inference_dense_11684_layer_call_and_return_conditional_losses_19859760*
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
#dense_11685/StatefulPartitionedCallStatefulPartitionedCall,dense_11684/StatefulPartitionedCall:output:0*dense_11685_statefulpartitionedcall_args_1*dense_11685_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859793*R
fMRK
I__inference_dense_11685_layer_call_and_return_conditional_losses_19859787*
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
#dense_11686/StatefulPartitionedCallStatefulPartitionedCall,dense_11685/StatefulPartitionedCall:output:0*dense_11686_statefulpartitionedcall_args_1*dense_11686_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859820*R
fMRK
I__inference_dense_11686_layer_call_and_return_conditional_losses_19859814*
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
#dense_11687/StatefulPartitionedCallStatefulPartitionedCall,dense_11686/StatefulPartitionedCall:output:0*dense_11687_statefulpartitionedcall_args_1*dense_11687_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859848*R
fMRK
I__inference_dense_11687_layer_call_and_return_conditional_losses_19859842*
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
:ÿÿÿÿÿÿÿÿÿþ
IdentityIdentity,dense_11687/StatefulPartitionedCall:output:0$^dense_11681/StatefulPartitionedCall$^dense_11682/StatefulPartitionedCall$^dense_11683/StatefulPartitionedCall$^dense_11684/StatefulPartitionedCall$^dense_11685/StatefulPartitionedCall$^dense_11686/StatefulPartitionedCall$^dense_11687/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2J
#dense_11681/StatefulPartitionedCall#dense_11681/StatefulPartitionedCall2J
#dense_11682/StatefulPartitionedCall#dense_11682/StatefulPartitionedCall2J
#dense_11683/StatefulPartitionedCall#dense_11683/StatefulPartitionedCall2J
#dense_11684/StatefulPartitionedCall#dense_11684/StatefulPartitionedCall2J
#dense_11685/StatefulPartitionedCall#dense_11685/StatefulPartitionedCall2J
#dense_11686/StatefulPartitionedCall#dense_11686/StatefulPartitionedCall2J
#dense_11687/StatefulPartitionedCall#dense_11687/StatefulPartitionedCall: : : : : :	 : : : :1 -
+
_user_specified_namedense_11681_input: : : : :
 
É)

M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859962

inputs.
*dense_11681_statefulpartitionedcall_args_1.
*dense_11681_statefulpartitionedcall_args_2.
*dense_11682_statefulpartitionedcall_args_1.
*dense_11682_statefulpartitionedcall_args_2.
*dense_11683_statefulpartitionedcall_args_1.
*dense_11683_statefulpartitionedcall_args_2.
*dense_11684_statefulpartitionedcall_args_1.
*dense_11684_statefulpartitionedcall_args_2.
*dense_11685_statefulpartitionedcall_args_1.
*dense_11685_statefulpartitionedcall_args_2.
*dense_11686_statefulpartitionedcall_args_1.
*dense_11686_statefulpartitionedcall_args_2.
*dense_11687_statefulpartitionedcall_args_1.
*dense_11687_statefulpartitionedcall_args_2
identity¢#dense_11681/StatefulPartitionedCall¢#dense_11682/StatefulPartitionedCall¢#dense_11683/StatefulPartitionedCall¢#dense_11684/StatefulPartitionedCall¢#dense_11685/StatefulPartitionedCall¢#dense_11686/StatefulPartitionedCall¢#dense_11687/StatefulPartitionedCall
#dense_11681/StatefulPartitionedCallStatefulPartitionedCallinputs*dense_11681_statefulpartitionedcall_args_1*dense_11681_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859685*R
fMRK
I__inference_dense_11681_layer_call_and_return_conditional_losses_19859679*
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
#dense_11682/StatefulPartitionedCallStatefulPartitionedCall,dense_11681/StatefulPartitionedCall:output:0*dense_11682_statefulpartitionedcall_args_1*dense_11682_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859712*R
fMRK
I__inference_dense_11682_layer_call_and_return_conditional_losses_19859706*
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
#dense_11683/StatefulPartitionedCallStatefulPartitionedCall,dense_11682/StatefulPartitionedCall:output:0*dense_11683_statefulpartitionedcall_args_1*dense_11683_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859739*R
fMRK
I__inference_dense_11683_layer_call_and_return_conditional_losses_19859733*
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
#dense_11684/StatefulPartitionedCallStatefulPartitionedCall,dense_11683/StatefulPartitionedCall:output:0*dense_11684_statefulpartitionedcall_args_1*dense_11684_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859766*R
fMRK
I__inference_dense_11684_layer_call_and_return_conditional_losses_19859760*
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
#dense_11685/StatefulPartitionedCallStatefulPartitionedCall,dense_11684/StatefulPartitionedCall:output:0*dense_11685_statefulpartitionedcall_args_1*dense_11685_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859793*R
fMRK
I__inference_dense_11685_layer_call_and_return_conditional_losses_19859787*
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
#dense_11686/StatefulPartitionedCallStatefulPartitionedCall,dense_11685/StatefulPartitionedCall:output:0*dense_11686_statefulpartitionedcall_args_1*dense_11686_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859820*R
fMRK
I__inference_dense_11686_layer_call_and_return_conditional_losses_19859814*
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
#dense_11687/StatefulPartitionedCallStatefulPartitionedCall,dense_11686/StatefulPartitionedCall:output:0*dense_11687_statefulpartitionedcall_args_1*dense_11687_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-19859848*R
fMRK
I__inference_dense_11687_layer_call_and_return_conditional_losses_19859842*
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
:ÿÿÿÿÿÿÿÿÿþ
IdentityIdentity,dense_11687/StatefulPartitionedCall:output:0$^dense_11681/StatefulPartitionedCall$^dense_11682/StatefulPartitionedCall$^dense_11683/StatefulPartitionedCall$^dense_11684/StatefulPartitionedCall$^dense_11685/StatefulPartitionedCall$^dense_11686/StatefulPartitionedCall$^dense_11687/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2J
#dense_11681/StatefulPartitionedCall#dense_11681/StatefulPartitionedCall2J
#dense_11682/StatefulPartitionedCall#dense_11682/StatefulPartitionedCall2J
#dense_11683/StatefulPartitionedCall#dense_11683/StatefulPartitionedCall2J
#dense_11684/StatefulPartitionedCall#dense_11684/StatefulPartitionedCall2J
#dense_11685/StatefulPartitionedCall#dense_11685/StatefulPartitionedCall2J
#dense_11686/StatefulPartitionedCall#dense_11686/StatefulPartitionedCall2J
#dense_11687/StatefulPartitionedCall#dense_11687/StatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
ù¿
µ
$__inference__traced_restore_19860597
file_prefix'
#assignvariableop_dense_11681_kernel'
#assignvariableop_1_dense_11681_bias)
%assignvariableop_2_dense_11682_kernel'
#assignvariableop_3_dense_11682_bias)
%assignvariableop_4_dense_11683_kernel'
#assignvariableop_5_dense_11683_bias)
%assignvariableop_6_dense_11684_kernel'
#assignvariableop_7_dense_11684_bias)
%assignvariableop_8_dense_11685_kernel'
#assignvariableop_9_dense_11685_bias*
&assignvariableop_10_dense_11686_kernel(
$assignvariableop_11_dense_11686_bias*
&assignvariableop_12_dense_11687_kernel(
$assignvariableop_13_dense_11687_bias"
assignvariableop_14_nadam_iter$
 assignvariableop_15_nadam_beta_1$
 assignvariableop_16_nadam_beta_2#
assignvariableop_17_nadam_decay+
'assignvariableop_18_nadam_learning_rate,
(assignvariableop_19_nadam_momentum_cache
assignvariableop_20_total
assignvariableop_21_count2
.assignvariableop_22_nadam_dense_11681_kernel_m0
,assignvariableop_23_nadam_dense_11681_bias_m2
.assignvariableop_24_nadam_dense_11682_kernel_m0
,assignvariableop_25_nadam_dense_11682_bias_m2
.assignvariableop_26_nadam_dense_11683_kernel_m0
,assignvariableop_27_nadam_dense_11683_bias_m2
.assignvariableop_28_nadam_dense_11684_kernel_m0
,assignvariableop_29_nadam_dense_11684_bias_m2
.assignvariableop_30_nadam_dense_11685_kernel_m0
,assignvariableop_31_nadam_dense_11685_bias_m2
.assignvariableop_32_nadam_dense_11686_kernel_m0
,assignvariableop_33_nadam_dense_11686_bias_m2
.assignvariableop_34_nadam_dense_11687_kernel_m0
,assignvariableop_35_nadam_dense_11687_bias_m2
.assignvariableop_36_nadam_dense_11681_kernel_v0
,assignvariableop_37_nadam_dense_11681_bias_v2
.assignvariableop_38_nadam_dense_11682_kernel_v0
,assignvariableop_39_nadam_dense_11682_bias_v2
.assignvariableop_40_nadam_dense_11683_kernel_v0
,assignvariableop_41_nadam_dense_11683_bias_v2
.assignvariableop_42_nadam_dense_11684_kernel_v0
,assignvariableop_43_nadam_dense_11684_bias_v2
.assignvariableop_44_nadam_dense_11685_kernel_v0
,assignvariableop_45_nadam_dense_11685_bias_v2
.assignvariableop_46_nadam_dense_11686_kernel_v0
,assignvariableop_47_nadam_dense_11686_bias_v2
.assignvariableop_48_nadam_dense_11687_kernel_v0
,assignvariableop_49_nadam_dense_11687_bias_v
identity_51¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*±
value§B¤2B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:2Ô
RestoreV2/shape_and_slicesConst"/device:CPU:0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:2
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
dtypes6
422	*Þ
_output_shapesË
È::::::::::::::::::::::::::::::::::::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_dense_11681_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp#assignvariableop_1_dense_11681_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dense_11682_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_11682_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp%assignvariableop_4_dense_11683_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_11683_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_dense_11684_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_11684_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dense_11685_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_11685_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_dense_11686_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_dense_11686_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_dense_11687_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp$assignvariableop_13_dense_11687_biasIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0	*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_nadam_iterIdentity_14:output:0*
dtype0	*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp assignvariableop_15_nadam_beta_1Identity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp assignvariableop_16_nadam_beta_2Identity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_nadam_decayIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_nadam_learning_rateIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_nadam_momentum_cacheIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:{
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:{
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp.assignvariableop_22_nadam_dense_11681_kernel_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp,assignvariableop_23_nadam_dense_11681_bias_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp.assignvariableop_24_nadam_dense_11682_kernel_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp,assignvariableop_25_nadam_dense_11682_bias_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp.assignvariableop_26_nadam_dense_11683_kernel_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp,assignvariableop_27_nadam_dense_11683_bias_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp.assignvariableop_28_nadam_dense_11684_kernel_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp,assignvariableop_29_nadam_dense_11684_bias_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp.assignvariableop_30_nadam_dense_11685_kernel_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp,assignvariableop_31_nadam_dense_11685_bias_mIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp.assignvariableop_32_nadam_dense_11686_kernel_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp,assignvariableop_33_nadam_dense_11686_bias_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp.assignvariableop_34_nadam_dense_11687_kernel_mIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp,assignvariableop_35_nadam_dense_11687_bias_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp.assignvariableop_36_nadam_dense_11681_kernel_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp,assignvariableop_37_nadam_dense_11681_bias_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp.assignvariableop_38_nadam_dense_11682_kernel_vIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp,assignvariableop_39_nadam_dense_11682_bias_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp.assignvariableop_40_nadam_dense_11683_kernel_vIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp,assignvariableop_41_nadam_dense_11683_bias_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp.assignvariableop_42_nadam_dense_11684_kernel_vIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp,assignvariableop_43_nadam_dense_11684_bias_vIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp.assignvariableop_44_nadam_dense_11685_kernel_vIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp,assignvariableop_45_nadam_dense_11685_bias_vIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp.assignvariableop_46_nadam_dense_11686_kernel_vIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp,assignvariableop_47_nadam_dense_11686_bias_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp.assignvariableop_48_nadam_dense_11687_kernel_vIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp,assignvariableop_49_nadam_dense_11687_bias_vIdentity_49:output:0*
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
 	
Identity_50Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ¨	
Identity_51IdentityIdentity_50:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_51Identity_51:output:0*ß
_input_shapesÍ
Ê: ::::::::::::::::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492
RestoreV2_1RestoreV2_12(
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
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_28: :' : : :/ : : : :& : : :. : : :! : : :) : : :1 :  : : :( : : :0 :# : :	 :+ : :+ '
%
_user_specified_namefile_prefix:" : : :* :% : : :2 :- : : :$ : : :, : :
 
	
â
I__inference_dense_11686_layer_call_and_return_conditional_losses_19859814

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
â
I__inference_dense_11681_layer_call_and_return_conditional_losses_19859679

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp£
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	j
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
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
ª
è
2__inference_sequential_2351_layer_call_fn_19860139

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*/
_gradient_op_typePartitionedCall-19859963*V
fQRO
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859962*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
	
â
I__inference_dense_11684_layer_call_and_return_conditional_losses_19860200

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

ç
&__inference_signature_wrapper_19860005
dense_11681_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCalldense_11681_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*/
_gradient_op_typePartitionedCall-19859988*,
f'R%
#__inference__wrapped_model_19859663*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :1 -
+
_user_specified_namedense_11681_input: : : : :
 : : : : : :	 : 
±?


M__inference_sequential_2351_layer_call_and_return_conditional_losses_19860101

inputs.
*dense_11681_matmul_readvariableop_resource/
+dense_11681_biasadd_readvariableop_resource.
*dense_11682_matmul_readvariableop_resource/
+dense_11682_biasadd_readvariableop_resource.
*dense_11683_matmul_readvariableop_resource/
+dense_11683_biasadd_readvariableop_resource.
*dense_11684_matmul_readvariableop_resource/
+dense_11684_biasadd_readvariableop_resource.
*dense_11685_matmul_readvariableop_resource/
+dense_11685_biasadd_readvariableop_resource.
*dense_11686_matmul_readvariableop_resource/
+dense_11686_biasadd_readvariableop_resource.
*dense_11687_matmul_readvariableop_resource/
+dense_11687_biasadd_readvariableop_resource
identity¢"dense_11681/BiasAdd/ReadVariableOp¢!dense_11681/MatMul/ReadVariableOp¢"dense_11682/BiasAdd/ReadVariableOp¢!dense_11682/MatMul/ReadVariableOp¢"dense_11683/BiasAdd/ReadVariableOp¢!dense_11683/MatMul/ReadVariableOp¢"dense_11684/BiasAdd/ReadVariableOp¢!dense_11684/MatMul/ReadVariableOp¢"dense_11685/BiasAdd/ReadVariableOp¢!dense_11685/MatMul/ReadVariableOp¢"dense_11686/BiasAdd/ReadVariableOp¢!dense_11686/MatMul/ReadVariableOp¢"dense_11687/BiasAdd/ReadVariableOp¢!dense_11687/MatMul/ReadVariableOp»
!dense_11681/MatMul/ReadVariableOpReadVariableOp*dense_11681_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_11681/MatMulMatMulinputs)dense_11681/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11681/BiasAdd/ReadVariableOpReadVariableOp+dense_11681_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11681/BiasAddBiasAdddense_11681/MatMul:product:0*dense_11681/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11682/MatMul/ReadVariableOpReadVariableOp*dense_11682_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11682/MatMulMatMuldense_11681/BiasAdd:output:0)dense_11682/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11682/BiasAdd/ReadVariableOpReadVariableOp+dense_11682_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11682/BiasAddBiasAdddense_11682/MatMul:product:0*dense_11682/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11683/MatMul/ReadVariableOpReadVariableOp*dense_11683_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11683/MatMulMatMuldense_11682/BiasAdd:output:0)dense_11683/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11683/BiasAdd/ReadVariableOpReadVariableOp+dense_11683_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11683/BiasAddBiasAdddense_11683/MatMul:product:0*dense_11683/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11684/MatMul/ReadVariableOpReadVariableOp*dense_11684_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11684/MatMulMatMuldense_11683/BiasAdd:output:0)dense_11684/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11684/BiasAdd/ReadVariableOpReadVariableOp+dense_11684_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11684/BiasAddBiasAdddense_11684/MatMul:product:0*dense_11684/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11685/MatMul/ReadVariableOpReadVariableOp*dense_11685_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11685/MatMulMatMuldense_11684/BiasAdd:output:0)dense_11685/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11685/BiasAdd/ReadVariableOpReadVariableOp+dense_11685_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11685/BiasAddBiasAdddense_11685/MatMul:product:0*dense_11685/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
!dense_11686/MatMul/ReadVariableOpReadVariableOp*dense_11686_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_11686/MatMulMatMuldense_11685/BiasAdd:output:0)dense_11686/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
"dense_11686/BiasAdd/ReadVariableOpReadVariableOp+dense_11686_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_11686/BiasAddBiasAdddense_11686/MatMul:product:0*dense_11686/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
!dense_11687/MatMul/ReadVariableOpReadVariableOp*dense_11687_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_11687/MatMulMatMuldense_11686/BiasAdd:output:0)dense_11687/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸
"dense_11687/BiasAdd/ReadVariableOpReadVariableOp+dense_11687_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_11687/BiasAddBiasAdddense_11687/MatMul:product:0*dense_11687/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_11687/ReluReludense_11687/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿå
IdentityIdentitydense_11687/Relu:activations:0#^dense_11681/BiasAdd/ReadVariableOp"^dense_11681/MatMul/ReadVariableOp#^dense_11682/BiasAdd/ReadVariableOp"^dense_11682/MatMul/ReadVariableOp#^dense_11683/BiasAdd/ReadVariableOp"^dense_11683/MatMul/ReadVariableOp#^dense_11684/BiasAdd/ReadVariableOp"^dense_11684/MatMul/ReadVariableOp#^dense_11685/BiasAdd/ReadVariableOp"^dense_11685/MatMul/ReadVariableOp#^dense_11686/BiasAdd/ReadVariableOp"^dense_11686/MatMul/ReadVariableOp#^dense_11687/BiasAdd/ReadVariableOp"^dense_11687/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"dense_11683/BiasAdd/ReadVariableOp"dense_11683/BiasAdd/ReadVariableOp2F
!dense_11684/MatMul/ReadVariableOp!dense_11684/MatMul/ReadVariableOp2H
"dense_11681/BiasAdd/ReadVariableOp"dense_11681/BiasAdd/ReadVariableOp2F
!dense_11681/MatMul/ReadVariableOp!dense_11681/MatMul/ReadVariableOp2H
"dense_11686/BiasAdd/ReadVariableOp"dense_11686/BiasAdd/ReadVariableOp2F
!dense_11685/MatMul/ReadVariableOp!dense_11685/MatMul/ReadVariableOp2H
"dense_11684/BiasAdd/ReadVariableOp"dense_11684/BiasAdd/ReadVariableOp2F
!dense_11682/MatMul/ReadVariableOp!dense_11682/MatMul/ReadVariableOp2F
!dense_11686/MatMul/ReadVariableOp!dense_11686/MatMul/ReadVariableOp2H
"dense_11682/BiasAdd/ReadVariableOp"dense_11682/BiasAdd/ReadVariableOp2H
"dense_11687/BiasAdd/ReadVariableOp"dense_11687/BiasAdd/ReadVariableOp2F
!dense_11683/MatMul/ReadVariableOp!dense_11683/MatMul/ReadVariableOp2H
"dense_11685/BiasAdd/ReadVariableOp"dense_11685/BiasAdd/ReadVariableOp2F
!dense_11687/MatMul/ReadVariableOp!dense_11687/MatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
	
â
I__inference_dense_11685_layer_call_and_return_conditional_losses_19860217

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
Ë
ó
2__inference_sequential_2351_layer_call_fn_19859933
dense_11681_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_11681_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*/
_gradient_op_typePartitionedCall-19859916*V
fQRO
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859915*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :1 -
+
_user_specified_namedense_11681_input: : : : :
 
×	
â
I__inference_dense_11687_layer_call_and_return_conditional_losses_19859842

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
	
â
I__inference_dense_11686_layer_call_and_return_conditional_losses_19860234

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
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*Â
serving_default®
O
dense_11681_input:
#serving_default_dense_11681_input:0ÿÿÿÿÿÿÿÿÿ?
dense_116870
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ÿ
»=
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"Ï9
_tf_keras_sequential°9{"class_name": "Sequential", "name": "sequential_2351", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2351", "layers": [{"class_name": "Dense", "config": {"name": "dense_11681", "trainable": true, "batch_input_shape": [null, 28], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11682", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11683", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11684", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11685", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11686", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11687", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2351", "layers": [{"class_name": "Dense", "config": {"name": "dense_11681", "trainable": true, "batch_input_shape": [null, 28], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11682", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11683", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11684", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11685", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11686", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_11687", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
¹
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¨
_tf_keras_layer{"class_name": "InputLayer", "name": "dense_11681_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28], "config": {"batch_input_shape": [null, 28], "dtype": "float32", "sparse": false, "name": "dense_11681_input"}}
Á

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Dense", "name": "dense_11681", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28], "config": {"name": "dense_11681", "trainable": true, "batch_input_shape": [null, 28], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28}}}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "Dense", "name": "dense_11682", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_11682", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
__call__
+&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "Dense", "name": "dense_11683", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_11683", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
__call__
+&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "Dense", "name": "dense_11684", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_11684", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
__call__
+&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "Dense", "name": "dense_11685", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_11685", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
__call__
+&call_and_return_all_conditional_losses"ô
_tf_keras_layerÚ{"class_name": "Dense", "name": "dense_11686", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_11686", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}


7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
 __call__
+¡&call_and_return_all_conditional_losses"ð
_tf_keras_layerÖ{"class_name": "Dense", "name": "dense_11687", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_11687", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
ò
=iter

>beta_1

?beta_2
	@decay
Alearning_rate
Bmomentum_cachemsmtmumvmw mx%my&mz+m{,m|1m}2m~7m8mvvvvv v%v&v+v,v1v2v7v8v"
	optimizer
 "
trackable_list_wrapper

0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813"
trackable_list_wrapper

0
1
2
3
4
 5
%6
&7
+8
,9
110
211
712
813"
trackable_list_wrapper
»

regularization_losses

Clayers
	variables
Dmetrics
Elayer_regularization_losses
trainable_variables
Fnon_trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¢serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

regularization_losses

Glayers
	variables
Hmetrics
Ilayer_regularization_losses
trainable_variables
Jnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#	2dense_11681/kernel
:2dense_11681/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

regularization_losses

Klayers
	variables
Lmetrics
Mlayer_regularization_losses
trainable_variables
Nnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2dense_11682/kernel
:2dense_11682/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

regularization_losses

Olayers
	variables
Pmetrics
Qlayer_regularization_losses
trainable_variables
Rnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2dense_11683/kernel
:2dense_11683/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper

!regularization_losses

Slayers
"	variables
Tmetrics
Ulayer_regularization_losses
#trainable_variables
Vnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2dense_11684/kernel
:2dense_11684/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper

'regularization_losses

Wlayers
(	variables
Xmetrics
Ylayer_regularization_losses
)trainable_variables
Znon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2dense_11685/kernel
:2dense_11685/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper

-regularization_losses

[layers
.	variables
\metrics
]layer_regularization_losses
/trainable_variables
^non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
&:$
2dense_11686/kernel
:2dense_11686/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper

3regularization_losses

_layers
4	variables
`metrics
alayer_regularization_losses
5trainable_variables
bnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
%:#	2dense_11687/kernel
:2dense_11687/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper

9regularization_losses

clayers
:	variables
dmetrics
elayer_regularization_losses
;trainable_variables
fnon_trainable_variables
 __call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: 2Nadam/momentum_cache
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
g0"
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

	htotal
	icount
j
_fn_kwargs
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper

kregularization_losses

olayers
l	variables
pmetrics
qlayer_regularization_losses
mtrainable_variables
rnon_trainable_variables
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
+:)	2Nadam/dense_11681/kernel/m
%:#2Nadam/dense_11681/bias/m
,:*
2Nadam/dense_11682/kernel/m
%:#2Nadam/dense_11682/bias/m
,:*
2Nadam/dense_11683/kernel/m
%:#2Nadam/dense_11683/bias/m
,:*
2Nadam/dense_11684/kernel/m
%:#2Nadam/dense_11684/bias/m
,:*
2Nadam/dense_11685/kernel/m
%:#2Nadam/dense_11685/bias/m
,:*
2Nadam/dense_11686/kernel/m
%:#2Nadam/dense_11686/bias/m
+:)	2Nadam/dense_11687/kernel/m
$:"2Nadam/dense_11687/bias/m
+:)	2Nadam/dense_11681/kernel/v
%:#2Nadam/dense_11681/bias/v
,:*
2Nadam/dense_11682/kernel/v
%:#2Nadam/dense_11682/bias/v
,:*
2Nadam/dense_11683/kernel/v
%:#2Nadam/dense_11683/bias/v
,:*
2Nadam/dense_11684/kernel/v
%:#2Nadam/dense_11684/bias/v
,:*
2Nadam/dense_11685/kernel/v
%:#2Nadam/dense_11685/bias/v
,:*
2Nadam/dense_11686/kernel/v
%:#2Nadam/dense_11686/bias/v
+:)	2Nadam/dense_11687/kernel/v
$:"2Nadam/dense_11687/bias/v
2
2__inference_sequential_2351_layer_call_fn_19859933
2__inference_sequential_2351_layer_call_fn_19860120
2__inference_sequential_2351_layer_call_fn_19859980
2__inference_sequential_2351_layer_call_fn_19860139À
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
#__inference__wrapped_model_19859663À
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
dense_11681_inputÿÿÿÿÿÿÿÿÿ
2ÿ
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19860054
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859887
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19860101
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859860À
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
.__inference_dense_11681_layer_call_fn_19860156¢
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
I__inference_dense_11681_layer_call_and_return_conditional_losses_19860149¢
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
.__inference_dense_11682_layer_call_fn_19860173¢
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
I__inference_dense_11682_layer_call_and_return_conditional_losses_19860166¢
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
.__inference_dense_11683_layer_call_fn_19860190¢
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
I__inference_dense_11683_layer_call_and_return_conditional_losses_19860183¢
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
.__inference_dense_11684_layer_call_fn_19860207¢
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
I__inference_dense_11684_layer_call_and_return_conditional_losses_19860200¢
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
.__inference_dense_11685_layer_call_fn_19860224¢
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
I__inference_dense_11685_layer_call_and_return_conditional_losses_19860217¢
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
.__inference_dense_11686_layer_call_fn_19860241¢
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
I__inference_dense_11686_layer_call_and_return_conditional_losses_19860234¢
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
.__inference_dense_11687_layer_call_fn_19860259¢
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
I__inference_dense_11687_layer_call_and_return_conditional_losses_19860252¢
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
&__inference_signature_wrapper_19860005dense_11681_input
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
 «
I__inference_dense_11684_layer_call_and_return_conditional_losses_19860200^%&0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_11683_layer_call_fn_19860190Q 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dense_11687_layer_call_fn_19860259P780¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
2__inference_sequential_2351_layer_call_fn_19859980n %&+,1278B¢?
8¢5
+(
dense_11681_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_11686_layer_call_and_return_conditional_losses_19860234^120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_11686_layer_call_fn_19860241Q120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dense_11682_layer_call_fn_19860173Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
#__inference__wrapped_model_19859663 %&+,1278:¢7
0¢-
+(
dense_11681_inputÿÿÿÿÿÿÿÿÿ
ª "9ª6
4
dense_11687%"
dense_11687ÿÿÿÿÿÿÿÿÿ
.__inference_dense_11685_layer_call_fn_19860224Q+,0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_11682_layer_call_and_return_conditional_losses_19860166^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dense_11681_layer_call_fn_19860156P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dense_11684_layer_call_fn_19860207Q%&0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_11685_layer_call_and_return_conditional_losses_19860217^+,0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ª
I__inference_dense_11687_layer_call_and_return_conditional_losses_19860252]780¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859860{ %&+,1278B¢?
8¢5
+(
dense_11681_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19860101p %&+,12787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_sequential_2351_layer_call_fn_19860120c %&+,12787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÁ
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19860054p %&+,12787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¤
2__inference_sequential_2351_layer_call_fn_19859933n %&+,1278B¢?
8¢5
+(
dense_11681_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÇ
&__inference_signature_wrapper_19860005 %&+,1278O¢L
¢ 
EªB
@
dense_11681_input+(
dense_11681_inputÿÿÿÿÿÿÿÿÿ"9ª6
4
dense_11687%"
dense_11687ÿÿÿÿÿÿÿÿÿª
I__inference_dense_11681_layer_call_and_return_conditional_losses_19860149]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_sequential_2351_layer_call_fn_19860139c %&+,12787¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dense_11683_layer_call_and_return_conditional_losses_19860183^ 0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Ì
M__inference_sequential_2351_layer_call_and_return_conditional_losses_19859887{ %&+,1278B¢?
8¢5
+(
dense_11681_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 