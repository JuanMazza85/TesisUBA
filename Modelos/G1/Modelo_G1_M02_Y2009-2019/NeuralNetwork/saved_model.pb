¨é
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
shapeshape"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8

dense_3399/kernelVarHandleOp*
shape:	*"
shared_namedense_3399/kernel*
dtype0*
_output_shapes
: 
x
%dense_3399/kernel/Read/ReadVariableOpReadVariableOpdense_3399/kernel*
dtype0*
_output_shapes
:	
w
dense_3399/biasVarHandleOp*
shape:* 
shared_namedense_3399/bias*
dtype0*
_output_shapes
: 
p
#dense_3399/bias/Read/ReadVariableOpReadVariableOpdense_3399/bias*
dtype0*
_output_shapes	
:

dense_3400/kernelVarHandleOp*
shape:
*"
shared_namedense_3400/kernel*
dtype0*
_output_shapes
: 
y
%dense_3400/kernel/Read/ReadVariableOpReadVariableOpdense_3400/kernel*
dtype0* 
_output_shapes
:

w
dense_3400/biasVarHandleOp*
shape:* 
shared_namedense_3400/bias*
dtype0*
_output_shapes
: 
p
#dense_3400/bias/Read/ReadVariableOpReadVariableOpdense_3400/bias*
dtype0*
_output_shapes	
:

dense_3401/kernelVarHandleOp*
shape:
*"
shared_namedense_3401/kernel*
dtype0*
_output_shapes
: 
y
%dense_3401/kernel/Read/ReadVariableOpReadVariableOpdense_3401/kernel*
dtype0* 
_output_shapes
:

w
dense_3401/biasVarHandleOp*
shape:* 
shared_namedense_3401/bias*
dtype0*
_output_shapes
: 
p
#dense_3401/bias/Read/ReadVariableOpReadVariableOpdense_3401/bias*
dtype0*
_output_shapes	
:

dense_3402/kernelVarHandleOp*
shape:
*"
shared_namedense_3402/kernel*
dtype0*
_output_shapes
: 
y
%dense_3402/kernel/Read/ReadVariableOpReadVariableOpdense_3402/kernel*
dtype0* 
_output_shapes
:

w
dense_3402/biasVarHandleOp*
shape:* 
shared_namedense_3402/bias*
dtype0*
_output_shapes
: 
p
#dense_3402/bias/Read/ReadVariableOpReadVariableOpdense_3402/bias*
dtype0*
_output_shapes	
:

dense_3403/kernelVarHandleOp*
shape:
*"
shared_namedense_3403/kernel*
dtype0*
_output_shapes
: 
y
%dense_3403/kernel/Read/ReadVariableOpReadVariableOpdense_3403/kernel*
dtype0* 
_output_shapes
:

w
dense_3403/biasVarHandleOp*
shape:* 
shared_namedense_3403/bias*
dtype0*
_output_shapes
: 
p
#dense_3403/bias/Read/ReadVariableOpReadVariableOpdense_3403/bias*
dtype0*
_output_shapes	
:

dense_3404/kernelVarHandleOp*
shape:
*"
shared_namedense_3404/kernel*
dtype0*
_output_shapes
: 
y
%dense_3404/kernel/Read/ReadVariableOpReadVariableOpdense_3404/kernel*
dtype0* 
_output_shapes
:

w
dense_3404/biasVarHandleOp*
shape:* 
shared_namedense_3404/bias*
dtype0*
_output_shapes
: 
p
#dense_3404/bias/Read/ReadVariableOpReadVariableOpdense_3404/bias*
dtype0*
_output_shapes	
:

dense_3405/kernelVarHandleOp*
shape:	*"
shared_namedense_3405/kernel*
dtype0*
_output_shapes
: 
x
%dense_3405/kernel/Read/ReadVariableOpReadVariableOpdense_3405/kernel*
dtype0*
_output_shapes
:	
v
dense_3405/biasVarHandleOp*
shape:* 
shared_namedense_3405/bias*
dtype0*
_output_shapes
: 
o
#dense_3405/bias/Read/ReadVariableOpReadVariableOpdense_3405/bias*
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

Nadam/dense_3399/kernel/mVarHandleOp*
shape:	**
shared_nameNadam/dense_3399/kernel/m*
dtype0*
_output_shapes
: 

-Nadam/dense_3399/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3399/kernel/m*
dtype0*
_output_shapes
:	

Nadam/dense_3399/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_3399/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_3399/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3399/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_3400/kernel/mVarHandleOp*
shape:
**
shared_nameNadam/dense_3400/kernel/m*
dtype0*
_output_shapes
: 

-Nadam/dense_3400/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3400/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_3400/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_3400/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_3400/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3400/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_3401/kernel/mVarHandleOp*
shape:
**
shared_nameNadam/dense_3401/kernel/m*
dtype0*
_output_shapes
: 

-Nadam/dense_3401/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3401/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_3401/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_3401/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_3401/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3401/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_3402/kernel/mVarHandleOp*
shape:
**
shared_nameNadam/dense_3402/kernel/m*
dtype0*
_output_shapes
: 

-Nadam/dense_3402/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3402/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_3402/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_3402/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_3402/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3402/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_3403/kernel/mVarHandleOp*
shape:
**
shared_nameNadam/dense_3403/kernel/m*
dtype0*
_output_shapes
: 

-Nadam/dense_3403/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3403/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_3403/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_3403/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_3403/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3403/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_3404/kernel/mVarHandleOp*
shape:
**
shared_nameNadam/dense_3404/kernel/m*
dtype0*
_output_shapes
: 

-Nadam/dense_3404/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3404/kernel/m*
dtype0* 
_output_shapes
:


Nadam/dense_3404/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_3404/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_3404/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3404/bias/m*
dtype0*
_output_shapes	
:

Nadam/dense_3405/kernel/mVarHandleOp*
shape:	**
shared_nameNadam/dense_3405/kernel/m*
dtype0*
_output_shapes
: 

-Nadam/dense_3405/kernel/m/Read/ReadVariableOpReadVariableOpNadam/dense_3405/kernel/m*
dtype0*
_output_shapes
:	

Nadam/dense_3405/bias/mVarHandleOp*
shape:*(
shared_nameNadam/dense_3405/bias/m*
dtype0*
_output_shapes
: 

+Nadam/dense_3405/bias/m/Read/ReadVariableOpReadVariableOpNadam/dense_3405/bias/m*
dtype0*
_output_shapes
:

Nadam/dense_3399/kernel/vVarHandleOp*
shape:	**
shared_nameNadam/dense_3399/kernel/v*
dtype0*
_output_shapes
: 

-Nadam/dense_3399/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3399/kernel/v*
dtype0*
_output_shapes
:	

Nadam/dense_3399/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_3399/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_3399/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3399/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_3400/kernel/vVarHandleOp*
shape:
**
shared_nameNadam/dense_3400/kernel/v*
dtype0*
_output_shapes
: 

-Nadam/dense_3400/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3400/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_3400/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_3400/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_3400/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3400/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_3401/kernel/vVarHandleOp*
shape:
**
shared_nameNadam/dense_3401/kernel/v*
dtype0*
_output_shapes
: 

-Nadam/dense_3401/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3401/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_3401/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_3401/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_3401/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3401/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_3402/kernel/vVarHandleOp*
shape:
**
shared_nameNadam/dense_3402/kernel/v*
dtype0*
_output_shapes
: 

-Nadam/dense_3402/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3402/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_3402/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_3402/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_3402/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3402/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_3403/kernel/vVarHandleOp*
shape:
**
shared_nameNadam/dense_3403/kernel/v*
dtype0*
_output_shapes
: 

-Nadam/dense_3403/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3403/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_3403/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_3403/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_3403/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3403/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_3404/kernel/vVarHandleOp*
shape:
**
shared_nameNadam/dense_3404/kernel/v*
dtype0*
_output_shapes
: 

-Nadam/dense_3404/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3404/kernel/v*
dtype0* 
_output_shapes
:


Nadam/dense_3404/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_3404/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_3404/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3404/bias/v*
dtype0*
_output_shapes	
:

Nadam/dense_3405/kernel/vVarHandleOp*
shape:	**
shared_nameNadam/dense_3405/kernel/v*
dtype0*
_output_shapes
: 

-Nadam/dense_3405/kernel/v/Read/ReadVariableOpReadVariableOpNadam/dense_3405/kernel/v*
dtype0*
_output_shapes
:	

Nadam/dense_3405/bias/vVarHandleOp*
shape:*(
shared_nameNadam/dense_3405/bias/v*
dtype0*
_output_shapes
: 

+Nadam/dense_3405/bias/v/Read/ReadVariableOpReadVariableOpNadam/dense_3405/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
V
ConstConst"/device:CPU:0*¾U
value´UB±U BªU
Ó
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
h

2kernel
3bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api
R
8regularization_losses
9	variables
:trainable_variables
;	keras_api
h

<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
R
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
ì
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[momentum_cachem m¡m¢m£(m¤)m¥2m¦3m§<m¨=m©FmªGm«Pm¬Qm­v®v¯v°v±(v²)v³2v´3vµ<v¶=v·Fv¸Gv¹PvºQv»
 
f
0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13
f
0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13

regularization_losses

\layers
	variables
]metrics
^layer_regularization_losses
trainable_variables
_non_trainable_variables
 
 
 
 

regularization_losses

`layers
	variables
ametrics
blayer_regularization_losses
trainable_variables
cnon_trainable_variables
][
VARIABLE_VALUEdense_3399/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3399/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses

dlayers
	variables
emetrics
flayer_regularization_losses
trainable_variables
gnon_trainable_variables
][
VARIABLE_VALUEdense_3400/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3400/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

 regularization_losses

hlayers
!	variables
imetrics
jlayer_regularization_losses
"trainable_variables
knon_trainable_variables
 
 
 

$regularization_losses

llayers
%	variables
mmetrics
nlayer_regularization_losses
&trainable_variables
onon_trainable_variables
][
VARIABLE_VALUEdense_3401/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3401/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1

*regularization_losses

players
+	variables
qmetrics
rlayer_regularization_losses
,trainable_variables
snon_trainable_variables
 
 
 

.regularization_losses

tlayers
/	variables
umetrics
vlayer_regularization_losses
0trainable_variables
wnon_trainable_variables
][
VARIABLE_VALUEdense_3402/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3402/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

20
31

20
31

4regularization_losses

xlayers
5	variables
ymetrics
zlayer_regularization_losses
6trainable_variables
{non_trainable_variables
 
 
 

8regularization_losses

|layers
9	variables
}metrics
~layer_regularization_losses
:trainable_variables
non_trainable_variables
][
VARIABLE_VALUEdense_3403/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3403/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1

>regularization_losses
layers
?	variables
metrics
 layer_regularization_losses
@trainable_variables
non_trainable_variables
 
 
 

Bregularization_losses
layers
C	variables
metrics
 layer_regularization_losses
Dtrainable_variables
non_trainable_variables
][
VARIABLE_VALUEdense_3404/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3404/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1

Hregularization_losses
layers
I	variables
metrics
 layer_regularization_losses
Jtrainable_variables
non_trainable_variables
 
 
 

Lregularization_losses
layers
M	variables
metrics
 layer_regularization_losses
Ntrainable_variables
non_trainable_variables
][
VARIABLE_VALUEdense_3405/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEdense_3405/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1

Rregularization_losses
layers
S	variables
metrics
 layer_regularization_losses
Ttrainable_variables
non_trainable_variables
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
V
0
1
2
3
4
5
6
	7

8
9
10
11

0
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


total

count

_fn_kwargs
regularization_losses
	variables
trainable_variables
	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
¡
regularization_losses
layers
	variables
metrics
 layer_regularization_losses
trainable_variables
non_trainable_variables
 
 
 

0
1

VARIABLE_VALUENadam/dense_3399/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3399/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3400/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3400/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3401/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3401/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3402/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3402/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3403/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3403/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3404/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3404/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3405/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3405/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3399/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3399/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3400/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3400/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3401/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3401/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3402/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3402/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3403/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3403/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3404/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3404/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUENadam/dense_3405/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUENadam/dense_3405/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 

 serving_default_dense_3399_inputPlaceholder*
shape:ÿÿÿÿÿÿÿÿÿ*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
°
StatefulPartitionedCallStatefulPartitionedCall serving_default_dense_3399_inputdense_3399/kerneldense_3399/biasdense_3400/kerneldense_3400/biasdense_3401/kerneldense_3401/biasdense_3402/kerneldense_3402/biasdense_3403/kerneldense_3403/biasdense_3404/kerneldense_3404/biasdense_3405/kerneldense_3405/bias*.
_gradient_op_typePartitionedCall-5791127*.
f)R'
%__inference_signature_wrapper_5790509*
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
ä
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_3399/kernel/Read/ReadVariableOp#dense_3399/bias/Read/ReadVariableOp%dense_3400/kernel/Read/ReadVariableOp#dense_3400/bias/Read/ReadVariableOp%dense_3401/kernel/Read/ReadVariableOp#dense_3401/bias/Read/ReadVariableOp%dense_3402/kernel/Read/ReadVariableOp#dense_3402/bias/Read/ReadVariableOp%dense_3403/kernel/Read/ReadVariableOp#dense_3403/bias/Read/ReadVariableOp%dense_3404/kernel/Read/ReadVariableOp#dense_3404/bias/Read/ReadVariableOp%dense_3405/kernel/Read/ReadVariableOp#dense_3405/bias/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp Nadam/beta_1/Read/ReadVariableOp Nadam/beta_2/Read/ReadVariableOpNadam/decay/Read/ReadVariableOp'Nadam/learning_rate/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp-Nadam/dense_3399/kernel/m/Read/ReadVariableOp+Nadam/dense_3399/bias/m/Read/ReadVariableOp-Nadam/dense_3400/kernel/m/Read/ReadVariableOp+Nadam/dense_3400/bias/m/Read/ReadVariableOp-Nadam/dense_3401/kernel/m/Read/ReadVariableOp+Nadam/dense_3401/bias/m/Read/ReadVariableOp-Nadam/dense_3402/kernel/m/Read/ReadVariableOp+Nadam/dense_3402/bias/m/Read/ReadVariableOp-Nadam/dense_3403/kernel/m/Read/ReadVariableOp+Nadam/dense_3403/bias/m/Read/ReadVariableOp-Nadam/dense_3404/kernel/m/Read/ReadVariableOp+Nadam/dense_3404/bias/m/Read/ReadVariableOp-Nadam/dense_3405/kernel/m/Read/ReadVariableOp+Nadam/dense_3405/bias/m/Read/ReadVariableOp-Nadam/dense_3399/kernel/v/Read/ReadVariableOp+Nadam/dense_3399/bias/v/Read/ReadVariableOp-Nadam/dense_3400/kernel/v/Read/ReadVariableOp+Nadam/dense_3400/bias/v/Read/ReadVariableOp-Nadam/dense_3401/kernel/v/Read/ReadVariableOp+Nadam/dense_3401/bias/v/Read/ReadVariableOp-Nadam/dense_3402/kernel/v/Read/ReadVariableOp+Nadam/dense_3402/bias/v/Read/ReadVariableOp-Nadam/dense_3403/kernel/v/Read/ReadVariableOp+Nadam/dense_3403/bias/v/Read/ReadVariableOp-Nadam/dense_3404/kernel/v/Read/ReadVariableOp+Nadam/dense_3404/bias/v/Read/ReadVariableOp-Nadam/dense_3405/kernel/v/Read/ReadVariableOp+Nadam/dense_3405/bias/v/Read/ReadVariableOpConst*.
_gradient_op_typePartitionedCall-5791199*)
f$R"
 __inference__traced_save_5791198*
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
÷

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_3399/kerneldense_3399/biasdense_3400/kerneldense_3400/biasdense_3401/kerneldense_3401/biasdense_3402/kerneldense_3402/biasdense_3403/kerneldense_3403/biasdense_3404/kerneldense_3404/biasdense_3405/kerneldense_3405/bias
Nadam/iterNadam/beta_1Nadam/beta_2Nadam/decayNadam/learning_rateNadam/momentum_cachetotalcountNadam/dense_3399/kernel/mNadam/dense_3399/bias/mNadam/dense_3400/kernel/mNadam/dense_3400/bias/mNadam/dense_3401/kernel/mNadam/dense_3401/bias/mNadam/dense_3402/kernel/mNadam/dense_3402/bias/mNadam/dense_3403/kernel/mNadam/dense_3403/bias/mNadam/dense_3404/kernel/mNadam/dense_3404/bias/mNadam/dense_3405/kernel/mNadam/dense_3405/bias/mNadam/dense_3399/kernel/vNadam/dense_3399/bias/vNadam/dense_3400/kernel/vNadam/dense_3400/bias/vNadam/dense_3401/kernel/vNadam/dense_3401/bias/vNadam/dense_3402/kernel/vNadam/dense_3402/bias/vNadam/dense_3403/kernel/vNadam/dense_3403/bias/vNadam/dense_3404/kernel/vNadam/dense_3404/bias/vNadam/dense_3405/kernel/vNadam/dense_3405/bias/v*.
_gradient_op_typePartitionedCall-5791362*,
f'R%
#__inference__traced_restore_5791361*
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
: ¶¢

à
­
,__inference_dense_3405_layer_call_fn_5791023

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790337*P
fKRI
G__inference_dense_3405_layer_call_and_return_conditional_losses_5790331*
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
Ã
ð
0__inference_sequential_685_layer_call_fn_5790484
dense_3399_input"
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3399_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*.
_gradient_op_typePartitionedCall-5790467*T
fORM
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790466*
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
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :0 ,
*
_user_specified_namedense_3399_input: : : : :
 
¶
h
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790154

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
º>
Ì	
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790349
dense_3399_input-
)dense_3399_statefulpartitionedcall_args_1-
)dense_3399_statefulpartitionedcall_args_2-
)dense_3400_statefulpartitionedcall_args_1-
)dense_3400_statefulpartitionedcall_args_2-
)dense_3401_statefulpartitionedcall_args_1-
)dense_3401_statefulpartitionedcall_args_2-
)dense_3402_statefulpartitionedcall_args_1-
)dense_3402_statefulpartitionedcall_args_2-
)dense_3403_statefulpartitionedcall_args_1-
)dense_3403_statefulpartitionedcall_args_2-
)dense_3404_statefulpartitionedcall_args_1-
)dense_3404_statefulpartitionedcall_args_2-
)dense_3405_statefulpartitionedcall_args_1-
)dense_3405_statefulpartitionedcall_args_2
identity¢"dense_3399/StatefulPartitionedCall¢"dense_3400/StatefulPartitionedCall¢"dense_3401/StatefulPartitionedCall¢"dense_3402/StatefulPartitionedCall¢"dense_3403/StatefulPartitionedCall¢"dense_3404/StatefulPartitionedCall¢"dense_3405/StatefulPartitionedCall¢$dropout_1354/StatefulPartitionedCall¢$dropout_1355/StatefulPartitionedCall¢$dropout_1356/StatefulPartitionedCall¢$dropout_1357/StatefulPartitionedCall¢$dropout_1358/StatefulPartitionedCall
"dense_3399/StatefulPartitionedCallStatefulPartitionedCalldense_3399_input)dense_3399_statefulpartitionedcall_args_1)dense_3399_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789954*P
fKRI
G__inference_dense_3399_layer_call_and_return_conditional_losses_5789948*
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
"dense_3400/StatefulPartitionedCallStatefulPartitionedCall+dense_3399/StatefulPartitionedCall:output:0)dense_3400_statefulpartitionedcall_args_1)dense_3400_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789981*P
fKRI
G__inference_dense_3400_layer_call_and_return_conditional_losses_5789975*
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
:ÿÿÿÿÿÿÿÿÿã
$dropout_1354/StatefulPartitionedCallStatefulPartitionedCall+dense_3400/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790023*R
fMRK
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790012*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3401/StatefulPartitionedCallStatefulPartitionedCall-dropout_1354/StatefulPartitionedCall:output:0)dense_3401_statefulpartitionedcall_args_1)dense_3401_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790052*P
fKRI
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790046*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1355/StatefulPartitionedCallStatefulPartitionedCall+dense_3401/StatefulPartitionedCall:output:0%^dropout_1354/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790094*R
fMRK
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790083*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3402/StatefulPartitionedCallStatefulPartitionedCall-dropout_1355/StatefulPartitionedCall:output:0)dense_3402_statefulpartitionedcall_args_1)dense_3402_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790123*P
fKRI
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790117*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1356/StatefulPartitionedCallStatefulPartitionedCall+dense_3402/StatefulPartitionedCall:output:0%^dropout_1355/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790165*R
fMRK
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790154*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3403/StatefulPartitionedCallStatefulPartitionedCall-dropout_1356/StatefulPartitionedCall:output:0)dense_3403_statefulpartitionedcall_args_1)dense_3403_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790194*P
fKRI
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790188*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1357/StatefulPartitionedCallStatefulPartitionedCall+dense_3403/StatefulPartitionedCall:output:0%^dropout_1356/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790236*R
fMRK
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790225*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3404/StatefulPartitionedCallStatefulPartitionedCall-dropout_1357/StatefulPartitionedCall:output:0)dense_3404_statefulpartitionedcall_args_1)dense_3404_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790265*P
fKRI
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790259*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1358/StatefulPartitionedCallStatefulPartitionedCall+dense_3404/StatefulPartitionedCall:output:0%^dropout_1357/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790307*R
fMRK
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790296*
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
:ÿÿÿÿÿÿÿÿÿ¸
"dense_3405/StatefulPartitionedCallStatefulPartitionedCall-dropout_1358/StatefulPartitionedCall:output:0)dense_3405_statefulpartitionedcall_args_1)dense_3405_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790337*P
fKRI
G__inference_dense_3405_layer_call_and_return_conditional_losses_5790331*
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
:ÿÿÿÿÿÿÿÿÿ¹
IdentityIdentity+dense_3405/StatefulPartitionedCall:output:0#^dense_3399/StatefulPartitionedCall#^dense_3400/StatefulPartitionedCall#^dense_3401/StatefulPartitionedCall#^dense_3402/StatefulPartitionedCall#^dense_3403/StatefulPartitionedCall#^dense_3404/StatefulPartitionedCall#^dense_3405/StatefulPartitionedCall%^dropout_1354/StatefulPartitionedCall%^dropout_1355/StatefulPartitionedCall%^dropout_1356/StatefulPartitionedCall%^dropout_1357/StatefulPartitionedCall%^dropout_1358/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2L
$dropout_1354/StatefulPartitionedCall$dropout_1354/StatefulPartitionedCall2H
"dense_3399/StatefulPartitionedCall"dense_3399/StatefulPartitionedCall2L
$dropout_1355/StatefulPartitionedCall$dropout_1355/StatefulPartitionedCall2L
$dropout_1356/StatefulPartitionedCall$dropout_1356/StatefulPartitionedCall2L
$dropout_1357/StatefulPartitionedCall$dropout_1357/StatefulPartitionedCall2L
$dropout_1358/StatefulPartitionedCall$dropout_1358/StatefulPartitionedCall2H
"dense_3400/StatefulPartitionedCall"dense_3400/StatefulPartitionedCall2H
"dense_3401/StatefulPartitionedCall"dense_3401/StatefulPartitionedCall2H
"dense_3402/StatefulPartitionedCall"dense_3402/StatefulPartitionedCall2H
"dense_3403/StatefulPartitionedCall"dense_3403/StatefulPartitionedCall2H
"dense_3404/StatefulPartitionedCall"dense_3404/StatefulPartitionedCall2H
"dense_3405/StatefulPartitionedCall"dense_3405/StatefulPartitionedCall: : : : : :	 : : : :0 ,
*
_user_specified_namedense_3399_input: : : : :
 
	
à
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790859

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
	
à
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790046

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
>
Â	
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790414

inputs-
)dense_3399_statefulpartitionedcall_args_1-
)dense_3399_statefulpartitionedcall_args_2-
)dense_3400_statefulpartitionedcall_args_1-
)dense_3400_statefulpartitionedcall_args_2-
)dense_3401_statefulpartitionedcall_args_1-
)dense_3401_statefulpartitionedcall_args_2-
)dense_3402_statefulpartitionedcall_args_1-
)dense_3402_statefulpartitionedcall_args_2-
)dense_3403_statefulpartitionedcall_args_1-
)dense_3403_statefulpartitionedcall_args_2-
)dense_3404_statefulpartitionedcall_args_1-
)dense_3404_statefulpartitionedcall_args_2-
)dense_3405_statefulpartitionedcall_args_1-
)dense_3405_statefulpartitionedcall_args_2
identity¢"dense_3399/StatefulPartitionedCall¢"dense_3400/StatefulPartitionedCall¢"dense_3401/StatefulPartitionedCall¢"dense_3402/StatefulPartitionedCall¢"dense_3403/StatefulPartitionedCall¢"dense_3404/StatefulPartitionedCall¢"dense_3405/StatefulPartitionedCall¢$dropout_1354/StatefulPartitionedCall¢$dropout_1355/StatefulPartitionedCall¢$dropout_1356/StatefulPartitionedCall¢$dropout_1357/StatefulPartitionedCall¢$dropout_1358/StatefulPartitionedCall
"dense_3399/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_3399_statefulpartitionedcall_args_1)dense_3399_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789954*P
fKRI
G__inference_dense_3399_layer_call_and_return_conditional_losses_5789948*
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
"dense_3400/StatefulPartitionedCallStatefulPartitionedCall+dense_3399/StatefulPartitionedCall:output:0)dense_3400_statefulpartitionedcall_args_1)dense_3400_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789981*P
fKRI
G__inference_dense_3400_layer_call_and_return_conditional_losses_5789975*
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
:ÿÿÿÿÿÿÿÿÿã
$dropout_1354/StatefulPartitionedCallStatefulPartitionedCall+dense_3400/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790023*R
fMRK
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790012*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3401/StatefulPartitionedCallStatefulPartitionedCall-dropout_1354/StatefulPartitionedCall:output:0)dense_3401_statefulpartitionedcall_args_1)dense_3401_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790052*P
fKRI
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790046*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1355/StatefulPartitionedCallStatefulPartitionedCall+dense_3401/StatefulPartitionedCall:output:0%^dropout_1354/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790094*R
fMRK
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790083*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3402/StatefulPartitionedCallStatefulPartitionedCall-dropout_1355/StatefulPartitionedCall:output:0)dense_3402_statefulpartitionedcall_args_1)dense_3402_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790123*P
fKRI
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790117*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1356/StatefulPartitionedCallStatefulPartitionedCall+dense_3402/StatefulPartitionedCall:output:0%^dropout_1355/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790165*R
fMRK
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790154*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3403/StatefulPartitionedCallStatefulPartitionedCall-dropout_1356/StatefulPartitionedCall:output:0)dense_3403_statefulpartitionedcall_args_1)dense_3403_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790194*P
fKRI
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790188*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1357/StatefulPartitionedCallStatefulPartitionedCall+dense_3403/StatefulPartitionedCall:output:0%^dropout_1356/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790236*R
fMRK
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790225*
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
:ÿÿÿÿÿÿÿÿÿ¹
"dense_3404/StatefulPartitionedCallStatefulPartitionedCall-dropout_1357/StatefulPartitionedCall:output:0)dense_3404_statefulpartitionedcall_args_1)dense_3404_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790265*P
fKRI
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790259*
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
:ÿÿÿÿÿÿÿÿÿ
$dropout_1358/StatefulPartitionedCallStatefulPartitionedCall+dense_3404/StatefulPartitionedCall:output:0%^dropout_1357/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-5790307*R
fMRK
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790296*
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
:ÿÿÿÿÿÿÿÿÿ¸
"dense_3405/StatefulPartitionedCallStatefulPartitionedCall-dropout_1358/StatefulPartitionedCall:output:0)dense_3405_statefulpartitionedcall_args_1)dense_3405_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790337*P
fKRI
G__inference_dense_3405_layer_call_and_return_conditional_losses_5790331*
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
:ÿÿÿÿÿÿÿÿÿ¹
IdentityIdentity+dense_3405/StatefulPartitionedCall:output:0#^dense_3399/StatefulPartitionedCall#^dense_3400/StatefulPartitionedCall#^dense_3401/StatefulPartitionedCall#^dense_3402/StatefulPartitionedCall#^dense_3403/StatefulPartitionedCall#^dense_3404/StatefulPartitionedCall#^dense_3405/StatefulPartitionedCall%^dropout_1354/StatefulPartitionedCall%^dropout_1355/StatefulPartitionedCall%^dropout_1356/StatefulPartitionedCall%^dropout_1357/StatefulPartitionedCall%^dropout_1358/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2L
$dropout_1354/StatefulPartitionedCall$dropout_1354/StatefulPartitionedCall2H
"dense_3399/StatefulPartitionedCall"dense_3399/StatefulPartitionedCall2L
$dropout_1355/StatefulPartitionedCall$dropout_1355/StatefulPartitionedCall2L
$dropout_1356/StatefulPartitionedCall$dropout_1356/StatefulPartitionedCall2L
$dropout_1357/StatefulPartitionedCall$dropout_1357/StatefulPartitionedCall2L
$dropout_1358/StatefulPartitionedCall$dropout_1358/StatefulPartitionedCall2H
"dense_3400/StatefulPartitionedCall"dense_3400/StatefulPartitionedCall2H
"dense_3401/StatefulPartitionedCall"dense_3401/StatefulPartitionedCall2H
"dense_3402/StatefulPartitionedCall"dense_3402/StatefulPartitionedCall2H
"dense_3403/StatefulPartitionedCall"dense_3403/StatefulPartitionedCall2H
"dense_3404/StatefulPartitionedCall"dense_3404/StatefulPartitionedCall2H
"dense_3405/StatefulPartitionedCall"dense_3405/StatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
¶
h
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790938

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
¶
h
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790225

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
Ç
g
.__inference_dropout_1358_layer_call_fn_5791000

inputs
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790307*R
fMRK
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790296*
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

g
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790232

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
¶
h
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790886

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
ïB
ô	
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790690

inputs-
)dense_3399_matmul_readvariableop_resource.
*dense_3399_biasadd_readvariableop_resource-
)dense_3400_matmul_readvariableop_resource.
*dense_3400_biasadd_readvariableop_resource-
)dense_3401_matmul_readvariableop_resource.
*dense_3401_biasadd_readvariableop_resource-
)dense_3402_matmul_readvariableop_resource.
*dense_3402_biasadd_readvariableop_resource-
)dense_3403_matmul_readvariableop_resource.
*dense_3403_biasadd_readvariableop_resource-
)dense_3404_matmul_readvariableop_resource.
*dense_3404_biasadd_readvariableop_resource-
)dense_3405_matmul_readvariableop_resource.
*dense_3405_biasadd_readvariableop_resource
identity¢!dense_3399/BiasAdd/ReadVariableOp¢ dense_3399/MatMul/ReadVariableOp¢!dense_3400/BiasAdd/ReadVariableOp¢ dense_3400/MatMul/ReadVariableOp¢!dense_3401/BiasAdd/ReadVariableOp¢ dense_3401/MatMul/ReadVariableOp¢!dense_3402/BiasAdd/ReadVariableOp¢ dense_3402/MatMul/ReadVariableOp¢!dense_3403/BiasAdd/ReadVariableOp¢ dense_3403/MatMul/ReadVariableOp¢!dense_3404/BiasAdd/ReadVariableOp¢ dense_3404/MatMul/ReadVariableOp¢!dense_3405/BiasAdd/ReadVariableOp¢ dense_3405/MatMul/ReadVariableOp¹
 dense_3399/MatMul/ReadVariableOpReadVariableOp)dense_3399_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3399/MatMulMatMulinputs(dense_3399/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3399/BiasAdd/ReadVariableOpReadVariableOp*dense_3399_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3399/BiasAddBiasAdddense_3399/MatMul:product:0)dense_3399/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3400/MatMul/ReadVariableOpReadVariableOp)dense_3400_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3400/MatMulMatMuldense_3399/BiasAdd:output:0(dense_3400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3400/BiasAdd/ReadVariableOpReadVariableOp*dense_3400_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3400/BiasAddBiasAdddense_3400/MatMul:product:0)dense_3400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_1354/IdentityIdentitydense_3400/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3401/MatMul/ReadVariableOpReadVariableOp)dense_3401_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3401/MatMulMatMuldropout_1354/Identity:output:0(dense_3401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3401/BiasAdd/ReadVariableOpReadVariableOp*dense_3401_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3401/BiasAddBiasAdddense_3401/MatMul:product:0)dense_3401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_1355/IdentityIdentitydense_3401/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3402/MatMul/ReadVariableOpReadVariableOp)dense_3402_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3402/MatMulMatMuldropout_1355/Identity:output:0(dense_3402/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3402/BiasAdd/ReadVariableOpReadVariableOp*dense_3402_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3402/BiasAddBiasAdddense_3402/MatMul:product:0)dense_3402/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_1356/IdentityIdentitydense_3402/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3403/MatMul/ReadVariableOpReadVariableOp)dense_3403_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3403/MatMulMatMuldropout_1356/Identity:output:0(dense_3403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3403/BiasAdd/ReadVariableOpReadVariableOp*dense_3403_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3403/BiasAddBiasAdddense_3403/MatMul:product:0)dense_3403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_1357/IdentityIdentitydense_3403/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3404/MatMul/ReadVariableOpReadVariableOp)dense_3404_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3404/MatMulMatMuldropout_1357/Identity:output:0(dense_3404/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3404/BiasAdd/ReadVariableOpReadVariableOp*dense_3404_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3404/BiasAddBiasAdddense_3404/MatMul:product:0)dense_3404/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
dropout_1358/IdentityIdentitydense_3404/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 dense_3405/MatMul/ReadVariableOpReadVariableOp)dense_3405_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3405/MatMulMatMuldropout_1358/Identity:output:0(dense_3405/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!dense_3405/BiasAdd/ReadVariableOpReadVariableOp*dense_3405_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_3405/BiasAddBiasAdddense_3405/MatMul:product:0)dense_3405/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3405/ReluReludense_3405/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
IdentityIdentitydense_3405/Relu:activations:0"^dense_3399/BiasAdd/ReadVariableOp!^dense_3399/MatMul/ReadVariableOp"^dense_3400/BiasAdd/ReadVariableOp!^dense_3400/MatMul/ReadVariableOp"^dense_3401/BiasAdd/ReadVariableOp!^dense_3401/MatMul/ReadVariableOp"^dense_3402/BiasAdd/ReadVariableOp!^dense_3402/MatMul/ReadVariableOp"^dense_3403/BiasAdd/ReadVariableOp!^dense_3403/MatMul/ReadVariableOp"^dense_3404/BiasAdd/ReadVariableOp!^dense_3404/MatMul/ReadVariableOp"^dense_3405/BiasAdd/ReadVariableOp!^dense_3405/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2F
!dense_3400/BiasAdd/ReadVariableOp!dense_3400/BiasAdd/ReadVariableOp2F
!dense_3405/BiasAdd/ReadVariableOp!dense_3405/BiasAdd/ReadVariableOp2D
 dense_3403/MatMul/ReadVariableOp dense_3403/MatMul/ReadVariableOp2F
!dense_3403/BiasAdd/ReadVariableOp!dense_3403/BiasAdd/ReadVariableOp2D
 dense_3400/MatMul/ReadVariableOp dense_3400/MatMul/ReadVariableOp2D
 dense_3404/MatMul/ReadVariableOp dense_3404/MatMul/ReadVariableOp2F
!dense_3401/BiasAdd/ReadVariableOp!dense_3401/BiasAdd/ReadVariableOp2D
 dense_3399/MatMul/ReadVariableOp dense_3399/MatMul/ReadVariableOp2D
 dense_3401/MatMul/ReadVariableOp dense_3401/MatMul/ReadVariableOp2F
!dense_3404/BiasAdd/ReadVariableOp!dense_3404/BiasAdd/ReadVariableOp2D
 dense_3405/MatMul/ReadVariableOp dense_3405/MatMul/ReadVariableOp2F
!dense_3399/BiasAdd/ReadVariableOp!dense_3399/BiasAdd/ReadVariableOp2D
 dense_3402/MatMul/ReadVariableOp dense_3402/MatMul/ReadVariableOp2F
!dense_3402/BiasAdd/ReadVariableOp!dense_3402/BiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : : : :
 : : : : : :	 : 
Ç
g
.__inference_dropout_1354_layer_call_fn_5790792

inputs
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790023*R
fMRK
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790012*
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
â
­
,__inference_dense_3402_layer_call_fn_5790866

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790123*P
fKRI
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790117*
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

g
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790019

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

g
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790995

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
Õ	
à
G__inference_dense_3405_layer_call_and_return_conditional_losses_5790331

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
Ã
J
.__inference_dropout_1357_layer_call_fn_5790953

inputs
identity¡
PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790244*R
fMRK
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790232*
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

g
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790839

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
â
­
,__inference_dense_3400_layer_call_fn_5790762

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789981*P
fKRI
G__inference_dense_3400_layer_call_and_return_conditional_losses_5789975*
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
Ã
J
.__inference_dropout_1355_layer_call_fn_5790849

inputs
identity¡
PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790102*R
fMRK
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790090*
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
â
­
,__inference_dense_3401_layer_call_fn_5790814

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790052*P
fKRI
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790046*
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
¶
h
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790296

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
	
à
G__inference_dense_3399_layer_call_and_return_conditional_losses_5790738

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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
¶
h
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790012

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
¥
æ
0__inference_sequential_685_layer_call_fn_5790728

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*.
_gradient_op_typePartitionedCall-5790467*T
fORM
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790466*
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
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
	
à
G__inference_dense_3400_layer_call_and_return_conditional_losses_5790755

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
Õ	
à
G__inference_dense_3405_layer_call_and_return_conditional_losses_5791016

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
	
à
G__inference_dense_3399_layer_call_and_return_conditional_losses_5789948

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
:ÿÿÿÿÿÿÿÿÿ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
á
­
,__inference_dense_3399_layer_call_fn_5790745

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789954*P
fKRI
G__inference_dense_3399_layer_call_and_return_conditional_losses_5789948*
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
 
_user_specified_nameinputs: 

g
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790891

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
¶
h
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790782

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
Ç
g
.__inference_dropout_1355_layer_call_fn_5790844

inputs
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790094*R
fMRK
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790083*
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

g
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790303

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
¶
h
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790834

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
â
­
,__inference_dense_3404_layer_call_fn_5790970

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790265*P
fKRI
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790259*
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
æ]
Ó
 __inference__traced_save_5791198
file_prefix0
,savev2_dense_3399_kernel_read_readvariableop.
*savev2_dense_3399_bias_read_readvariableop0
,savev2_dense_3400_kernel_read_readvariableop.
*savev2_dense_3400_bias_read_readvariableop0
,savev2_dense_3401_kernel_read_readvariableop.
*savev2_dense_3401_bias_read_readvariableop0
,savev2_dense_3402_kernel_read_readvariableop.
*savev2_dense_3402_bias_read_readvariableop0
,savev2_dense_3403_kernel_read_readvariableop.
*savev2_dense_3403_bias_read_readvariableop0
,savev2_dense_3404_kernel_read_readvariableop.
*savev2_dense_3404_bias_read_readvariableop0
,savev2_dense_3405_kernel_read_readvariableop.
*savev2_dense_3405_bias_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	+
'savev2_nadam_beta_1_read_readvariableop+
'savev2_nadam_beta_2_read_readvariableop*
&savev2_nadam_decay_read_readvariableop2
.savev2_nadam_learning_rate_read_readvariableop3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop8
4savev2_nadam_dense_3399_kernel_m_read_readvariableop6
2savev2_nadam_dense_3399_bias_m_read_readvariableop8
4savev2_nadam_dense_3400_kernel_m_read_readvariableop6
2savev2_nadam_dense_3400_bias_m_read_readvariableop8
4savev2_nadam_dense_3401_kernel_m_read_readvariableop6
2savev2_nadam_dense_3401_bias_m_read_readvariableop8
4savev2_nadam_dense_3402_kernel_m_read_readvariableop6
2savev2_nadam_dense_3402_bias_m_read_readvariableop8
4savev2_nadam_dense_3403_kernel_m_read_readvariableop6
2savev2_nadam_dense_3403_bias_m_read_readvariableop8
4savev2_nadam_dense_3404_kernel_m_read_readvariableop6
2savev2_nadam_dense_3404_bias_m_read_readvariableop8
4savev2_nadam_dense_3405_kernel_m_read_readvariableop6
2savev2_nadam_dense_3405_bias_m_read_readvariableop8
4savev2_nadam_dense_3399_kernel_v_read_readvariableop6
2savev2_nadam_dense_3399_bias_v_read_readvariableop8
4savev2_nadam_dense_3400_kernel_v_read_readvariableop6
2savev2_nadam_dense_3400_bias_v_read_readvariableop8
4savev2_nadam_dense_3401_kernel_v_read_readvariableop6
2savev2_nadam_dense_3401_bias_v_read_readvariableop8
4savev2_nadam_dense_3402_kernel_v_read_readvariableop6
2savev2_nadam_dense_3402_bias_v_read_readvariableop8
4savev2_nadam_dense_3403_kernel_v_read_readvariableop6
2savev2_nadam_dense_3403_bias_v_read_readvariableop8
4savev2_nadam_dense_3404_kernel_v_read_readvariableop6
2savev2_nadam_dense_3404_bias_v_read_readvariableop8
4savev2_nadam_dense_3405_kernel_v_read_readvariableop6
2savev2_nadam_dense_3405_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_910124aa44904665975918cd12bbc287/part*
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
:2Û
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_3399_kernel_read_readvariableop*savev2_dense_3399_bias_read_readvariableop,savev2_dense_3400_kernel_read_readvariableop*savev2_dense_3400_bias_read_readvariableop,savev2_dense_3401_kernel_read_readvariableop*savev2_dense_3401_bias_read_readvariableop,savev2_dense_3402_kernel_read_readvariableop*savev2_dense_3402_bias_read_readvariableop,savev2_dense_3403_kernel_read_readvariableop*savev2_dense_3403_bias_read_readvariableop,savev2_dense_3404_kernel_read_readvariableop*savev2_dense_3404_bias_read_readvariableop,savev2_dense_3405_kernel_read_readvariableop*savev2_dense_3405_bias_read_readvariableop%savev2_nadam_iter_read_readvariableop'savev2_nadam_beta_1_read_readvariableop'savev2_nadam_beta_2_read_readvariableop&savev2_nadam_decay_read_readvariableop.savev2_nadam_learning_rate_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop4savev2_nadam_dense_3399_kernel_m_read_readvariableop2savev2_nadam_dense_3399_bias_m_read_readvariableop4savev2_nadam_dense_3400_kernel_m_read_readvariableop2savev2_nadam_dense_3400_bias_m_read_readvariableop4savev2_nadam_dense_3401_kernel_m_read_readvariableop2savev2_nadam_dense_3401_bias_m_read_readvariableop4savev2_nadam_dense_3402_kernel_m_read_readvariableop2savev2_nadam_dense_3402_bias_m_read_readvariableop4savev2_nadam_dense_3403_kernel_m_read_readvariableop2savev2_nadam_dense_3403_bias_m_read_readvariableop4savev2_nadam_dense_3404_kernel_m_read_readvariableop2savev2_nadam_dense_3404_bias_m_read_readvariableop4savev2_nadam_dense_3405_kernel_m_read_readvariableop2savev2_nadam_dense_3405_bias_m_read_readvariableop4savev2_nadam_dense_3399_kernel_v_read_readvariableop2savev2_nadam_dense_3399_bias_v_read_readvariableop4savev2_nadam_dense_3400_kernel_v_read_readvariableop2savev2_nadam_dense_3400_bias_v_read_readvariableop4savev2_nadam_dense_3401_kernel_v_read_readvariableop2savev2_nadam_dense_3401_bias_v_read_readvariableop4savev2_nadam_dense_3402_kernel_v_read_readvariableop2savev2_nadam_dense_3402_bias_v_read_readvariableop4savev2_nadam_dense_3403_kernel_v_read_readvariableop2savev2_nadam_dense_3403_bias_v_read_readvariableop4savev2_nadam_dense_3404_kernel_v_read_readvariableop2savev2_nadam_dense_3404_bias_v_read_readvariableop4savev2_nadam_dense_3405_kernel_v_read_readvariableop2savev2_nadam_dense_3405_bias_v_read_readvariableop"/device:CPU:0*@
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
: :	::
::
::
::
::
::	:: : : : : : : : :	::
::
::
::
::
::	::	::
::
::
::
::
::	:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:$ : : :, : :
 : :' : : :/ : : : :& : : :. : : :! : : :) : : :1 :  : : :( : : :0 :# : :	 :+ : :+ '
%
_user_specified_namefile_prefix:3 :" : : :* :% : : :2 :- : : 
¤¿

#__inference__traced_restore_5791361
file_prefix&
"assignvariableop_dense_3399_kernel&
"assignvariableop_1_dense_3399_bias(
$assignvariableop_2_dense_3400_kernel&
"assignvariableop_3_dense_3400_bias(
$assignvariableop_4_dense_3401_kernel&
"assignvariableop_5_dense_3401_bias(
$assignvariableop_6_dense_3402_kernel&
"assignvariableop_7_dense_3402_bias(
$assignvariableop_8_dense_3403_kernel&
"assignvariableop_9_dense_3403_bias)
%assignvariableop_10_dense_3404_kernel'
#assignvariableop_11_dense_3404_bias)
%assignvariableop_12_dense_3405_kernel'
#assignvariableop_13_dense_3405_bias"
assignvariableop_14_nadam_iter$
 assignvariableop_15_nadam_beta_1$
 assignvariableop_16_nadam_beta_2#
assignvariableop_17_nadam_decay+
'assignvariableop_18_nadam_learning_rate,
(assignvariableop_19_nadam_momentum_cache
assignvariableop_20_total
assignvariableop_21_count1
-assignvariableop_22_nadam_dense_3399_kernel_m/
+assignvariableop_23_nadam_dense_3399_bias_m1
-assignvariableop_24_nadam_dense_3400_kernel_m/
+assignvariableop_25_nadam_dense_3400_bias_m1
-assignvariableop_26_nadam_dense_3401_kernel_m/
+assignvariableop_27_nadam_dense_3401_bias_m1
-assignvariableop_28_nadam_dense_3402_kernel_m/
+assignvariableop_29_nadam_dense_3402_bias_m1
-assignvariableop_30_nadam_dense_3403_kernel_m/
+assignvariableop_31_nadam_dense_3403_bias_m1
-assignvariableop_32_nadam_dense_3404_kernel_m/
+assignvariableop_33_nadam_dense_3404_bias_m1
-assignvariableop_34_nadam_dense_3405_kernel_m/
+assignvariableop_35_nadam_dense_3405_bias_m1
-assignvariableop_36_nadam_dense_3399_kernel_v/
+assignvariableop_37_nadam_dense_3399_bias_v1
-assignvariableop_38_nadam_dense_3400_kernel_v/
+assignvariableop_39_nadam_dense_3400_bias_v1
-assignvariableop_40_nadam_dense_3401_kernel_v/
+assignvariableop_41_nadam_dense_3401_bias_v1
-assignvariableop_42_nadam_dense_3402_kernel_v/
+assignvariableop_43_nadam_dense_3402_bias_v1
-assignvariableop_44_nadam_dense_3403_kernel_v/
+assignvariableop_45_nadam_dense_3403_bias_v1
-assignvariableop_46_nadam_dense_3404_kernel_v/
+assignvariableop_47_nadam_dense_3404_bias_v1
-assignvariableop_48_nadam_dense_3405_kernel_v/
+assignvariableop_49_nadam_dense_3405_bias_v
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
:~
AssignVariableOpAssignVariableOp"assignvariableop_dense_3399_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_3399_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_3400_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_3400_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_3401_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_3401_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp$assignvariableop_6_dense_3402_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp"assignvariableop_7_dense_3402_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp$assignvariableop_8_dense_3403_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp"assignvariableop_9_dense_3403_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp%assignvariableop_10_dense_3404_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp#assignvariableop_11_dense_3404_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp%assignvariableop_12_dense_3405_kernelIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp#assignvariableop_13_dense_3405_biasIdentity_13:output:0*
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
:
AssignVariableOp_22AssignVariableOp-assignvariableop_22_nadam_dense_3399_kernel_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_nadam_dense_3399_bias_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp-assignvariableop_24_nadam_dense_3400_kernel_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_nadam_dense_3400_bias_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp-assignvariableop_26_nadam_dense_3401_kernel_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_nadam_dense_3401_bias_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp-assignvariableop_28_nadam_dense_3402_kernel_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_nadam_dense_3402_bias_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp-assignvariableop_30_nadam_dense_3403_kernel_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_nadam_dense_3403_bias_mIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp-assignvariableop_32_nadam_dense_3404_kernel_mIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_nadam_dense_3404_bias_mIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp-assignvariableop_34_nadam_dense_3405_kernel_mIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_nadam_dense_3405_bias_mIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp-assignvariableop_36_nadam_dense_3399_kernel_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp+assignvariableop_37_nadam_dense_3399_bias_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp-assignvariableop_38_nadam_dense_3400_kernel_vIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp+assignvariableop_39_nadam_dense_3400_bias_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp-assignvariableop_40_nadam_dense_3401_kernel_vIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_nadam_dense_3401_bias_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp-assignvariableop_42_nadam_dense_3402_kernel_vIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp+assignvariableop_43_nadam_dense_3402_bias_vIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp-assignvariableop_44_nadam_dense_3403_kernel_vIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp+assignvariableop_45_nadam_dense_3403_bias_vIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp-assignvariableop_46_nadam_dense_3404_kernel_vIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp+assignvariableop_47_nadam_dense_3404_bias_vIdentity_47:output:0*
dtype0*
_output_shapes
 P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp-assignvariableop_48_nadam_dense_3405_kernel_vIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp+assignvariableop_49_nadam_dense_3405_bias_vIdentity_49:output:0*
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
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
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
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
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
	RestoreV2	RestoreV2: : :1 :  : : :( : : :0 :# : :	 :+ : :+ '
%
_user_specified_namefile_prefix:" : : :* :% : : :2 :- : : :$ : : :, : :
 : :' : : :/ : : : :& : : :. : : :! : : :) 
â
­
,__inference_dense_3403_layer_call_fn_5790918

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790194*P
fKRI
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790188*
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
	
à
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790117

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
Ã
ð
0__inference_sequential_685_layer_call_fn_5790432
dense_3399_input"
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_3399_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*.
_gradient_op_typePartitionedCall-5790415*T
fORM
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790414*
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
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :0 ,
*
_user_specified_namedense_3399_input: : : : :
 
¶
h
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790990

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
¶
h
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790083

inputs
identityQ
dropout/rateConst*
valueB
 *  >*
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
	
à
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790807

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
ÒT
ù
"__inference__wrapped_model_5789932
dense_3399_input<
8sequential_685_dense_3399_matmul_readvariableop_resource=
9sequential_685_dense_3399_biasadd_readvariableop_resource<
8sequential_685_dense_3400_matmul_readvariableop_resource=
9sequential_685_dense_3400_biasadd_readvariableop_resource<
8sequential_685_dense_3401_matmul_readvariableop_resource=
9sequential_685_dense_3401_biasadd_readvariableop_resource<
8sequential_685_dense_3402_matmul_readvariableop_resource=
9sequential_685_dense_3402_biasadd_readvariableop_resource<
8sequential_685_dense_3403_matmul_readvariableop_resource=
9sequential_685_dense_3403_biasadd_readvariableop_resource<
8sequential_685_dense_3404_matmul_readvariableop_resource=
9sequential_685_dense_3404_biasadd_readvariableop_resource<
8sequential_685_dense_3405_matmul_readvariableop_resource=
9sequential_685_dense_3405_biasadd_readvariableop_resource
identity¢0sequential_685/dense_3399/BiasAdd/ReadVariableOp¢/sequential_685/dense_3399/MatMul/ReadVariableOp¢0sequential_685/dense_3400/BiasAdd/ReadVariableOp¢/sequential_685/dense_3400/MatMul/ReadVariableOp¢0sequential_685/dense_3401/BiasAdd/ReadVariableOp¢/sequential_685/dense_3401/MatMul/ReadVariableOp¢0sequential_685/dense_3402/BiasAdd/ReadVariableOp¢/sequential_685/dense_3402/MatMul/ReadVariableOp¢0sequential_685/dense_3403/BiasAdd/ReadVariableOp¢/sequential_685/dense_3403/MatMul/ReadVariableOp¢0sequential_685/dense_3404/BiasAdd/ReadVariableOp¢/sequential_685/dense_3404/MatMul/ReadVariableOp¢0sequential_685/dense_3405/BiasAdd/ReadVariableOp¢/sequential_685/dense_3405/MatMul/ReadVariableOp×
/sequential_685/dense_3399/MatMul/ReadVariableOpReadVariableOp8sequential_685_dense_3399_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	¨
 sequential_685/dense_3399/MatMulMatMuldense_3399_input7sequential_685/dense_3399/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_685/dense_3399/BiasAdd/ReadVariableOpReadVariableOp9sequential_685_dense_3399_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_685/dense_3399/BiasAddBiasAdd*sequential_685/dense_3399/MatMul:product:08sequential_685/dense_3399/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
/sequential_685/dense_3400/MatMul/ReadVariableOpReadVariableOp8sequential_685_dense_3400_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Â
 sequential_685/dense_3400/MatMulMatMul*sequential_685/dense_3399/BiasAdd:output:07sequential_685/dense_3400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_685/dense_3400/BiasAdd/ReadVariableOpReadVariableOp9sequential_685_dense_3400_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_685/dense_3400/BiasAddBiasAdd*sequential_685/dense_3400/MatMul:product:08sequential_685/dense_3400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_685/dropout_1354/IdentityIdentity*sequential_685/dense_3400/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
/sequential_685/dense_3401/MatMul/ReadVariableOpReadVariableOp8sequential_685_dense_3401_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Å
 sequential_685/dense_3401/MatMulMatMul-sequential_685/dropout_1354/Identity:output:07sequential_685/dense_3401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_685/dense_3401/BiasAdd/ReadVariableOpReadVariableOp9sequential_685_dense_3401_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_685/dense_3401/BiasAddBiasAdd*sequential_685/dense_3401/MatMul:product:08sequential_685/dense_3401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_685/dropout_1355/IdentityIdentity*sequential_685/dense_3401/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
/sequential_685/dense_3402/MatMul/ReadVariableOpReadVariableOp8sequential_685_dense_3402_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Å
 sequential_685/dense_3402/MatMulMatMul-sequential_685/dropout_1355/Identity:output:07sequential_685/dense_3402/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_685/dense_3402/BiasAdd/ReadVariableOpReadVariableOp9sequential_685_dense_3402_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_685/dense_3402/BiasAddBiasAdd*sequential_685/dense_3402/MatMul:product:08sequential_685/dense_3402/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_685/dropout_1356/IdentityIdentity*sequential_685/dense_3402/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
/sequential_685/dense_3403/MatMul/ReadVariableOpReadVariableOp8sequential_685_dense_3403_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Å
 sequential_685/dense_3403/MatMulMatMul-sequential_685/dropout_1356/Identity:output:07sequential_685/dense_3403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_685/dense_3403/BiasAdd/ReadVariableOpReadVariableOp9sequential_685_dense_3403_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_685/dense_3403/BiasAddBiasAdd*sequential_685/dense_3403/MatMul:product:08sequential_685/dense_3403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_685/dropout_1357/IdentityIdentity*sequential_685/dense_3403/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
/sequential_685/dense_3404/MatMul/ReadVariableOpReadVariableOp8sequential_685_dense_3404_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Å
 sequential_685/dense_3404/MatMulMatMul-sequential_685/dropout_1357/Identity:output:07sequential_685/dense_3404/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
0sequential_685/dense_3404/BiasAdd/ReadVariableOpReadVariableOp9sequential_685_dense_3404_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Å
!sequential_685/dense_3404/BiasAddBiasAdd*sequential_685/dense_3404/MatMul:product:08sequential_685/dense_3404/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$sequential_685/dropout_1358/IdentityIdentity*sequential_685/dense_3404/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
/sequential_685/dense_3405/MatMul/ReadVariableOpReadVariableOp8sequential_685_dense_3405_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Ä
 sequential_685/dense_3405/MatMulMatMul-sequential_685/dropout_1358/Identity:output:07sequential_685/dense_3405/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
0sequential_685/dense_3405/BiasAdd/ReadVariableOpReadVariableOp9sequential_685_dense_3405_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ä
!sequential_685/dense_3405/BiasAddBiasAdd*sequential_685/dense_3405/MatMul:product:08sequential_685/dense_3405/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_685/dense_3405/ReluRelu*sequential_685/dense_3405/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
IdentityIdentity,sequential_685/dense_3405/Relu:activations:01^sequential_685/dense_3399/BiasAdd/ReadVariableOp0^sequential_685/dense_3399/MatMul/ReadVariableOp1^sequential_685/dense_3400/BiasAdd/ReadVariableOp0^sequential_685/dense_3400/MatMul/ReadVariableOp1^sequential_685/dense_3401/BiasAdd/ReadVariableOp0^sequential_685/dense_3401/MatMul/ReadVariableOp1^sequential_685/dense_3402/BiasAdd/ReadVariableOp0^sequential_685/dense_3402/MatMul/ReadVariableOp1^sequential_685/dense_3403/BiasAdd/ReadVariableOp0^sequential_685/dense_3403/MatMul/ReadVariableOp1^sequential_685/dense_3404/BiasAdd/ReadVariableOp0^sequential_685/dense_3404/MatMul/ReadVariableOp1^sequential_685/dense_3405/BiasAdd/ReadVariableOp0^sequential_685/dense_3405/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2b
/sequential_685/dense_3400/MatMul/ReadVariableOp/sequential_685/dense_3400/MatMul/ReadVariableOp2d
0sequential_685/dense_3399/BiasAdd/ReadVariableOp0sequential_685/dense_3399/BiasAdd/ReadVariableOp2d
0sequential_685/dense_3402/BiasAdd/ReadVariableOp0sequential_685/dense_3402/BiasAdd/ReadVariableOp2b
/sequential_685/dense_3404/MatMul/ReadVariableOp/sequential_685/dense_3404/MatMul/ReadVariableOp2b
/sequential_685/dense_3399/MatMul/ReadVariableOp/sequential_685/dense_3399/MatMul/ReadVariableOp2d
0sequential_685/dense_3400/BiasAdd/ReadVariableOp0sequential_685/dense_3400/BiasAdd/ReadVariableOp2d
0sequential_685/dense_3405/BiasAdd/ReadVariableOp0sequential_685/dense_3405/BiasAdd/ReadVariableOp2b
/sequential_685/dense_3401/MatMul/ReadVariableOp/sequential_685/dense_3401/MatMul/ReadVariableOp2b
/sequential_685/dense_3405/MatMul/ReadVariableOp/sequential_685/dense_3405/MatMul/ReadVariableOp2d
0sequential_685/dense_3403/BiasAdd/ReadVariableOp0sequential_685/dense_3403/BiasAdd/ReadVariableOp2b
/sequential_685/dense_3402/MatMul/ReadVariableOp/sequential_685/dense_3402/MatMul/ReadVariableOp2d
0sequential_685/dense_3401/BiasAdd/ReadVariableOp0sequential_685/dense_3401/BiasAdd/ReadVariableOp2b
/sequential_685/dense_3403/MatMul/ReadVariableOp/sequential_685/dense_3403/MatMul/ReadVariableOp2d
0sequential_685/dense_3404/BiasAdd/ReadVariableOp0sequential_685/dense_3404/BiasAdd/ReadVariableOp: : : : : :	 : : : :0 ,
*
_user_specified_namedense_3399_input: : : : :
 

å
%__inference_signature_wrapper_5790509
dense_3399_input"
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
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCalldense_3399_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*.
_gradient_op_typePartitionedCall-5790492*+
f&R$
"__inference__wrapped_model_5789932*
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
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :0 ,
*
_user_specified_namedense_3399_input: : : : :
 
	
à
G__inference_dense_3400_layer_call_and_return_conditional_losses_5789975

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
Ç
g
.__inference_dropout_1357_layer_call_fn_5790948

inputs
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790236*R
fMRK
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790225*
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
ü5
ÿ
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790466

inputs-
)dense_3399_statefulpartitionedcall_args_1-
)dense_3399_statefulpartitionedcall_args_2-
)dense_3400_statefulpartitionedcall_args_1-
)dense_3400_statefulpartitionedcall_args_2-
)dense_3401_statefulpartitionedcall_args_1-
)dense_3401_statefulpartitionedcall_args_2-
)dense_3402_statefulpartitionedcall_args_1-
)dense_3402_statefulpartitionedcall_args_2-
)dense_3403_statefulpartitionedcall_args_1-
)dense_3403_statefulpartitionedcall_args_2-
)dense_3404_statefulpartitionedcall_args_1-
)dense_3404_statefulpartitionedcall_args_2-
)dense_3405_statefulpartitionedcall_args_1-
)dense_3405_statefulpartitionedcall_args_2
identity¢"dense_3399/StatefulPartitionedCall¢"dense_3400/StatefulPartitionedCall¢"dense_3401/StatefulPartitionedCall¢"dense_3402/StatefulPartitionedCall¢"dense_3403/StatefulPartitionedCall¢"dense_3404/StatefulPartitionedCall¢"dense_3405/StatefulPartitionedCall
"dense_3399/StatefulPartitionedCallStatefulPartitionedCallinputs)dense_3399_statefulpartitionedcall_args_1)dense_3399_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789954*P
fKRI
G__inference_dense_3399_layer_call_and_return_conditional_losses_5789948*
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
"dense_3400/StatefulPartitionedCallStatefulPartitionedCall+dense_3399/StatefulPartitionedCall:output:0)dense_3400_statefulpartitionedcall_args_1)dense_3400_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789981*P
fKRI
G__inference_dense_3400_layer_call_and_return_conditional_losses_5789975*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1354/PartitionedCallPartitionedCall+dense_3400/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790031*R
fMRK
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790019*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3401/StatefulPartitionedCallStatefulPartitionedCall%dropout_1354/PartitionedCall:output:0)dense_3401_statefulpartitionedcall_args_1)dense_3401_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790052*P
fKRI
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790046*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1355/PartitionedCallPartitionedCall+dense_3401/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790102*R
fMRK
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790090*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3402/StatefulPartitionedCallStatefulPartitionedCall%dropout_1355/PartitionedCall:output:0)dense_3402_statefulpartitionedcall_args_1)dense_3402_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790123*P
fKRI
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790117*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1356/PartitionedCallPartitionedCall+dense_3402/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790173*R
fMRK
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790161*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3403/StatefulPartitionedCallStatefulPartitionedCall%dropout_1356/PartitionedCall:output:0)dense_3403_statefulpartitionedcall_args_1)dense_3403_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790194*P
fKRI
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790188*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1357/PartitionedCallPartitionedCall+dense_3403/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790244*R
fMRK
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790232*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3404/StatefulPartitionedCallStatefulPartitionedCall%dropout_1357/PartitionedCall:output:0)dense_3404_statefulpartitionedcall_args_1)dense_3404_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790265*P
fKRI
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790259*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1358/PartitionedCallPartitionedCall+dense_3404/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790315*R
fMRK
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790303*
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
:ÿÿÿÿÿÿÿÿÿ°
"dense_3405/StatefulPartitionedCallStatefulPartitionedCall%dropout_1358/PartitionedCall:output:0)dense_3405_statefulpartitionedcall_args_1)dense_3405_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790337*P
fKRI
G__inference_dense_3405_layer_call_and_return_conditional_losses_5790331*
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
:ÿÿÿÿÿÿÿÿÿö
IdentityIdentity+dense_3405/StatefulPartitionedCall:output:0#^dense_3399/StatefulPartitionedCall#^dense_3400/StatefulPartitionedCall#^dense_3401/StatefulPartitionedCall#^dense_3402/StatefulPartitionedCall#^dense_3403/StatefulPartitionedCall#^dense_3404/StatefulPartitionedCall#^dense_3405/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"dense_3401/StatefulPartitionedCall"dense_3401/StatefulPartitionedCall2H
"dense_3402/StatefulPartitionedCall"dense_3402/StatefulPartitionedCall2H
"dense_3403/StatefulPartitionedCall"dense_3403/StatefulPartitionedCall2H
"dense_3404/StatefulPartitionedCall"dense_3404/StatefulPartitionedCall2H
"dense_3405/StatefulPartitionedCall"dense_3405/StatefulPartitionedCall2H
"dense_3399/StatefulPartitionedCall"dense_3399/StatefulPartitionedCall2H
"dense_3400/StatefulPartitionedCall"dense_3400/StatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 

g
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790943

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
¥
æ
0__inference_sequential_685_layer_call_fn_5790709

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14*.
_gradient_op_typePartitionedCall-5790415*T
fORM
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790414*
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
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 

g
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790090

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
	
à
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790188

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
	
à
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790911

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
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790259

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
Ç
g
.__inference_dropout_1356_layer_call_fn_5790896

inputs
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790165*R
fMRK
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790154*
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

g
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790787

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

g
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790161

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
Ã
J
.__inference_dropout_1354_layer_call_fn_5790797

inputs
identity¡
PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790031*R
fMRK
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790019*
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
ü
ô	
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790638

inputs-
)dense_3399_matmul_readvariableop_resource.
*dense_3399_biasadd_readvariableop_resource-
)dense_3400_matmul_readvariableop_resource.
*dense_3400_biasadd_readvariableop_resource-
)dense_3401_matmul_readvariableop_resource.
*dense_3401_biasadd_readvariableop_resource-
)dense_3402_matmul_readvariableop_resource.
*dense_3402_biasadd_readvariableop_resource-
)dense_3403_matmul_readvariableop_resource.
*dense_3403_biasadd_readvariableop_resource-
)dense_3404_matmul_readvariableop_resource.
*dense_3404_biasadd_readvariableop_resource-
)dense_3405_matmul_readvariableop_resource.
*dense_3405_biasadd_readvariableop_resource
identity¢!dense_3399/BiasAdd/ReadVariableOp¢ dense_3399/MatMul/ReadVariableOp¢!dense_3400/BiasAdd/ReadVariableOp¢ dense_3400/MatMul/ReadVariableOp¢!dense_3401/BiasAdd/ReadVariableOp¢ dense_3401/MatMul/ReadVariableOp¢!dense_3402/BiasAdd/ReadVariableOp¢ dense_3402/MatMul/ReadVariableOp¢!dense_3403/BiasAdd/ReadVariableOp¢ dense_3403/MatMul/ReadVariableOp¢!dense_3404/BiasAdd/ReadVariableOp¢ dense_3404/MatMul/ReadVariableOp¢!dense_3405/BiasAdd/ReadVariableOp¢ dense_3405/MatMul/ReadVariableOp¹
 dense_3399/MatMul/ReadVariableOpReadVariableOp)dense_3399_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3399/MatMulMatMulinputs(dense_3399/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3399/BiasAdd/ReadVariableOpReadVariableOp*dense_3399_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3399/BiasAddBiasAdddense_3399/MatMul:product:0)dense_3399/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3400/MatMul/ReadVariableOpReadVariableOp)dense_3400_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3400/MatMulMatMuldense_3399/BiasAdd:output:0(dense_3400/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3400/BiasAdd/ReadVariableOpReadVariableOp*dense_3400_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3400/BiasAddBiasAdddense_3400/MatMul:product:0)dense_3400/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dropout_1354/dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: e
dropout_1354/dropout/ShapeShapedense_3400/BiasAdd:output:0*
T0*
_output_shapes
:l
'dropout_1354/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_1354/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: §
1dropout_1354/dropout/random_uniform/RandomUniformRandomUniform#dropout_1354/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
'dropout_1354/dropout/random_uniform/subSub0dropout_1354/dropout/random_uniform/max:output:00dropout_1354/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ê
'dropout_1354/dropout/random_uniform/mulMul:dropout_1354/dropout/random_uniform/RandomUniform:output:0+dropout_1354/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
#dropout_1354/dropout/random_uniformAdd+dropout_1354/dropout/random_uniform/mul:z:00dropout_1354/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_1354/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1354/dropout/subSub#dropout_1354/dropout/sub/x:output:0"dropout_1354/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_1354/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1354/dropout/truedivRealDiv'dropout_1354/dropout/truediv/x:output:0dropout_1354/dropout/sub:z:0*
T0*
_output_shapes
: ±
!dropout_1354/dropout/GreaterEqualGreaterEqual'dropout_1354/dropout/random_uniform:z:0"dropout_1354/dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1354/dropout/mulMuldense_3400/BiasAdd:output:0 dropout_1354/dropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1354/dropout/CastCast%dropout_1354/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1354/dropout/mul_1Muldropout_1354/dropout/mul:z:0dropout_1354/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3401/MatMul/ReadVariableOpReadVariableOp)dense_3401_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3401/MatMulMatMuldropout_1354/dropout/mul_1:z:0(dense_3401/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3401/BiasAdd/ReadVariableOpReadVariableOp*dense_3401_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3401/BiasAddBiasAdddense_3401/MatMul:product:0)dense_3401/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dropout_1355/dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: e
dropout_1355/dropout/ShapeShapedense_3401/BiasAdd:output:0*
T0*
_output_shapes
:l
'dropout_1355/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_1355/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: §
1dropout_1355/dropout/random_uniform/RandomUniformRandomUniform#dropout_1355/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
'dropout_1355/dropout/random_uniform/subSub0dropout_1355/dropout/random_uniform/max:output:00dropout_1355/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ê
'dropout_1355/dropout/random_uniform/mulMul:dropout_1355/dropout/random_uniform/RandomUniform:output:0+dropout_1355/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
#dropout_1355/dropout/random_uniformAdd+dropout_1355/dropout/random_uniform/mul:z:00dropout_1355/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_1355/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1355/dropout/subSub#dropout_1355/dropout/sub/x:output:0"dropout_1355/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_1355/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1355/dropout/truedivRealDiv'dropout_1355/dropout/truediv/x:output:0dropout_1355/dropout/sub:z:0*
T0*
_output_shapes
: ±
!dropout_1355/dropout/GreaterEqualGreaterEqual'dropout_1355/dropout/random_uniform:z:0"dropout_1355/dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1355/dropout/mulMuldense_3401/BiasAdd:output:0 dropout_1355/dropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1355/dropout/CastCast%dropout_1355/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1355/dropout/mul_1Muldropout_1355/dropout/mul:z:0dropout_1355/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3402/MatMul/ReadVariableOpReadVariableOp)dense_3402_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3402/MatMulMatMuldropout_1355/dropout/mul_1:z:0(dense_3402/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3402/BiasAdd/ReadVariableOpReadVariableOp*dense_3402_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3402/BiasAddBiasAdddense_3402/MatMul:product:0)dense_3402/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dropout_1356/dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: e
dropout_1356/dropout/ShapeShapedense_3402/BiasAdd:output:0*
T0*
_output_shapes
:l
'dropout_1356/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_1356/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: §
1dropout_1356/dropout/random_uniform/RandomUniformRandomUniform#dropout_1356/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
'dropout_1356/dropout/random_uniform/subSub0dropout_1356/dropout/random_uniform/max:output:00dropout_1356/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ê
'dropout_1356/dropout/random_uniform/mulMul:dropout_1356/dropout/random_uniform/RandomUniform:output:0+dropout_1356/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
#dropout_1356/dropout/random_uniformAdd+dropout_1356/dropout/random_uniform/mul:z:00dropout_1356/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_1356/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1356/dropout/subSub#dropout_1356/dropout/sub/x:output:0"dropout_1356/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_1356/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1356/dropout/truedivRealDiv'dropout_1356/dropout/truediv/x:output:0dropout_1356/dropout/sub:z:0*
T0*
_output_shapes
: ±
!dropout_1356/dropout/GreaterEqualGreaterEqual'dropout_1356/dropout/random_uniform:z:0"dropout_1356/dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1356/dropout/mulMuldense_3402/BiasAdd:output:0 dropout_1356/dropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1356/dropout/CastCast%dropout_1356/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1356/dropout/mul_1Muldropout_1356/dropout/mul:z:0dropout_1356/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3403/MatMul/ReadVariableOpReadVariableOp)dense_3403_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3403/MatMulMatMuldropout_1356/dropout/mul_1:z:0(dense_3403/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3403/BiasAdd/ReadVariableOpReadVariableOp*dense_3403_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3403/BiasAddBiasAdddense_3403/MatMul:product:0)dense_3403/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dropout_1357/dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: e
dropout_1357/dropout/ShapeShapedense_3403/BiasAdd:output:0*
T0*
_output_shapes
:l
'dropout_1357/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_1357/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: §
1dropout_1357/dropout/random_uniform/RandomUniformRandomUniform#dropout_1357/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
'dropout_1357/dropout/random_uniform/subSub0dropout_1357/dropout/random_uniform/max:output:00dropout_1357/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ê
'dropout_1357/dropout/random_uniform/mulMul:dropout_1357/dropout/random_uniform/RandomUniform:output:0+dropout_1357/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
#dropout_1357/dropout/random_uniformAdd+dropout_1357/dropout/random_uniform/mul:z:00dropout_1357/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_1357/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1357/dropout/subSub#dropout_1357/dropout/sub/x:output:0"dropout_1357/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_1357/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1357/dropout/truedivRealDiv'dropout_1357/dropout/truediv/x:output:0dropout_1357/dropout/sub:z:0*
T0*
_output_shapes
: ±
!dropout_1357/dropout/GreaterEqualGreaterEqual'dropout_1357/dropout/random_uniform:z:0"dropout_1357/dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1357/dropout/mulMuldense_3403/BiasAdd:output:0 dropout_1357/dropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1357/dropout/CastCast%dropout_1357/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1357/dropout/mul_1Muldropout_1357/dropout/mul:z:0dropout_1357/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 dense_3404/MatMul/ReadVariableOpReadVariableOp)dense_3404_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense_3404/MatMulMatMuldropout_1357/dropout/mul_1:z:0(dense_3404/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
!dense_3404/BiasAdd/ReadVariableOpReadVariableOp*dense_3404_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense_3404/BiasAddBiasAdddense_3404/MatMul:product:0)dense_3404/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
dropout_1358/dropout/rateConst*
valueB
 *  >*
dtype0*
_output_shapes
: e
dropout_1358/dropout/ShapeShapedense_3404/BiasAdd:output:0*
T0*
_output_shapes
:l
'dropout_1358/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: l
'dropout_1358/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: §
1dropout_1358/dropout/random_uniform/RandomUniformRandomUniform#dropout_1358/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
'dropout_1358/dropout/random_uniform/subSub0dropout_1358/dropout/random_uniform/max:output:00dropout_1358/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ê
'dropout_1358/dropout/random_uniform/mulMul:dropout_1358/dropout/random_uniform/RandomUniform:output:0+dropout_1358/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
#dropout_1358/dropout/random_uniformAdd+dropout_1358/dropout/random_uniform/mul:z:00dropout_1358/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
dropout_1358/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1358/dropout/subSub#dropout_1358/dropout/sub/x:output:0"dropout_1358/dropout/rate:output:0*
T0*
_output_shapes
: c
dropout_1358/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout_1358/dropout/truedivRealDiv'dropout_1358/dropout/truediv/x:output:0dropout_1358/dropout/sub:z:0*
T0*
_output_shapes
: ±
!dropout_1358/dropout/GreaterEqualGreaterEqual'dropout_1358/dropout/random_uniform:z:0"dropout_1358/dropout/rate:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1358/dropout/mulMuldense_3404/BiasAdd:output:0 dropout_1358/dropout/truediv:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1358/dropout/CastCast%dropout_1358/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1358/dropout/mul_1Muldropout_1358/dropout/mul:z:0dropout_1358/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 dense_3405/MatMul/ReadVariableOpReadVariableOp)dense_3405_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
dense_3405/MatMulMatMuldropout_1358/dropout/mul_1:z:0(dense_3405/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶
!dense_3405/BiasAdd/ReadVariableOpReadVariableOp*dense_3405_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
dense_3405/BiasAddBiasAdddense_3405/MatMul:product:0)dense_3405/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
dense_3405/ReluReludense_3405/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
IdentityIdentitydense_3405/Relu:activations:0"^dense_3399/BiasAdd/ReadVariableOp!^dense_3399/MatMul/ReadVariableOp"^dense_3400/BiasAdd/ReadVariableOp!^dense_3400/MatMul/ReadVariableOp"^dense_3401/BiasAdd/ReadVariableOp!^dense_3401/MatMul/ReadVariableOp"^dense_3402/BiasAdd/ReadVariableOp!^dense_3402/MatMul/ReadVariableOp"^dense_3403/BiasAdd/ReadVariableOp!^dense_3403/MatMul/ReadVariableOp"^dense_3404/BiasAdd/ReadVariableOp!^dense_3404/MatMul/ReadVariableOp"^dense_3405/BiasAdd/ReadVariableOp!^dense_3405/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2D
 dense_3404/MatMul/ReadVariableOp dense_3404/MatMul/ReadVariableOp2F
!dense_3401/BiasAdd/ReadVariableOp!dense_3401/BiasAdd/ReadVariableOp2D
 dense_3399/MatMul/ReadVariableOp dense_3399/MatMul/ReadVariableOp2D
 dense_3401/MatMul/ReadVariableOp dense_3401/MatMul/ReadVariableOp2F
!dense_3404/BiasAdd/ReadVariableOp!dense_3404/BiasAdd/ReadVariableOp2D
 dense_3405/MatMul/ReadVariableOp dense_3405/MatMul/ReadVariableOp2F
!dense_3399/BiasAdd/ReadVariableOp!dense_3399/BiasAdd/ReadVariableOp2F
!dense_3402/BiasAdd/ReadVariableOp!dense_3402/BiasAdd/ReadVariableOp2D
 dense_3402/MatMul/ReadVariableOp dense_3402/MatMul/ReadVariableOp2F
!dense_3400/BiasAdd/ReadVariableOp!dense_3400/BiasAdd/ReadVariableOp2F
!dense_3405/BiasAdd/ReadVariableOp!dense_3405/BiasAdd/ReadVariableOp2D
 dense_3403/MatMul/ReadVariableOp dense_3403/MatMul/ReadVariableOp2F
!dense_3403/BiasAdd/ReadVariableOp!dense_3403/BiasAdd/ReadVariableOp2D
 dense_3400/MatMul/ReadVariableOp dense_3400/MatMul/ReadVariableOp: : : : : :	 : : : :& "
 
_user_specified_nameinputs: : : : :
 
6

K__inference_sequential_685_layer_call_and_return_conditional_losses_5790381
dense_3399_input-
)dense_3399_statefulpartitionedcall_args_1-
)dense_3399_statefulpartitionedcall_args_2-
)dense_3400_statefulpartitionedcall_args_1-
)dense_3400_statefulpartitionedcall_args_2-
)dense_3401_statefulpartitionedcall_args_1-
)dense_3401_statefulpartitionedcall_args_2-
)dense_3402_statefulpartitionedcall_args_1-
)dense_3402_statefulpartitionedcall_args_2-
)dense_3403_statefulpartitionedcall_args_1-
)dense_3403_statefulpartitionedcall_args_2-
)dense_3404_statefulpartitionedcall_args_1-
)dense_3404_statefulpartitionedcall_args_2-
)dense_3405_statefulpartitionedcall_args_1-
)dense_3405_statefulpartitionedcall_args_2
identity¢"dense_3399/StatefulPartitionedCall¢"dense_3400/StatefulPartitionedCall¢"dense_3401/StatefulPartitionedCall¢"dense_3402/StatefulPartitionedCall¢"dense_3403/StatefulPartitionedCall¢"dense_3404/StatefulPartitionedCall¢"dense_3405/StatefulPartitionedCall
"dense_3399/StatefulPartitionedCallStatefulPartitionedCalldense_3399_input)dense_3399_statefulpartitionedcall_args_1)dense_3399_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789954*P
fKRI
G__inference_dense_3399_layer_call_and_return_conditional_losses_5789948*
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
"dense_3400/StatefulPartitionedCallStatefulPartitionedCall+dense_3399/StatefulPartitionedCall:output:0)dense_3400_statefulpartitionedcall_args_1)dense_3400_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5789981*P
fKRI
G__inference_dense_3400_layer_call_and_return_conditional_losses_5789975*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1354/PartitionedCallPartitionedCall+dense_3400/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790031*R
fMRK
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790019*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3401/StatefulPartitionedCallStatefulPartitionedCall%dropout_1354/PartitionedCall:output:0)dense_3401_statefulpartitionedcall_args_1)dense_3401_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790052*P
fKRI
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790046*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1355/PartitionedCallPartitionedCall+dense_3401/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790102*R
fMRK
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790090*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3402/StatefulPartitionedCallStatefulPartitionedCall%dropout_1355/PartitionedCall:output:0)dense_3402_statefulpartitionedcall_args_1)dense_3402_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790123*P
fKRI
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790117*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1356/PartitionedCallPartitionedCall+dense_3402/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790173*R
fMRK
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790161*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3403/StatefulPartitionedCallStatefulPartitionedCall%dropout_1356/PartitionedCall:output:0)dense_3403_statefulpartitionedcall_args_1)dense_3403_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790194*P
fKRI
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790188*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1357/PartitionedCallPartitionedCall+dense_3403/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790244*R
fMRK
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790232*
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
:ÿÿÿÿÿÿÿÿÿ±
"dense_3404/StatefulPartitionedCallStatefulPartitionedCall%dropout_1357/PartitionedCall:output:0)dense_3404_statefulpartitionedcall_args_1)dense_3404_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790265*P
fKRI
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790259*
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
:ÿÿÿÿÿÿÿÿÿÓ
dropout_1358/PartitionedCallPartitionedCall+dense_3404/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-5790315*R
fMRK
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790303*
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
:ÿÿÿÿÿÿÿÿÿ°
"dense_3405/StatefulPartitionedCallStatefulPartitionedCall%dropout_1358/PartitionedCall:output:0)dense_3405_statefulpartitionedcall_args_1)dense_3405_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-5790337*P
fKRI
G__inference_dense_3405_layer_call_and_return_conditional_losses_5790331*
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
:ÿÿÿÿÿÿÿÿÿö
IdentityIdentity+dense_3405/StatefulPartitionedCall:output:0#^dense_3399/StatefulPartitionedCall#^dense_3400/StatefulPartitionedCall#^dense_3401/StatefulPartitionedCall#^dense_3402/StatefulPartitionedCall#^dense_3403/StatefulPartitionedCall#^dense_3404/StatefulPartitionedCall#^dense_3405/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*^
_input_shapesM
K:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"dense_3401/StatefulPartitionedCall"dense_3401/StatefulPartitionedCall2H
"dense_3402/StatefulPartitionedCall"dense_3402/StatefulPartitionedCall2H
"dense_3403/StatefulPartitionedCall"dense_3403/StatefulPartitionedCall2H
"dense_3404/StatefulPartitionedCall"dense_3404/StatefulPartitionedCall2H
"dense_3405/StatefulPartitionedCall"dense_3405/StatefulPartitionedCall2H
"dense_3399/StatefulPartitionedCall"dense_3399/StatefulPartitionedCall2H
"dense_3400/StatefulPartitionedCall"dense_3400/StatefulPartitionedCall: : : : : :	 : : : :0 ,
*
_user_specified_namedense_3399_input: : : : :
 
	
à
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790963

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
Ã
J
.__inference_dropout_1356_layer_call_fn_5790901

inputs
identity¡
PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790173*R
fMRK
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790161*
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
Ã
J
.__inference_dropout_1358_layer_call_fn_5791005

inputs
identity¡
PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-5790315*R
fMRK
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790303*
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
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*¿
serving_default«
M
dense_3399_input9
"serving_default_dense_3399_input:0ÿÿÿÿÿÿÿÿÿ>

dense_34050
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ñà
ÑI
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
¼__call__
½_default_save_signature
+¾&call_and_return_all_conditional_losses"¡E
_tf_keras_sequentialE{"class_name": "Sequential", "name": "sequential_685", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_685", "layers": [{"class_name": "Dense", "config": {"name": "dense_3399", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3400", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1354", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3401", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1355", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3402", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1356", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3403", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1357", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3404", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1358", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3405", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_685", "layers": [{"class_name": "Dense", "config": {"name": "dense_3399", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_3400", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1354", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3401", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1355", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3402", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1356", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3403", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1357", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3404", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1358", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_3405", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mae", "metrics": ["mae"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.0010000000474974513, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
µ
regularization_losses
	variables
trainable_variables
	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"¤
_tf_keras_layer{"class_name": "InputLayer", "name": "dense_3399_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 8], "config": {"batch_input_shape": [null, 8], "dtype": "float32", "sparse": false, "name": "dense_3399_input"}}
¼

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layerû{"class_name": "Dense", "name": "dense_3399", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 8], "config": {"name": "dense_3399", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}}


kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "Dense", "name": "dense_3400", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3400", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¸
$regularization_losses
%	variables
&trainable_variables
'	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"§
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1354", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1354", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}


(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "Dense", "name": "dense_3401", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3401", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¸
.regularization_losses
/	variables
0trainable_variables
1	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"§
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1355", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1355", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}


2kernel
3bias
4regularization_losses
5	variables
6trainable_variables
7	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "Dense", "name": "dense_3402", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3402", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¸
8regularization_losses
9	variables
:trainable_variables
;	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"§
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1356", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1356", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}


<kernel
=bias
>regularization_losses
?	variables
@trainable_variables
A	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "Dense", "name": "dense_3403", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3403", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¸
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"§
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1357", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1357", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}


Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"ò
_tf_keras_layerØ{"class_name": "Dense", "name": "dense_3404", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3404", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¸
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"§
_tf_keras_layer{"class_name": "Dropout", "name": "dropout_1358", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1358", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}


Pkernel
Qbias
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"î
_tf_keras_layerÔ{"class_name": "Dense", "name": "dense_3405", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_3405", "trainable": true, "dtype": "float32", "units": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
ÿ
Viter

Wbeta_1

Xbeta_2
	Ydecay
Zlearning_rate
[momentum_cachem m¡m¢m£(m¤)m¥2m¦3m§<m¨=m©FmªGm«Pm¬Qm­v®v¯v°v±(v²)v³2v´3vµ<v¶=v·Fv¸Gv¹PvºQv»"
	optimizer
 "
trackable_list_wrapper

0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13"
trackable_list_wrapper

0
1
2
3
(4
)5
26
37
<8
=9
F10
G11
P12
Q13"
trackable_list_wrapper
»
regularization_losses

\layers
	variables
]metrics
^layer_regularization_losses
trainable_variables
_non_trainable_variables
¼__call__
½_default_save_signature
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
-
Ùserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

regularization_losses

`layers
	variables
ametrics
blayer_regularization_losses
trainable_variables
cnon_trainable_variables
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
$:"	2dense_3399/kernel
:2dense_3399/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

regularization_losses

dlayers
	variables
emetrics
flayer_regularization_losses
trainable_variables
gnon_trainable_variables
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_3400/kernel
:2dense_3400/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

 regularization_losses

hlayers
!	variables
imetrics
jlayer_regularization_losses
"trainable_variables
knon_trainable_variables
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

$regularization_losses

llayers
%	variables
mmetrics
nlayer_regularization_losses
&trainable_variables
onon_trainable_variables
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_3401/kernel
:2dense_3401/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper

*regularization_losses

players
+	variables
qmetrics
rlayer_regularization_losses
,trainable_variables
snon_trainable_variables
Ç__call__
+È&call_and_return_all_conditional_losses
'È"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

.regularization_losses

tlayers
/	variables
umetrics
vlayer_regularization_losses
0trainable_variables
wnon_trainable_variables
É__call__
+Ê&call_and_return_all_conditional_losses
'Ê"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_3402/kernel
:2dense_3402/bias
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper

4regularization_losses

xlayers
5	variables
ymetrics
zlayer_regularization_losses
6trainable_variables
{non_trainable_variables
Ë__call__
+Ì&call_and_return_all_conditional_losses
'Ì"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

8regularization_losses

|layers
9	variables
}metrics
~layer_regularization_losses
:trainable_variables
non_trainable_variables
Í__call__
+Î&call_and_return_all_conditional_losses
'Î"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_3403/kernel
:2dense_3403/bias
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
¡
>regularization_losses
layers
?	variables
metrics
 layer_regularization_losses
@trainable_variables
non_trainable_variables
Ï__call__
+Ð&call_and_return_all_conditional_losses
'Ð"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡
Bregularization_losses
layers
C	variables
metrics
 layer_regularization_losses
Dtrainable_variables
non_trainable_variables
Ñ__call__
+Ò&call_and_return_all_conditional_losses
'Ò"call_and_return_conditional_losses"
_generic_user_object
%:#
2dense_3404/kernel
:2dense_3404/bias
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
¡
Hregularization_losses
layers
I	variables
metrics
 layer_regularization_losses
Jtrainable_variables
non_trainable_variables
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¡
Lregularization_losses
layers
M	variables
metrics
 layer_regularization_losses
Ntrainable_variables
non_trainable_variables
Õ__call__
+Ö&call_and_return_all_conditional_losses
'Ö"call_and_return_conditional_losses"
_generic_user_object
$:"	2dense_3405/kernel
:2dense_3405/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
¡
Rregularization_losses
layers
S	variables
metrics
 layer_regularization_losses
Ttrainable_variables
non_trainable_variables
×__call__
+Ø&call_and_return_all_conditional_losses
'Ø"call_and_return_conditional_losses"
_generic_user_object
:	 (2
Nadam/iter
: (2Nadam/beta_1
: (2Nadam/beta_2
: (2Nadam/decay
: (2Nadam/learning_rate
: 2Nadam/momentum_cache
v
0
1
2
3
4
5
6
	7

8
9
10
11"
trackable_list_wrapper
(
0"
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


total

count

_fn_kwargs
regularization_losses
	variables
trainable_variables
	keras_api
Ú__call__
+Û&call_and_return_all_conditional_losses"Û
_tf_keras_layerÁ{"class_name": "MeanMetricWrapper", "name": "mae", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "mae", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¤
regularization_losses
layers
	variables
metrics
 layer_regularization_losses
trainable_variables
non_trainable_variables
Ú__call__
+Û&call_and_return_all_conditional_losses
'Û"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
*:(	2Nadam/dense_3399/kernel/m
$:"2Nadam/dense_3399/bias/m
+:)
2Nadam/dense_3400/kernel/m
$:"2Nadam/dense_3400/bias/m
+:)
2Nadam/dense_3401/kernel/m
$:"2Nadam/dense_3401/bias/m
+:)
2Nadam/dense_3402/kernel/m
$:"2Nadam/dense_3402/bias/m
+:)
2Nadam/dense_3403/kernel/m
$:"2Nadam/dense_3403/bias/m
+:)
2Nadam/dense_3404/kernel/m
$:"2Nadam/dense_3404/bias/m
*:(	2Nadam/dense_3405/kernel/m
#:!2Nadam/dense_3405/bias/m
*:(	2Nadam/dense_3399/kernel/v
$:"2Nadam/dense_3399/bias/v
+:)
2Nadam/dense_3400/kernel/v
$:"2Nadam/dense_3400/bias/v
+:)
2Nadam/dense_3401/kernel/v
$:"2Nadam/dense_3401/bias/v
+:)
2Nadam/dense_3402/kernel/v
$:"2Nadam/dense_3402/bias/v
+:)
2Nadam/dense_3403/kernel/v
$:"2Nadam/dense_3403/bias/v
+:)
2Nadam/dense_3404/kernel/v
$:"2Nadam/dense_3404/bias/v
*:(	2Nadam/dense_3405/kernel/v
#:!2Nadam/dense_3405/bias/v
2
0__inference_sequential_685_layer_call_fn_5790709
0__inference_sequential_685_layer_call_fn_5790432
0__inference_sequential_685_layer_call_fn_5790484
0__inference_sequential_685_layer_call_fn_5790728À
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
"__inference__wrapped_model_5789932¿
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
dense_3399_inputÿÿÿÿÿÿÿÿÿ
ú2÷
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790690
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790638
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790381
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790349À
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
,__inference_dense_3399_layer_call_fn_5790745¢
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
G__inference_dense_3399_layer_call_and_return_conditional_losses_5790738¢
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
,__inference_dense_3400_layer_call_fn_5790762¢
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
G__inference_dense_3400_layer_call_and_return_conditional_losses_5790755¢
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
2
.__inference_dropout_1354_layer_call_fn_5790792
.__inference_dropout_1354_layer_call_fn_5790797´
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
Ð2Í
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790787
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790782´
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
Ö2Ó
,__inference_dense_3401_layer_call_fn_5790814¢
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
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790807¢
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
2
.__inference_dropout_1355_layer_call_fn_5790844
.__inference_dropout_1355_layer_call_fn_5790849´
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
Ð2Í
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790834
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790839´
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
Ö2Ó
,__inference_dense_3402_layer_call_fn_5790866¢
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
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790859¢
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
2
.__inference_dropout_1356_layer_call_fn_5790896
.__inference_dropout_1356_layer_call_fn_5790901´
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
Ð2Í
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790891
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790886´
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
Ö2Ó
,__inference_dense_3403_layer_call_fn_5790918¢
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
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790911¢
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
2
.__inference_dropout_1357_layer_call_fn_5790948
.__inference_dropout_1357_layer_call_fn_5790953´
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
Ð2Í
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790943
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790938´
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
Ö2Ó
,__inference_dense_3404_layer_call_fn_5790970¢
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
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790963¢
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
2
.__inference_dropout_1358_layer_call_fn_5791005
.__inference_dropout_1358_layer_call_fn_5791000´
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
Ð2Í
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790995
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790990´
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
Ö2Ó
,__inference_dense_3405_layer_call_fn_5791023¢
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
G__inference_dense_3405_layer_call_and_return_conditional_losses_5791016¢
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
%__inference_signature_wrapper_5790509dense_3399_input
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
 ©
G__inference_dense_3402_layer_call_and_return_conditional_losses_5790859^230¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_1356_layer_call_fn_5790896Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dense_3403_layer_call_fn_5790918Q<=0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_3403_layer_call_and_return_conditional_losses_5790911^<=0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_3399_layer_call_fn_5790745P/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dense_3400_layer_call_fn_5790762Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÉ
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790349z()23<=FGPQA¢>
7¢4
*'
dense_3399_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_1357_layer_call_fn_5790948Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dropout_1357_layer_call_fn_5790953Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¿
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790638p()23<=FGPQ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¡
0__inference_sequential_685_layer_call_fn_5790432m()23<=FGPQA¢>
7¢4
*'
dense_3399_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dropout_1354_layer_call_fn_5790792Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dense_3401_layer_call_fn_5790814Q()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_685_layer_call_fn_5790709c()23<=FGPQ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÉ
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790381z()23<=FGPQA¢>
7¢4
*'
dense_3399_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_1358_layer_call_fn_5791000Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dropout_1354_layer_call_fn_5790797Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ«
"__inference__wrapped_model_5789932()23<=FGPQ9¢6
/¢,
*'
dense_3399_inputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

dense_3405$!

dense_3405ÿÿÿÿÿÿÿÿÿ«
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790891^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_1356_layer_call_and_return_conditional_losses_5790886^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_1358_layer_call_fn_5791005Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_3400_layer_call_and_return_conditional_losses_5790755^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_sequential_685_layer_call_fn_5790728c()23<=FGPQ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_dropout_1355_layer_call_fn_5790844Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_dense_3404_layer_call_fn_5790970QFG0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790938^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_1357_layer_call_and_return_conditional_losses_5790943^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 Â
%__inference_signature_wrapper_5790509()23<=FGPQM¢J
¢ 
Cª@
>
dense_3399_input*'
dense_3399_inputÿÿÿÿÿÿÿÿÿ"7ª4
2

dense_3405$!

dense_3405ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_3401_layer_call_and_return_conditional_losses_5790807^()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_1355_layer_call_fn_5790849Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_dense_3404_layer_call_and_return_conditional_losses_5790963^FG0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
G__inference_dense_3399_layer_call_and_return_conditional_losses_5790738]/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¿
K__inference_sequential_685_layer_call_and_return_conditional_losses_5790690p()23<=FGPQ7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_dropout_1356_layer_call_fn_5790901Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¡
0__inference_sequential_685_layer_call_fn_5790484m()23<=FGPQA¢>
7¢4
*'
dense_3399_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¨
G__inference_dense_3405_layer_call_and_return_conditional_losses_5791016]PQ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_3405_layer_call_fn_5791023PPQ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790782^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_dense_3402_layer_call_fn_5790866Q230¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ«
I__inference_dropout_1354_layer_call_and_return_conditional_losses_5790787^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790834^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790990^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_1355_layer_call_and_return_conditional_losses_5790839^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 «
I__inference_dropout_1358_layer_call_and_return_conditional_losses_5790995^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 