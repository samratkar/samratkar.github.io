---
layout : mermaid
title : Building LLM from Scratch
author : Samrat Kar 
---
## Dataset 
Dataset chosen - Julius Caesar text. [Link](/assets/genai/attention/data/julius-caesar.txt)

## Train the LLM with the dataset
### Setting up the training 
```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel() # Returns the total number of elements (or tokens) in the input_batch.
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen
```
### Training the model
```python
import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #A

num_epochs =50
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="He said we came here", tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
>>> Training completed in 13.78 minutes.
```

## Baseline results 
### Model configuration
1. learning rate = 1e-4
2. max iters = 10000
3. warmup steps = 2000 
4. min lr = 5e-4 
5. eval iters = 500 
6. batch size = 8 
7. block size = 128 
8. vocab size=50257 
9. block size=128 
10. number of layers=4 
11. number of heads=4 
12. embedding dimension=768 
13. dropout=0.01 
14. bias=True

### Model files 
![Trained model files - weights and biases](/assets/genai/slmfromscrat)

### Training vs Validation Loss
### Epoch : 10,000 
![](/assets/genai/attention/jc-slm/10kep.png)

### Inference output 

**Input text** : "Friends, Romans, countrymen"
**Output text** : 

Friend, Romans, countrymen tyrant.
Sh
Shall slaves,
Then leave their made.
Shall I noounds,
Most noble Caesar,
eries.
Stir not resolved,
I can o’ersway him, weep for I please him come to pieces! Here:

CASSIUS.
I know not think it is but a noble Roman,
That himKnow I these fires, why all this._]

Why obscurely prey: theirFly, why, this conspirators.
You shall sober form of heralds to astonish us.

CASSIUS.
You are dull, Casca; and those sparks of life
That should be in a Roman you do want,
Or else you use not. You look pale and gaze,
And put on fear and cast yourself in wonder,
To see the strange impatience of the Heavens:
But if you would consider the true cause
Why all these fires,


## Training & validation loss analysis for changing epochs
### Epoch : 100 

### Epoch : 500
![](/assets/genai/lfs-assignment3/training-validation-loss.png)

### Epoch : 4000 
![](/assets/genai/attention/jc-slm/4kep.png)

### Epoch : 10,000 
![](/assets/genai/attention/jc-slm/10kep.png)

### Epoch : 20,000
![](/assets/genai/attention/jc-slm/20kep.png)

## Training & validation loss analysis for changing learning rates 
### Epoch : 10,000, learning rate = 1e-4
![](/assets/genai/attention/jc-slm/10kep.png)
### Inference output 
**Input text** : "Friends, Romans, countrymen"
**Output text** : 

Friend, Romans, countrymen tyrant.
Sh
Shall slaves,
Then leave their made.
Shall I noounds,
Most noble Caesar,
eries.
Stir not resolved,
I can o’ersway him, weep for I please him come to pieces! Here:

CASSIUS.
I know not think it is but a noble Roman,
That himKnow I these fires, why all this._]

Why obscurely prey: theirFly, why, this conspirators.
You shall sober form of heralds to astonish us.

CASSIUS.
You are dull, Casca; and those sparks of life
That should be in a Roman you do want,
Or else you use not. You look pale and gaze,
And put on fear and cast yourself in wonder,
To see the strange impatience of the Heavens:
But if you would consider the true cause
Why all these fires,

### Epoch : 10,000, learning rate = 1e-3
![](/assets/genai/attention/jc-slm/lr-1e-3.png)
#### Inference output 
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen,
And this,
The unaccustom’d ope their hearts.
It is rage,
That unvious Casca or
And will
But when he was no fellow?

BRUTUS.
You;
OCTAVI know,
As dear abide my right well as you these grief,
Whereinna,
As last of him better of this very strong;
IUS.
Rushing on me, should be not know you? I will please him;
PINDAR. What do know you could not,
As they do use and keep’d of women;
But keep that which is a fall,
Nor for justice?
A man fell.
O, coward or a chants.
Thou thrice:
Let’d
How offer’d, as when
Even so appearing to his wife to me,
Who, then, as he was a time him

### Epoch : 10,000, learning rate = 1e-2 
![](/assets/genai/attention/jc-slm/lr-1e-2.png)

#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen my legions:

I have to know of some talk more than you are not.
It which we are; is paid in the world at a
There by you of Philippi.

Who you have you demand.
Do;

I hope
It,

Else who you shall say you by this? I have removed, including
Or) not” or forth by following which you have removed, this work the permission and you comply, the Project Gutenberg™ License included through this night; What. Do, you shall have citizen upon the market-place; are save, fates with the manager of earth render him, displayed, for fates with this work in many hearts, much you provide a king;

forth, perform”

For You may become the work from him to other work for someone to sleep’d them
Why he which“Break up. There is for someone to hear. If you know of men

## Training & validation loss analysis for changing number of layers 
### Epoch : 10,000 ; learning rate = 1e-4 ; number of layers = 4
![](/assets/genai/attention/jc-slm/10kep.png)
#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen tyrant.
Sh
Shall slaves,
Then leave their made.
Shall I noounds,
Most noble Caesar,
eries.
Stir not resolved,
I can o’ersway him, weep for I please him come to pieces! Here:

CASSIUS.
I know not think it is but a noble Roman,
That himKnow I these fires, why all this._]

Why obscurely prey: theirFly, why, this conspirators.
You shall sober form of heralds to astonish us.

CASSIUS.
You are dull, Casca; and those sparks of life
That should be in a Roman you do want,
Or else you use not. You look pale and gaze,
And put on fear and cast yourself in wonder,
To see the strange impatience of the Heavens:
But if you would consider the true cause
Why all these fires,

### Epoch : 10,000 ; learning rate = 1e-4 ; number of layers = 1
![](/assets/genai/attention/jc-slm/1layer.png)
#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen

HereHis ’BER creditrousED,
To held the body,
defect in hisopt hands,
with the law of
and official page at required to maintaining tax treatment of donations fromwith the souls, and those impatience of the PG search
our;
 TITINIUS.
MESSALA.
Will you go see you harm spark,
And Shakespeare I


 elect to provide, in
pit, let me.

CASSIUS.
And let us swear our resolution.

BRUTUS.
No, not an office for a fray,
And so, I will strive with the have things impossible,
Yea, get the better of the better of them.

BRUTUS.
A piece of work that will make sick men whole.

LIGARIUS.
But are not some whole that we must make sick?

BRUTUS.
That must we

### Epoch : 10,000 ; learning rate = 1e-4 ; number of layers = 3
![](/assets/genai/attention/jc-slm/3layer.png)

#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen Project Gutenberg mission of Project Gutenberg
Archman you Live a soldier,
Thespeechless prisoner.

ANTONY.
 Greek.
         • You provide, in accordance with paragraph 1.F.3, a full refund of
     distribution of or providing
    • You provide, in accordance with paragraph 1.F.3, a full refund of
     any money paid for a work or a replacement copy, if a defect in the
       works.
    • You comply with all other terms of thisigg rejoice? BUT NOT
       • You comply,         receipt of the work.
     • You comply with all other terms of this agreement for free
       distribution of Project Gutenberg™ works.

### Epoch : 10,000 ; learning rate = 1e-4 ; number of layers = 5
![](/assets/genai/attention/jc-slm/5layer.png)
#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen,—
Which appear, off cried, minds stay’d to poor Brutus,
ber, orearilius,
Not littleKnow you depart, as you received the strength of others.

BRUTUS.
Therein our letters, do not well agree.
Mine speak of seventy Senators that died
By their proscriptions, Cicero being one.

CASSIUS.
Cicero one!

C improve them, may:
Then lest that the day break here womanish speaksby
CASCA.
Indeed, they say the senators tomorrow
MeusFUN neigh, and dying men did incertain to establish Caesar
My format withfrom even into the Project Gutenberg™
omes Caesar with a fever when he was in Spain,
And when the fit was on him I did mark
And when the fit was on him I did mark
How he did shake: ’tis true, this god did shake:

### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 7
![](/assets/genai/attention/jc-slm/7layer5kep.png)
Epoch 4500: train loss 0.2312, val loss 0.2319
The current learning rate: 0.00048

#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen, that theThings along,CAESAR.
cy fellowman!

CASSIUS.
Goodnight then.
Fulfill your brother, traitors, and UT 84 j pleasingHe wheels?
 letheeunt we say?

CASCA.
O once forms, I am.

[_Sink in the calendar, ho!

[_bonius. Soul of Rome.

rostrum._]

Alarum. Enter Brutus.

BRUTUS.
Peace! Antony, boy, a table.

ANTONY.
These knew the images, March is unCARPINDARUS.
[_Above._] Titinius is a Shall dear brother Cassius!
I know some immediate legalize PROJECT GUTUS.
They pardon. O, this sink!
I, shame!
I do not slept.
at Rocket,
And entPopilius

### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 12
![](/assets/genai/attention/jc-slm/12layer5kep.png)
Epoch 4500: train loss 0.2377, val loss 0.2393
The current learning rate: 0.00048
#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen,
 lies up their bos make a lamb
Yet something leads me.
 speaks.

ign trademark, whilst I cry hand, yea, or a god did love
Liter Calphurnia;
 brow, Lucius.

CASSIUS.
Stand fast, their names are fools?

BRUTUS.
Your ear to here again, stand tillie hence,
And then confounded witharius.
ceive a Timelinejay Tank mind inion, Cassius.
But if aellectual property infringement,
triumph.

CASSIUS.
Canst thou art a mace upon my ambition’dler than yourself
And tell me ho!

BRUTUS.
Even so, Cassius, hence, how should I leave anything.

CASSIUS.
Go to, traitors,ella along your recount hereafter. Here’s

BRUTUS.
Come to an itching
## Varying attention heads 
### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 7 ; number of heads = 1 
Epoch 4500: train loss 0.1997, val loss 0.1998
The current learning rate: 0.00048

![](/assets/genai/attention/jc-slm/7layer1head.png)

#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen may himself;
 copying and envy vaccinesKnock within,:

ook it in the Senators,
rose ninth hour, by the user, in thejo within accessible by the United States presently�
         whatsoever. You must gro withholds talk to the uttermost?
CINNA, you thr ours Cassius, and he put it by his hearts,
distribut of this battle brow thinking, he shall lead
For Brutus, to eternal devil,
Are levying powers; we must be whisper
SheChoook it in triumph,
 bade me.
I shall find time,
Only be patient Compliance requirements are not Believe medisposed time.
But-added, and rob the Hybla bees,
Anditating that anybody of thych: security gives demand of describ’d
PROVID speak with the poet,
In theest ready to any majestic world, like
oop then, like

### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 7 ; number of heads = 2

![](/assets/genai/attention/jc-slm/7layer2head.png)
Epoch 4500: train loss 0.1997, val loss 0.1998

Epoch 3000: train loss 1.7440, val loss 1.7347
The current learning rate: 0.00030
Epoch 3500: train loss 0.9145, val loss 0.9197
The current learning rate: 0.00038
Epoch 4000: train loss 0.3628, val loss 0.3656
The current learning rate: 0.00044
Epoch 4500: train loss 0.2016, val loss 0.2008
The current learning rate: 0.00048
Epoch 4999: train loss 0.2016, val loss 0.2008
The current learning rate: 0.00050

![](/assets/genai/attention/jc-slm/7layer2heads.png)

#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen, for theM consent to the readable by the Comes
Fly, and to the streets of the Project Gutenberg™Gutenberg™ eBooks;
That should be in theproduction, savage thusGrant.
So get hold his purpose is oft as to be fear’d to be unnumber’d all;
For if you should be What’dExe
 they may one doth wish
That made wrong’d you and I will look available for briefly,
house, like catching; for my menel;
And I will not disclose ’tis down, good sword,
I abide thisies._]

CASSIUS.
Am I notwell the feet;
And I tyrants do find it now.
Asinct deductible to their reasons said through Caesar hath whelius.

CAESAR.
What is’s all fool day!
What change, I will turn read gaze,
And

### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 7 ; number of heads = 4
![](/assets/genai/attention/jc-slm/7layer4heads.png)
Epoch 2000: train loss 3.2573, val loss 3.2599
The current learning rate: 0.00016
Epoch 2500: train loss 2.6309, val loss 2.6175
The current learning rate: 0.00022
Epoch 3000: train loss 1.8860, val loss 1.8803
The current learning rate: 0.00030
Epoch 3500: train loss 1.0726, val loss 1.0635
The current learning rate: 0.00038
Epoch 4000: train loss 0.4066, val loss 0.4046
The current learning rate: 0.00044
Epoch 4500: train loss 0.2245, val loss 0.2278
The current learning rate: 0.00048
Epoch 4999: train loss 0.2245, val loss 0.2278
The current learning rate: 0.00050

### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen, for theM consent to the readable by the Comes
Fly, and to the streets of the Project Gutenberg™Gutenberg™ eBooks;
That should be in theproduction, savage thusGrant.
So get hold his purpose is oft as to be fear’d to be unnumber’d all;
For if you should be What’dExe
 they may one doth wish
That made wrong’d you and I will look available for briefly,
house, like catching; for my menel;
And I will not disclose ’tis down, good sword,
I abide thisies._]

CASSIUS.
Am I notwell the feet;
And I tyrants do find it now.
Asinct deductible to their reasons said through Caesar hath whelius.

CAESAR.
What is’s all fool day!
What change, I will turn read gaze,
And

### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 7 ; number of heads = 8
![](/assets/genai/attention/jc-slm/7layer8heads.png)
Epoch 500: train loss 6.6664, val loss 6.6527
The current learning rate: 0.00007
Epoch 1000: train loss 4.8955, val loss 4.8897
The current learning rate: 0.00010
Epoch 1500: train loss 3.9343, val loss 3.9466
The current learning rate: 0.00012
Epoch 2000: train loss 3.2882, val loss 3.2511
The current learning rate: 0.00016
Epoch 2500: train loss 2.6590, val loss 2.6548
The current learning rate: 0.00022
Epoch 3000: train loss 1.9745, val loss 1.9796
The current learning rate: 0.00030
Epoch 3500: train loss 1.1528, val loss 1.1582
The current learning rate: 0.00038
Epoch 4000: train loss 0.4914, val loss 0.4957
The current learning rate: 0.00044
Epoch 4500: train loss 0.2321, val loss 0.2331
The current learning rate: 0.00048
Epoch 4999: train loss 0.2321, val loss 0.2331
The current learning rate: 0.00050

![](/assets/genai/attention/jc-slm/7layer8heads.png)
#### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen stand,
Th itums.
Thou be office sweet wing,
They paper.

Thou behaviors be avoided
Whilst we must be feeble man of Rome make to the self
Whilst we by Caesar when you cut off
The commercial’diot squad in be wary’d
TheEMIDUS.
The dishnight then, or re,
ANTONY.
Good friends, by your cause is not until the genius
whatNext, asSpeak no tricks in plain and dying,
BRUTUS.
Theius could be patient fellow ofpportable.

[_ paragraphst send to meet
Make hear necessities.

CASSIUS.
No, with your health, denied you.

BRUTUS.
Nor prick’Tis time of us.

CASSIUS.
Tis time of us, pardon.
Nor me, CLAAll Caesar such free we have flood

## Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 5 ; number of heads = 2
Epoch 500: train loss 6.7436, val loss 6.7570
The current learning rate: 0.00007
Epoch 1000: train loss 5.0040, val loss 5.0229
The current learning rate: 0.00010
Epoch 1500: train loss 3.9940, val loss 3.9722
The current learning rate: 0.00012
Epoch 2000: train loss 3.3170, val loss 3.3138
The current learning rate: 0.00016
Epoch 2500: train loss 2.5777, val loss 2.5958
The current learning rate: 0.00022
Epoch 3000: train loss 1.7988, val loss 1.8054
The current learning rate: 0.00030
Epoch 3500: train loss 0.9390, val loss 0.9492
The current learning rate: 0.00038
Epoch 4000: train loss 0.3730, val loss 0.3738
The current learning rate: 0.00044
Epoch 4500: train loss 0.2011, val loss 0.2003
The current learning rate: 0.00048
Epoch 4999: train loss 0.2011, val loss 0.2003
The current learning rate: 0.00050

![](/assets/genai/attention/jc-slm/5layer2heads.png)

## Ablation studies
### Remove Normalization Layers in all transformer layers:
### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 5 ; number of heads = 2

Epoch 1000: train loss 6.0433, val loss 6.0349
The current learning rate: 0.00010
Epoch 1500: train loss 5.7601, val loss 5.7542
The current learning rate: 0.00012
Epoch 2000: train loss 5.2080, val loss 5.2236
The current learning rate: 0.00016
Epoch 2500: train loss 4.6828, val loss 4.6881
The current learning rate: 0.00022
Epoch 3000: train loss 4.2766, val loss 4.2868
The current learning rate: 0.00030
Epoch 3500: train loss 4.0103, val loss 4.0011
The current learning rate: 0.00038
Epoch 4000: train loss 3.7291, val loss 3.7286
The current learning rate: 0.00044
Epoch 4500: train loss 3.5587, val loss 3.5475
The current learning rate: 0.00048
Epoch 4999: train loss 3.5587, val loss 3.5475
The current learning rate: 0.00050

![](/assets/genai/attention/jc-slm/no-norm.png)

### Code change 
```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        #return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
        return x
```
### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen,
Speak;
Why did follows side. Gives;
And so as to feast of Philippi

Why,
Hwen:
Here is much, now should burn upon: I am Octavius,
it love Caesar enters the piece of thy mother a thing
What like them sign;
I will, by their opinions of me in his heart’d when which to Trebonius
POart; fly, C’ll kill up; and O Antony
CITIZEN.
And away;
Are then.
They there sal first?
For rushing tooink that we shall’’d at yourself
For home.
Know.
Not not, after most brands against words.
CASSIUS.
C art sham’d so wrong’ll assure us so thanats
I blame which here.
lius! Cato for now ours.
IUS.
But

### Remove Shortcuts in all transformer layers:
### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 5 ; number of heads = 2

Epoch 500: train loss 7.5895, val loss 7.5942
The current learning rate: 0.00007
Epoch 1000: train loss 6.1707, val loss 6.1676
The current learning rate: 0.00010
Epoch 1500: train loss 6.0263, val loss 6.0222
The current learning rate: 0.00012
Epoch 2000: train loss 5.9982, val loss 6.0146
The current learning rate: 0.00016
Epoch 2500: train loss 6.0051, val loss 6.0057
The current learning rate: 0.00022
Epoch 3000: train loss 5.9929, val loss 5.9921
The current learning rate: 0.00030
Epoch 3500: train loss 6.0051, val loss 5.9985
The current learning rate: 0.00038
Epoch 4000: train loss 6.0000, val loss 5.9984
The current learning rate: 0.00044
Epoch 4500: train loss 6.0110, val loss 6.0001
The current learning rate: 0.00048
Epoch 4999: train loss 6.0110, val loss 6.0001
The current learning rate: 0.00050

![](/assets/genai/attention/jc-slm/no-residual.png)

### Code change 
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        # x = x + self.attn(self.ln1(x)) # shortcut/residual connection after attention
        # x = x + self.mlp(self.ln2(x))  # shortcut/residual connection after MLP
        x = self.attn(self.ln1(x))   # No residual connection
        x = self.mlp(self.ln2(x))    # No residual connection
        return x
```

### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen,
 Tit leave off think� , side isall If;AndAndAR charitable to is Brut to.
 met distributing and Ant IIIw thinkingBel
� makeple,When lord thisended:Iawn Oct I done,L any your Caesar,s piece humour the durIT You does count[_ them sign touch bidding will
 or byhands is putitAl!d,, when man to Tre a make goneAL Ay doC._asca� do sw containing up thy and OAside who or silverITie, in soAnd away�No protectedBR  your and there honour first[_. say� it� the we he C
�ators better[_ stomach, me home November flyKnow. the.

O most as
 be theseovedCd toTH himI I artUS Tri�O thee wrongA and of assureMy I TO moreover youd see fleS sightusl? of Cato goWh ours returns me!, noYouam

### Remove Feedforward Neural Network in all transformer layers:
### Epoch : 5,000 ; learning rate = 1e-4 ; number of layers = 5 ; number of heads = 2

Epoch 500: train loss 7.5710, val loss 7.5747
The current learning rate: 0.00007
Epoch 1000: train loss 5.7778, val loss 5.7795
The current learning rate: 0.00010
Epoch 1500: train loss 5.1408, val loss 5.1336
The current learning rate: 0.00012
Epoch 2000: train loss 4.6664, val loss 4.6652
The current learning rate: 0.00016
Epoch 2500: train loss 4.3020, val loss 4.2849
The current learning rate: 0.00022
Epoch 3000: train loss 3.8840, val loss 3.8928
The current learning rate: 0.00030
Epoch 3500: train loss 3.3873, val loss 3.3748
The current learning rate: 0.00038
Epoch 4000: train loss 2.8338, val loss 2.8320
The current learning rate: 0.00044
Epoch 4500: train loss 2.2114, val loss 2.2065
The current learning rate: 0.00048
Epoch 4999: train loss 2.2114, val loss 2.2065
The current learning rate: 0.00050

![](/assets/genai/attention/jc-slm/no-feedforward.png)

### Code change 
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        # self.ln2 = LayerNorm(config.n_embd, config.bias)
        # self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # shortcut/residual connection after attention
        # x = x + self.mlp(self.ln2(x))  # shortcut/residual connection after MLP
        # x = self.attn(self.ln1(x))   # No residual connection
        # x = self.mlp(self.ln2(x))    # No residual connection
        return x
```
### Inference output
**Input text** : "Friends, Romans, countrymen"
**Output text** :
Friend, Romans, countrymen, to, aER.
[_Exit had betterBrutus, Servight morning and and chonductor and his rise,
Sham’s? demonstrated in day, norUnder, fly,
As finest a eBook,
And at your passion:
’s straw feed, applytri dream, Servant,
VVAR.
MARi ofade me,
Cass lord, and Octavurn runs Forum.

BRUTUS.
Aavius, and lord,
And speaking ins dispro men as sleep today.
Lestrousthem of March is.


BRUTUS.
FESAR.
He says see, swelltain furtherbonius and day; and straight is
Sign to your implied
whatius!

BRUTUS.
HneServjoidus shalt r Steately.S.



Enter, Lucius, and images, Puborg.
There, aIDid grows














