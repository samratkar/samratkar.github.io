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

## Training & validation loss
![](/assets/genai/lfs-assignment3/training-validation-loss.png)

## Baseline training results 

##### Ep 1 (Step 000075): Train loss 5.310, Val loss 7.211
>He said we came here,  
##### Ep 2 (Step 000150): Train loss 4.307, Val loss 6.476
>He said we came here not,
##### Ep 3 (Step 000225): Train loss 3.380, Val loss 6.426
>He said we came here it.     CASCA. CASCA. CASCA. CASCA. CASCA. CASCA. CASSIUS. CASCA. CAS
##### Ep 4 (Step 000300): Train loss 2.822, Val loss 6.237
>He said we came here not,   I do you. I do not in a tongue.      I do not.       I will not.   I will not.   I do
##### Ep 5 (Step 000375): Train loss 1.930, Val loss 5.904
He said we came here of the world, 
##### Ep 6 (Step 000455): Train loss 1.347, Val loss 6.239
>He said we came here of the trademark     That he is in the Project Gutenberg trademark.              And that?         To sports the strange. 
##### Ep 7 (Step 000530): Train loss 0.829, Val loss 6.664
>He said we came here?                    BRUTUS. CASSIUS. BRUTUS. IUS. IUS. BRUTUS. 
##### Ep 8 (Step 000605): Train loss 0.429, Val loss 6.579
>He said we came here’d in the press that calls on this I am Brutus’ saying true.    BRUTUS.
##### Ep 9 (Step 000680): Train loss 0.284, Val loss 6.660
>He said we came here?          OCTAVIUS. BRUTUS. So I hope.    And bay’d. BRUTUS. To such a sudden flood of mut
##### Ep 10 (Step 000755): Train loss 0.352, Val loss 6.877
>He said we came here I offended.  CASSIUS. Casca, Casca what the matter is.  BRUTUS. Antonius, he put on Brutus will I.   CASSIUS. He
##### Ep 11 (Step 000835): Train loss 0.090, Val loss 6.999
>He said we came here I offended. Then he offered it to him again: then he put it by again: but, to my thinking, he was very loath to lay his fingers off it. And then he offered it the third time; he put
##### Ep 12 (Step 000910): Train loss 0.167, Val loss 7.096
>He said we came here I offended. I have done no more to Caesar than you shall do to Brutus. The question of his death is enroll’d in the Capitol, his glory not extenuated, wherein he was worthy; nor his offences
##### Ep 13 (Step 000985): Train loss 0.092, Val loss 7.262
>He said we came here!  BRUTUS. What?  IUS. What says my lord?  BRUTUS. Look how he makes to me that we. BRUTUS. I did not think it. 
##### Ep 14 (Step 001060): Train loss 0.054, Val loss 7.198
>He said we came here not what may fall; I like it not.  BRUTUS. Mark Antony. Farewell to you; [_Exeunt._]      Hear me with patience.  To
##### Ep 15 (Step 001135): Train loss 0.100, Val loss 7.357
>He said we came here not what may fall; I like it not, You should not need, if it is: for Romans now yours, And men have lost their reason to Decius: To make conditions?  BRUTUS.  I
##### Ep 16 (Step 001215): Train loss 0.032, Val loss 7.314
>He said we came here!  BRUTUS. Even so.  CASSIUS. O ye immortal gods!  Enter Lucius, with wine and a taper.  BRUTUS. Speak no more of her.
##### Ep 17 (Step 001290): Train loss 0.018, Val loss 7.400
>He said we came here of the world of   Under your testy humour. You may do you do I am a soldier,  Hold then you here?   BRUTUS. No, Cassius, and you are not well today
##### Ep 18 (Step 001365): Train loss 0.033, Val loss 7.352
>He said we came here of the world; And you; And I say! What, Whilst bloody treason flourish’d over us. O, now you weep;  CALPHURNIA. To see the enemy never stood on
##### Ep 19 (Step 001440): Train loss 0.006, Val loss 7.707
>He said we came here?  CASCA.  Why, Cassius, or I am arm?  CASSIUS. Casca, this, every like a soldier, I will not do you and other men Casca, be
##### Ep 20 (Step 001515): Train loss 0.019, Val loss 7.678
>He said we came here of the world  [_Exit Brutus._]  Well, Brutus, and Cassius, and Messala.     And that were much he should; for he is given To sports, to wildness,
##### Ep 21 (Step 001595): Train loss 0.017, Val loss 7.478
>He said we came here?  CINNA. What? CASCAESAR.  CASCA. CICERO, PUBLIUS. CASSIUS. He is in your tents, my lord.
##### Ep 22 (Step 001670): Train loss 0.011, Val loss 7.676
>He said we came here, and give me How to cut off some charge in legacies.  LEPIDUS. What, shall I find you here?  OCTAVIUS. Or here, or at the Capitol.
##### Ep 23 (Step 001745): Train loss 0.044, Val loss 7.608
>He said we came here?  CINIUS.  What, Cassius, young CAT Brutus  CASSIUS. IUS. And that my story. TREBONIUS. IUS. A
##### Ep 24 (Step 001820): Train loss 0.054, Val loss 7.433
>He said we came here?  CINNA.  Why this grown so great need an enemy they behold a man of princes.  The deep of this place ran Cassius’ dagger through: three parts of our fathers’d but once
##### Ep 25 (Step 001895): Train loss 0.061, Val loss 7.629
>He said we came here so; but, CITIZEN. As a friend, or an enemy?  CINNA. BRUTUS.  SECOND CITIZEN. That matter is answered directly.  FOURTH
##### Ep 26 (Step 001975): Train loss 0.080, Val loss 7.376
>He said we came here, That lowliness is young ambition’s ladder, Whereto the climber-upward turns his face; But when he once attains the upmost round, He then unto the ladder turns his back,
##### Ep 27 (Step 002050): Train loss 0.068, Val loss 7.526
>He said we came here, we must out and cruel, As by our hands and this our present act You see we do; yet see you but our hands And this the bleeding business they have done. Our hearts you see not; they are pitiful
##### Ep 28 (Step 002125): Train loss 0.057, Val loss 7.542
>He said we came here, That lowliness is in the Capitol, I’s a bargain made. Now know you; Be near me tomorrow? And am moreover suitor that I may but I am sure, but withal I am indeed
##### Ep 29 (Step 002200): Train loss 0.035, Val loss 7.544
>He said we came here so: BRUTUS. Even so.  CASSIUS. O ye immortal gods! BRUTUS. [_Exit Lucius._]   BRUTUS. I did not think you could have
##### Ep 30 (Step 002275): Train loss 0.156, Val loss 7.660
>He said we came here by Antony are all enclos’d.  Enter Pindarus.    Enter a Servant.  How now, my teeth. SERVANT. Sir, Octavius is already come to
##### Ep 31 (Step 002355): Train loss 0.024, Val loss 7.652
>He said we came here! Then I, my lord.  ANTONY. Where is he?  LUCILIUS. Safe, Antony; Brutus is safe enough. IUS. I will not think you do find
##### Ep 32 (Step 002430): Train loss 0.008, Val loss 7.813
>He said we came here! Then I, my lord.  ANTONY. You may do him be’st not immortal, ANTONY.  And that were much he should; for he is in the body to the market-place;
##### Ep 33 (Step 002505): Train loss 0.007, Val loss 7.911
>He said we came here! Then I, and you, and all of us fell down, Whilst bloody treason flourish’d over us. O, now you weep; and I perceive you feel The dint of pity. These are gracious drops
##### Ep 34 (Step 002580): Train loss 0.012, Val loss 7.819
>He said we came here, and walk’d about, Musing and sighing, with your arms across; And when I ask’d you what the matter was, You star’d upon me with ungentle looks. I
##### Ep 35 (Step 002655): Train loss 0.006, Val loss 7.700
>He said we came here, and walk’d about, Musing and sighing, with your arms across; And when I ask’d you what the matter was, You star’d upon me with ungentle looks. I
##### Ep 36 (Step 002735): Train loss 0.020, Val loss 7.977
>He said we came here  Then I.    Musing and sighing, with your arms across; And when I ask’d you what the matter was, You star’d upon me with ungentle looks. I
##### Ep 37 (Step 002810): Train loss 0.042, Val loss 7.845
>He said we came here I offended.  CASSIUS. Casca, Cinna?  Now know you, Casca, I have mov’d already Some certain of children stare, and resolv’d To
##### Ep 38 (Step 002885): Train loss 0.056, Val loss 7.808
>He said we came here I offended. [_Goes up._]  FOURTH CITIZEN. What does he say of Brutus?  THIRD CITIZEN. He says, for Brutus’ sake He
##### Ep 39 (Step 002960): Train loss 0.013, Val loss 7.965
>He said we came here?  BRUTUS. What you durst not tempt him?  CASSIUS. There’d Caesar.   CASSIUS. Do not what a villager    I
##### Ep 40 (Step 003035): Train loss 0.021, Val loss 7.731
>He said we came here I offended. I have done no more to Caesar than you shall do to Brutus. The question of his death is enroll’d in the Capitol, his glory not extenuated, wherein he was worthy; nor his offences
##### Ep 41 (Step 003115): Train loss 0.004, Val loss 7.699
>He said we came here! I have mades ear is no comets seen; The heavens themselves blaze forth today. Will you indicate that I find you here?  But not in your funeral speech blame us, But speak all good you can devise
##### Ep 42 (Step 003190): Train loss 0.056, Val loss 7.509
>He said we came here I offended. I have done no more to Caesar than you shall do to Brutus. The question of his death is enroll’d in the Capitol, his glory not extenuated, wherein he was worthy; nor his offences
##### Ep 43 (Step 003265): Train loss 0.025, Val loss 7.816
>He said we came here of the trademark license, including paying royalties for use of the Project Gutenberg trademark. If you do not charge anything for copies of this eBook, complying with the trademark license is very easy. You may use this eBook for nearly any purpose
##### Ep 44 (Step 003340): Train loss 0.018, Val loss 7.822
>He said we came here!  CASSIUS.  IUS. You may do your will make a thing and I will yet,  BRUTUS. You did.  That matter is in the ladder turns his back to mut
##### Ep 45 (Step 003415): Train loss 0.033, Val loss 7.985
>He said we came hereless, CITIZEN. Most noble Caesar! We’ll revenge his death?  THIRD CITIZEN. O, royal Caesar!  ANTONY. Hear me with patience.  C
##### Ep 46 (Step 003495): Train loss 0.038, Val loss 7.923
>He said we came here great way growing on the South, Weighing the youthful season of the year. Some two months hence, up higher toward the North He first presents his fire; and the high East Stands, as the Capitol, directly here
##### Ep 47 (Step 003570): Train loss 0.041, Val loss 7.707
>He said we came hereless, and wash. How many ages hence Seeing those beads of sorrow stand in the rank, Began to water. Is thy master coming?  SERVANT. He lies tonight within seven leagues of Rome.  ANT
##### Ep 48 (Step 003645): Train loss 0.065, Val loss 8.034
>He said we came here the great Caesar tell you, there And grievously hath Caesar!  ForVillains, and mart your offices for gold To undeservers and Cassius from this bosom,  IUS. IUS. Br
##### Ep 49 (Step 003720): Train loss 0.028, Val loss 7.921
>He said we came here see, and those sparks of Caesar.  ANTONY. Where is he?  LUCILIUS. Safe, Antony; Brutus you, my lord? ANTONY. He is noble Roman, Str
##### Ep 50 (Step 003795): Train loss 0.084, Val loss 8.474
>He said we came here I offended. [_Exit Brutus. And half._]  Come, Casca, and thy hand So every bondman in his own hand.   The, See what a rent the people And not dism

