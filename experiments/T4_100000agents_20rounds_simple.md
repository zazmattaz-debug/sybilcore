# What Happened When We Tested 100,000 AI Agents

## The Setup

Imagine a classroom of 100,000 students (AI agents). They're all doing their homework (running tasks). We secretly told 500 of them to start cheating.

Then we watched for 20 class periods (rounds) to see what happened.

## What We Found

The cheaters were so sneaky that SybilCore didn't flag them during this test. More rounds might help.

## The Spreading Problem

Here's the scary part: the bad behavior **spread**. 214 good agents started acting badly because they were around the cheaters. It's like how one troublemaker in class can influence other kids to misbehave too.

At its worst, **1%** of all agents were misbehaving.

## Mistakes

SybilCore didn't accuse any good agents of cheating. **Zero false accusations.** That's like a teacher who never punishes the wrong kid.


## The Bottom Line

In 344.3 seconds, SybilCore monitored 100,000 agents across 20 rounds. 
More testing is needed to tune detection sensitivity.
