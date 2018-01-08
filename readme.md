Blog post summarising what I done did and what I found:
https://medium.com/@jbknight07/winterbot9000-replacing-legendary-dota-draft-analyst-with-artificial-intelligence-68b9be169187

This aims to try and use neural nets to predict next pick/ban in dota 2 draft.

Named after Winter for his insanely insightful and predictive draft analysis

Uses keras/python3. pip install -r requirements.txt 'should' be all you need to get it working.

However scipy uses quite a lot of packages like blas/lapack that arent installed by default on most OS's.
These need to be isntalled by default package manager....or if on WIndows...erm..google it?

Yes code kind of sucks in places, I could do with making a Neural net class and then RNN and vanilla NN subclasses
commonise the next_pick for rnn's and normal nn's etc. it never got really past the prototyping/testing and into something that could be used in production.
