This aims to try and use neural nets to predict next pick/ban in dota 2 draft.

Named after Winter for his insanely insightful and predictive draft analysis

Uses keras/python3. pip install -r requirements.txt 'should' be all you need to get it working.

However scipy uses quite a lot of packages like blas/lapack that arent installed by default on most OS's.
These need to be isntalled by default package manager....or if on WIndows...erm..google it?

Yes code kind of sucks in places, I could do with making a Neural net class and then RNN and vanilla NN subclasses
commonise the next_pick for rnn's and normal nn's etc.

but writing good quality code is effort and time for a one-off script/model. If I wasnt moving house soon I'd tidy it up,
however if you want to re-write in a more readable/usable fashion, Ill happily check and accept any changes/pull requests