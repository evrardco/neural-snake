# Neural snake
## What is this ?
I wanted to put in practice what I learned in my various AI courses by doing an apparently simple snake AI.
## Which AI technique is used ?
A simple deep neural network with an auto encoding layer, linear layers separated by ReLU activations functions.
## Does it work ?
Sort of, currently you have to run it, wait for it to gather data and train and hope for the best. I didn't add the stuff to make it reproducible yet.

You can expect it to reach 10 points in a single game by 30000 states recorded. It's not too slow (20 minutes) on my crappy laptop so you shouldn't have to wait for too long. 

## How clean is the code ?
The snake part (game, graphics, keyboard management, etc) is pretty clean even though it's uncommented.

The AI part is however undocumented, dirty and unreadable at the moment.
## Dependencies
- Julia
- Python 3.8.6
- PyJulia
- Pytorch
- pygame
- Both Julia and Python in the PATH.
## How to run
Follow the procedure of PyJulia configuration. 
### Windows
In the powershell, from the project's directory, run 
```
python .\game.py 
```
### Linux
I tested it on ubuntu 18.04, it might be different on other systems.
In the terminal, from the project's directory, run
```
python3 game.py
```