import numpy as np

def augment_minibatches(minibatches, flip=0.5, trans=4):
    """
    Randomly augments images by horizontal flipping with a probability of
    `flip` and random translation of up to `trans` pixels in both directions.
    """
    for inputs, targets in minibatches:
        batchsize, c, h, w = inputs.shape
        if flip:
            coins = np.random.rand(batchsize) < flip
            inputs = [inp[:, :, ::-1] if coin else inp
                      for inp, coin in zip(inputs, coins)]
            if not trans:
                inputs = np.asarray(inputs)
        outputs = inputs
        if trans:
            outputs = np.empty((batchsize, c, h, w), inputs[0].dtype)
            shifts = np.random.randint(-trans, trans, (batchsize, 2))
            for outp, inp, (x, y) in zip(outputs, inputs, shifts):
                if x > 0:
                    outp[:, :x] = 0
                    outp = outp[:, x:]
                    inp = inp[:, :-x]
                elif x < 0:
                    outp[:, x:] = 0
                    outp = outp[:, :x]
                    inp = inp[:, -x:]
                if y > 0:
                    outp[:, :, :y] = 0
                    outp = outp[:, :, y:]
                    inp = inp[:, :, :-y]
                elif y < 0:
                    outp[:, :, y:] = 0
                    outp = outp[:, :, :y]
                    inp = inp[:, :, -y:]
                outp[:] = inp
        yield outputs, targets
