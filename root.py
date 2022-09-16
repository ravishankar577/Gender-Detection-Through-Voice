from flask import Flask, render_template, redirect, url_for
import random
import predict


col = {}
mines = [16, 48, 82]
n = 104
flag = False

app = Flask(__name__)


@app.route('/')
def main():
    global flag
    flag = False

    choice = [i for i in range(9)]
    weights = [2]
    weights.extend([1 for _ in range(8)])

    for i in range(n):
        col[i] = random.choices(choice, weights=weights)[0]

    return render_template('index.html', n=n, col=col, pred=None, enumerate=enumerate, mine=mines)


@app.route('/predict')
def pred():
    global flag

    if flag:
        return redirect(url_for('main'))

    predfinal = predict.main()

    flag = True
    return render_template('index.html', n=n, col=col, pred=predfinal[0], enumerate=enumerate, mine=mines)


if __name__== '__main__':
    app.run(debug=True, host='localhost')
