import clingo


class Context:
    def id(x):
        return x

    def seq(x, y):
        return [x, y]


def main(prg):
    prg.ground([("base", [])], context=Context())
    prg.solve()

def on_model(m):
    print(m)

if __name__ == "__main__":
    ctl = clingo.Control()
    ctl.add("base", [], '''
    p(@id(10)).
    q(@seq(1,2)).
    ''')
    ctl.ground([("base", [])], context=Context())
    ctl.solve(on_model=on_model)

