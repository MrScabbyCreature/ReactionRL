from multiprocessing import Process, Manager


def asd(dicty):
    dicty["asd"] = [1, 2, 3]

def run(di):
    p = Process(target=asd, args=(di,))
    p.start()
    p.join()


if __name__ == "__main__":
    manager = Manager()
    d = manager.dict()

    run(d)

    print(d)
