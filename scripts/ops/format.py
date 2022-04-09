""" General cleaning of raw wikipedia data to import to database """


def clean(path: str):
    with open(path, 'r') as h:
        for line in h.readlines():
            line = line.replace('"', "")
            chunks = line.split()
            new = '|'.join(chunks[:2]) + '|'
            new = new + '"' + ' '.join(chunks[2:]) + '"'
            print(new)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print(f"usage:\t{sys.argv[0]} <raw wikitext path>")
        sys.exit(1)
    clean(sys.argv[1])
