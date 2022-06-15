

def extract_nums(text):
    for item in text.split(','):
        try:
            yield float(item)
        except ValueError:
            pass

text = str(input())

print(list(extract_nums(text)))